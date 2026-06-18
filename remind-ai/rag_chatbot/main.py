"""
=============================================================
Re:Mind — RAG 기반 심리 상담 챗봇 백엔드
=============================================================
확정 사항:
  - 대화 기록: SQLite 영속화 (session_id별)
  - LLM 컨텍스트: 최근 30턴
  - Response: {"answer": "...", "is_risk": bool}
  - 위험 감지:
      매 발화마다 독립적으로 LLM이 9번 슬롯 판단
      → 이번 발화 score >= 2 이면 is_risk=True
      → 누적/감쇠 없음. 발화 하나하나가 독립적
      RAG 응답 + 9번 슬롯 판단 병렬 호출 (응답 속도 유지)
  - is_risk=True: RAG 응답 버리고 위기 전용 LLM 응답 생성
      사용자 발화에 맞는 공감 + 109 등 전문가 연결 안내
  - /ai/report: SQLite 전체 기록 기반 PHQ-9 전체 분석
  - /ai/context: RAGAS 평가용 컨텍스트 검색 엔드포인트
=============================================================
"""

import os
import json
import sqlite3
import re
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv(dotenv_path="secrets.txt")

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 배포 시 백엔드 서버 IP로 변경
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
# 설정
# ─────────────────────────────────────────
DB_PATH        = "./chat_history.db"
HISTORY_LIMIT  = 30   # LLM 컨텍스트에 넣을 최근 턴 수


# ═══════════════════════════════════════════════════════════════
# 1. SQLite — 초기화 & CRUD
# ═══════════════════════════════════════════════════════════════

def init_db():
    """서버 시작 시 1회 실행 — 테이블 없으면 생성"""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT    NOT NULL,
                role       TEXT    NOT NULL,   -- 'user' | 'ai'
                content    TEXT    NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

def save_message(session_id: str, role: str, content: str):
    """대화 1턴 저장"""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content)
        )
        conn.commit()

def load_recent_history(session_id: str, limit: int = HISTORY_LIMIT) -> list[dict]:
    """최근 N턴 로드 (시간 오름차순)"""
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT role, content FROM chat_history
            WHERE session_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (session_id, limit)
        ).fetchall()
    # DESC로 가져온 뒤 뒤집어서 시간순 정렬
    return [{"role": r[0], "content": r[1]} for r in reversed(rows)]

def load_full_history(session_id: str) -> list[dict]:
    """/ai/report용 — 전체 대화 기록 로드"""
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT role, content FROM chat_history
            WHERE session_id = ?
            ORDER BY created_at ASC
            """,
            (session_id,)
        ).fetchall()
    return [{"role": r[0], "content": r[1]} for r in rows]

def load_history_by_date(session_id: str, start_date: str, end_date: str) -> list[dict]:
    """날짜 범위 기반 대화 기록 로드 (의사가 기간 지정할 때)"""
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT role, content FROM chat_history
            WHERE session_id = ?
            AND DATE(created_at) BETWEEN ? AND ?
            ORDER BY created_at ASC
            """,
            (session_id, start_date, end_date)
        ).fetchall()
    return [{"role": r[0], "content": r[1]} for r in rows]


# ═══════════════════════════════════════════════════════════════
# 2. 위험 감지 로직 — LLM 기반 PHQ-9 9번 슬롯 실시간 판단
# ═══════════════════════════════════════════════════════════════

CRISIS_NOTICE = (
    "\n\n---\n"
    "지금 많이 힘드신 것 같아서 걱정이 돼요. "
    "혼자 감당하기 너무 힘들 때는 전문가의 도움을 받는 것이 좋아요.\n"
    "자살예방상담전화 ☎ 109 (24시간)\n"
    "정신건강위기상담전화 ☎ 1577-0199 (24시간)"
)

# 위기 상황 전용 응답 프롬프트
CRISIS_TEMPLATE = """당신은 정신건강 위기 상담 전문가입니다.
사용자가 자해 또는 자살 사고를 표현했습니다.

아래 규칙을 반드시 지키세요:
1. 사용자의 고통과 감정을 먼저 진심으로 공감하고 인정하세요.
2. 절대 가볍게 여기거나 판단하지 마세요.
3. 혼자가 아님을 따뜻하게 전달하세요.
4. 전문가의 도움을 받도록 자연스럽게 유도하세요.
5. 마지막에 아래 연락처를 반드시 포함하세요.
6. 전체 길이는 5~8문장 이내로 작성하세요.

포함할 연락처 (형식 그대로 유지):
자살예방 상담전화: ☎ 109 (24시간)
정신건강 상담전화: ☎ 1577-0199 (24시간)
청소년 상담전화: ☎ 1388 (24시간)

사용자 발화: {input}"""

crisis_prompt = ChatPromptTemplate.from_template(CRISIS_TEMPLATE)

async def get_crisis_response(text: str, llm: ChatOpenAI) -> str:
    chain = crisis_prompt | llm | StrOutputParser()
    try:
        return await chain.ainvoke({"input": text})
    except Exception:
        return (
            "지금 많이 힘드신 것 같아서 걱정이 돼요. "
            "혼자 감당하지 않아도 괜찮아요. 전문가의 도움을 받아보세요.\n"
            "자살예방 상담전화: ☎ 109 (24시간)\n"
            "정신건강 상담전화: ☎ 1577-0199 (24시간)\n"
            "청소년 상담전화: ☎ 1388 (24시간)"
        )

# PHQ-9 9번 슬롯 전용 미니 프롬프트
SLOT9_TEMPLATE = """다음 문장에서 화자 본인의 자해 또는 자살 사고 여부를 판단하세요.

[판단 기준]
0점: 관련 없음, 또는 타인 얘기 ("친구가 죽고싶다고 했어요" → 0점)
1점: 간접적 암시 ("사라지고 싶다", "힘들어 죽겠다" 등 관용적 표현)
2점: 본인의 자해/자살 사고를 명확히 표현
3점: 구체적 계획 또는 즉각적 위험 신호

반드시 숫자 하나만 반환하세요. (0, 1, 2, 3 중 하나, 다른 텍스트 절대 금지)

문장: {text}"""

slot9_prompt = ChatPromptTemplate.from_template(SLOT9_TEMPLATE)

async def get_slot9_score_llm(text: str, llm: ChatOpenAI) -> int:
    chain = slot9_prompt | llm | StrOutputParser()
    try:
        result = await chain.ainvoke({"text": text})
        match  = re.search(r"[0-3]", result.strip())
        return int(match.group()) if match else 0
    except Exception:
        return 0


# ═══════════════════════════════════════════════════════════════
# 3. RAG 초기화
# ═══════════════════════════════════════════════════════════════

print("🔍 [1/3] PDF 문서를 읽고 학습을 시작합니다...")

if not os.path.exists("./data"):
    os.makedirs("./data")
    print("⚠️  'data' 폴더가 비어있습니다.")

loader = DirectoryLoader("./data", glob="*.pdf", loader_cls=PyPDFLoader)
docs   = loader.load()

retriever = None
llm       = None

if not docs:
    print("❌ 학습할 PDF 파일이 data 폴더에 없습니다!")
else:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    splits        = text_splitter.split_documents(docs)

    print("🔍 [2/3] 임베딩 중입니다...")
    embeddings  = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./db"
    )
    print(f"✅ [3/3] 총 {len(splits)}개 학습 완료!")

    llm       = ChatOpenAI(model_name="gpt-4o-mini")
    retriever = vectorstore.as_retriever()

# DB 초기화
init_db()
print("✅ SQLite DB 초기화 완료")


# ═══════════════════════════════════════════════════════════════
# 4. 챗봇 프롬프트
# ═══════════════════════════════════════════════════════════════

# ── 정보성 질문 템플릿 (비율, 정의, 원인, 증상 등)
INFO_TEMPLATE = """당신은 정신건강 전문 상담사 AI입니다.

다음 규칙을 반드시 지키세요:
1. 아래 상담 자료에 근거한 정확한 정보를 먼저 명확하게 제공하세요.
2. 마지막에 따뜻한 한 마디를 짧게 덧붙이세요.
3. 필요한 만큼 충분히 답변하세요.
4. 상담 자료에 없는 내용은 추측하거나 지어내지 마세요.
5. 상담 자료에 관련 내용이 없을 경우, "제가 가진 자료에서는 찾기 어렵지만, " 으로 시작하세요.
6. 이전 대화 내용을 기억하고 연결해서 답변하세요.

상담 자료: {context}
이전 대화: {chat_history}
사용자: {input}
상담사:"""

# ── 감정적 질문 템플릿 (고민, 어려움, 감정 표현 등)
EMOTION_TEMPLATE = """당신은 따뜻하고 공감 능력이 뛰어난 마음 상담사 AI입니다.

다음 규칙을 반드시 지키세요:
1. 사용자의 감정을 먼저 충분히 인정하고 공감하세요.
2. 판단하거나 비판하지 마세요.
3. 따뜻하고 부드러운 말투를 사용하세요.
4. 필요한 만큼 충분히 답변하세요.
5. 상담 자료에 근거한 조언을 부드럽게 덧붙이세요.
6. 상담 자료에 관련 내용이 없을 경우, "제가 가진 자료에서는 찾기 어렵지만, " 으로 시작하여 공감 위주로 답변하세요.
7. 이전 대화 내용을 기억하고 연결해서 답변하세요.

상담 자료: {context}
이전 대화: {chat_history}
사용자: {input}
상담사:"""

# ── 질문 분류 프롬프트
CLASSIFY_TEMPLATE = """아래 질문이 '정보성'인지 '감정적'인지 분류하세요.

정보성: 사실, 비율, 정의, 원인, 증상, 방법 등을 묻는 질문
감정적: 고민, 어려움, 감정 표현, 위로 요청 등이 포함된 질문
애매한 경우: 감정적으로 분류하세요.

반드시 '정보성' 또는 '감정적' 중 하나만 반환하세요. 다른 텍스트 금지.

질문: {question}"""

classify_prompt = ChatPromptTemplate.from_template(CLASSIFY_TEMPLATE)
info_prompt     = ChatPromptTemplate.from_template(INFO_TEMPLATE)
emotion_prompt  = ChatPromptTemplate.from_template(EMOTION_TEMPLATE)


async def classify_question(question: str, llm: ChatOpenAI) -> str:
    """질문 유형 분류: '정보성' 또는 '감정적'"""
    chain = classify_prompt | llm | StrOutputParser()
    try:
        result = await chain.ainvoke({"question": question})
        return "정보성" if "정보성" in result else "감정적"
    except Exception:
        return "감정적"  # 오류 시 감정적으로 처리


# ═══════════════════════════════════════════════════════════════
# 5. Request / Response 스키마
# ═══════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    question:   str
    session_id: str = "default"

class ChatResponse(BaseModel):
    answer:  str
    is_risk: bool

class ContextRequest(BaseModel):
    question: str

class DiaryLog(BaseModel):
    date:               str
    emotion:            int
    sleep_start:        str = ""
    sleep_end:          str = ""
    sleep_hours:        str = ""
    took_medicine:      bool = False
    medicine_reaction:  str = ""
    diary:              str = ""

class ReportRequest(BaseModel):
    session_id: str
    start_date: str
    end_date:   str
    diary_logs: list[DiaryLog] = []


# ═══════════════════════════════════════════════════════════════
# 6. /ai/chat 엔드포인트
# ═══════════════════════════════════════════════════════════════

@app.post("/ai/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):

    if retriever is None or llm is None:
        return ChatResponse(
            answer="학습된 문서가 없습니다. data 폴더에 PDF를 넣고 서버를 재시작하세요.",
            is_risk=False
        )

    # ── 1. SQLite에서 최근 30턴 로드
    history      = load_recent_history(req.session_id, limit=HISTORY_LIMIT)
    history_text = ""
    for h in history:
        role_label    = "사용자" if h["role"] == "user" else "상담사"
        history_text += f"{role_label}: {h['content']}\n"

    # ── 2. RAG 컨텍스트 준비
    context_docs = retriever.invoke(req.question)
    context_text = "\n".join([d.page_content for d in context_docs])

    # ── 3. 질문 분류 + LLM 병렬 호출
    q_type, slot9_score = await asyncio.gather(
        classify_question(req.question, llm),
        get_slot9_score_llm(req.question, llm)
    )

    # 분류 결과에 따라 프롬프트 선택
    selected_prompt = info_prompt if q_type == "정보성" else emotion_prompt
    rag_chain  = selected_prompt | llm | StrOutputParser()
    rag_answer = await rag_chain.ainvoke({
        "context":      context_text,
        "chat_history": history_text,
        "input":        req.question
    })

    # ── 4. 위험 판정
    is_risk = slot9_score >= 2

    # ── 5. 응답 결정
    if is_risk:
        answer = await get_crisis_response(req.question, llm)
    else:
        answer = rag_answer

    # ── 6. SQLite에 대화 저장
    save_message(req.session_id, "user", req.question)
    save_message(req.session_id, "ai",   answer)

    return ChatResponse(answer=answer, is_risk=is_risk)


# ═══════════════════════════════════════════════════════════════
# 6-1. /ai/context 엔드포인트 (RAGAS 평가용)
# ═══════════════════════════════════════════════════════════════

@app.post("/ai/context")
async def get_context(req: ContextRequest):
    """RAGAS 평가용 — 질문에 관련된 RAG 컨텍스트 반환"""
    if retriever is None:
        return {"contexts": []}
    docs = retriever.invoke(req.question)
    return {"contexts": [doc.page_content for doc in docs]}


# ═══════════════════════════════════════════════════════════════
# 7. /ai/report 엔드포인트
# ═══════════════════════════════════════════════════════════════

PHQ9_TEMPLATE = """
당신은 전문 정신과 의사를 보조하여 환자의 데이터를 객관적으로 분석하는 데이터 분석가입니다.
아래 데이터를 종합하여 PHQ-9 지표를 추출하고 관찰된 사실만 요약하세요.

주의사항:
- 진료 권고, 치료 방향, 결론 등 의사의 판단 영역은 절대 포함하지 마세요.
- 관찰된 사실과 데이터만 기술하세요.
- 반드시 주요 증상, 위험요인, 개선요인 3가지 섹션으로만 작성하세요.

[점수 산출 기준]
- 0점: 전혀 없음
- 1점: 며칠 동안 (가끔)
- 2점: 일주일 이상 (자주)
- 3점: 거의 매일 (항상)

반드시 아래 JSON 형식으로만 답변하세요:
{{
    "summary": "주요 증상: (관찰된 증상 사실만 기술)\\n\\n위험요인: (관찰된 위험 지표만 기술)\\n\\n개선요인: (관찰된 긍정적 지표만 기술)",
    "phq9_slots": [
        {{"item_no": 1, "symptom": "흥미/즐거움 저하",      "score": 0, "evidence": "..."}},
        {{"item_no": 2, "symptom": "기분 저하",             "score": 0, "evidence": "..."}},
        {{"item_no": 3, "symptom": "수면 장애",             "score": 0, "evidence": "..."}},
        {{"item_no": 4, "symptom": "피로감",                "score": 0, "evidence": "..."}},
        {{"item_no": 5, "symptom": "식욕 변화",             "score": 0, "evidence": "..."}},
        {{"item_no": 6, "symptom": "부정적 인식",           "score": 0, "evidence": "..."}},
        {{"item_no": 7, "symptom": "집중력 저하",           "score": 0, "evidence": "..."}},
        {{"item_no": 8, "symptom": "정신운동 지체/초조",    "score": 0, "evidence": "..."}},
        {{"item_no": 9, "symptom": "자해/자살 사고",        "score": 0, "evidence": "..."}}
    ],
    "total_score": 0,
    "risk_level": "저위험/중간위험/고위험"
}}

[1. 일기 및 생활 로그]
{diary_section}

[2. 챗봇 대화 기록]
{conversation}
"""

report_prompt = ChatPromptTemplate.from_template(PHQ9_TEMPLATE)

@app.post("/ai/report")
async def generate_report(req: ReportRequest):

    if llm is None:
        return {"error": "LLM이 초기화되지 않았습니다. data 폴더에 PDF를 넣고 서버를 재시작하세요."}

    history = load_history_by_date(req.session_id, req.start_date, req.end_date)

    if not history and not req.diary_logs:
        return {"error": "분석할 데이터가 없습니다."}

    full_conversation = ""
    for h in history:
        role_label = "사용자" if h["role"] == "user" else "AI"
        full_conversation += f"{role_label}: {h['content']}\n"

    if not full_conversation:
        full_conversation = "챗봇 대화 기록 없음"

    diary_section = ""
    for log in req.diary_logs:
        diary_section += f"\n[{log.date}]\n"
        diary_section += f"- 감정 점수: {log.emotion}/5\n"
        diary_section += f"- 수면: {log.sleep_start} ~ {log.sleep_end} ({log.sleep_hours})\n"
        took = "복약함" if log.took_medicine else "복약 안 함"
        diary_section += f"- 복약 여부: {took}\n"
        if log.medicine_reaction:
            diary_section += f"- 복약 반응: {log.medicine_reaction}\n"
        if log.diary:
            diary_section += f"- 일기: {log.diary}\n"

    if not diary_section:
        diary_section = "일기/생활 로그 없음"

    chain      = report_prompt | llm | StrOutputParser()
    raw_result = chain.invoke({
        "conversation":  full_conversation,
        "diary_section": diary_section
    })

    try:
        json_start = raw_result.find("{")
        json_end   = raw_result.rfind("}") + 1
        json_str   = raw_result[json_start:json_end]
        result     = json.loads(json_str)

        slot9       = next((s for s in result.get("phq9_slots", []) if s["item_no"] == 9), None)
        slot9_score = slot9["score"] if slot9 else 0
        total_score = result.get("total_score", 0)

        if slot9_score == 3:
            result["risk_level"] = "고위험 (즉각 개입 필요)"
        elif slot9_score == 2:
            result["risk_level"] = "중간위험 (면밀한 모니터링 필요)"
        else:
            if total_score >= 20:
                result["risk_level"] = "고위험"
            elif total_score >= 15:
                result["risk_level"] = "중등도위험"
            elif total_score >= 10:
                result["risk_level"] = "중간위험"
            elif total_score >= 5:
                result["risk_level"] = "경미한위험"
            else:
                result["risk_level"] = "정상"

        return result

    except Exception:
        return {
            "error":        "JSON 파싱에 실패했습니다.",
            "raw_response": raw_result
        }


# ═══════════════════════════════════════════════════════════════
# 8. 병원 검색 엔드포인트
# ═══════════════════════════════════════════════════════════════

HOSPITAL_DB_PATH = "./hospitals.db"

def get_hospital_conn():
    if not os.path.exists(HOSPITAL_DB_PATH):
        return None
    return sqlite3.connect(HOSPITAL_DB_PATH)

@app.get("/hospital/search")
def search_hospitals(
    city:     str = "",
    district: str = "",
    dong:     str = "",
    limit:    int = 20,
):
    conn = get_hospital_conn()
    if conn is None:
        return {"error": "병원 DB가 없습니다. fetch_hospitals.py를 먼저 실행해주세요."}

    conditions = []
    params     = []

    if city:
        conditions.append("city LIKE ?")
        params.append(f"%{city}%")
    if district:
        conditions.append("district LIKE ?")
        params.append(f"%{district}%")
    if dong:
        conditions.append("dong LIKE ?")
        params.append(f"%{dong}%")

    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    query = f"""
        SELECT name, address, city, district, dong, phone, lat, lng
        FROM hospitals
        {where}
        ORDER BY name
        LIMIT ?
    """
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    conn.close()

    hospitals = [
        {
            "name":     row[0],
            "address":  row[1],
            "city":     row[2],
            "district": row[3],
            "dong":     row[4],
            "phone":    row[5],
            "lat":      row[6],
            "lng":      row[7],
        }
        for row in rows
    ]

    return {
        "count":     len(hospitals),
        "hospitals": hospitals,
    }

@app.get("/hospital/regions")
def get_regions(city: str = ""):
    conn = get_hospital_conn()
    if conn is None:
        return {"error": "병원 DB가 없습니다."}

    if not city:
        rows = conn.execute(
            "SELECT DISTINCT city FROM hospitals WHERE city != '' ORDER BY city"
        ).fetchall()
        conn.close()
        return {"cities": [r[0] for r in rows]}
    else:
        rows = conn.execute(
            "SELECT DISTINCT district FROM hospitals WHERE city LIKE ? AND district != '' ORDER BY district",
            (f"%{city}%",)
        ).fetchall()
        conn.close()
        return {"districts": [r[0] for r in rows]}


# ═══════════════════════════════════════════════════════════════
# 9. 헬스체크
# ═══════════════════════════════════════════════════════════════

@app.get("/")
def read_root():
    return {
        "status": "Re:Mind 서버 정상 가동 중",
        "rag":    "활성화" if retriever else "비활성화 (PDF 없음)",
        "db":     DB_PATH
    }