"""
=============================================================
Re:Mind RAG 챗봇 RAGAS 테스트셋 생성기
=============================================================
- data/ 폴더 PDF 내용 기반으로 GPT가 질문/정답 자동 생성
- 출력: ragas_testset.json
=============================================================
"""

import json
import os
import random
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(dotenv_path="secrets.txt")

client    = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
DATA_DIR  = "./data"
OUTPUT    = "./ragas_testset.json"
NUM_QA    = 50   # 생성할 질문 수
CHUNK_SIZE = 800


def load_pdf_chunks() -> list[str]:
    """data/ 폴더 PDF 로드 후 청크 분할"""
    print("PDF 로드 중...")
    loader = DirectoryLoader(DATA_DIR, glob="*.pdf", loader_cls=PyPDFLoader)
    docs   = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)
    print(f"총 {len(chunks)}개 청크 로드 완료")
    return chunks


def generate_qa(chunk_text: str, existing_questions: list[str] = []) -> dict | None:
    """청크 하나로 GPT가 질문/정답 생성 (중복 방지)"""

    # 기존 질문 목록 텍스트 구성
    existing_str = ""
    if existing_questions:
        existing_str = "\n\n이미 생성된 질문들 (아래와 중복되는 질문 절대 금지):\n"
        existing_str += "\n".join(f"- {q}" for q in existing_questions)

    prompt = f"""아래는 정신건강 관련 전문 문서의 일부입니다.

이 내용을 바탕으로, 정신건강 상담 AI 챗봇에게 실제로 물어볼 법한 질문 1개와 정답 1개를 만들어주세요.

규칙:
- 질문은 반드시 일반인이 상담 챗봇에게 자연스럽게 물어볼 법한 형태여야 함
  (예: "우울증 증상이 어떤 게 있나요?", "PHQ-9 점수가 높으면 어떻게 해야 하나요?")
- 논문 저자, 저널명, 연구 방법론 등 학술적인 질문은 절대 금지
- 정답은 텍스트 내용을 바탕으로 2~3문장으로 친절하게 작성
- 반드시 JSON 형식으로만 반환 (다른 텍스트 금지)
- 이미 생성된 질문과 의미가 비슷한 질문 금지{existing_str}

텍스트:
{chunk_text[:1000]}

출력 형식:
{{
  "question": "질문 내용",
  "answer": "정답 내용"
}}"""

    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.9,  # 다양성을 위해 높게 설정
        )
        raw  = res.choices[0].message.content.strip()
        js   = raw[raw.find("{"):raw.rfind("}")+1]
        data = json.loads(js)

        if "question" in data and "answer" in data:
            return {
                "question": data["question"],
                "answer":   data["answer"],
                "context":  chunk_text[:1000],
            }
    except Exception as e:
        print(f"  ⚠️ 생성 실패: {e}")
    return None


def main():
    print("=" * 55)
    print("  RAGAS 테스트셋 생성 시작")
    print("=" * 55)

    # PDF 청크 로드
    chunks = load_pdf_chunks()
    if not chunks:
        print("❌ PDF 파일이 없습니다. data/ 폴더를 확인해주세요.")
        return

    # 랜덤 청크 선택
    selected = random.sample(chunks, min(NUM_QA, len(chunks)))
    print(f"\n{len(selected)}개 청크에서 질문 생성 중...")
    print()

    qa_list = []
    for i, chunk in enumerate(selected, 1):
        # 지금까지 생성된 질문 목록 전달 (중복 방지)
        existing_questions = [qa["question"] for qa in qa_list]
        qa = generate_qa(chunk.page_content, existing_questions)
        if qa:
            qa["source"] = chunk.metadata.get("source", "")
            qa_list.append(qa)
            print(f"  [{i}/{len(selected)}] ✅ {qa['question'][:50]}...")
        else:
            print(f"  [{i}/{len(selected)}] ❌ 스킵")

    # 저장
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(qa_list, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 55)
    print("  완료!")
    print("=" * 55)
    print(f"  생성된 QA 수: {len(qa_list)}개")
    print(f"  저장 경로: {OUTPUT}")
    print()
    print("[샘플 미리보기]")
    if qa_list:
        print(f"  Q: {qa_list[0]['question']}")
        print(f"  A: {qa_list[0]['answer'][:80]}...")


if __name__ == "__main__":
    main()
