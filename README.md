# Re:Mind

AI 기반 정신건강 케어 플랫폼

사용자의 상담 대화, 생활 로그, PHQ-9 평가 데이터를 기반으로
심리 상담 챗봇과 의사용 임상 리포트를 제공하는 서비스입니다.

## 🤖 AI Architecture

Re:Mind의 AI 파트는 **심리 상담 챗봇**과 **진료 리포트 생성**이라는 두 가지 핵심 엔진으로 구성됩니다.

---

### 1. Counseling Chatbot (RAG-based)

환자와의 대화를 통해 심리 상담을 제공하는 RAG 기반 챗봇입니다.

**Tech Stack**
- `GPT-4o-mini` — 응답 생성 LLM
- `LangChain` — RAG 파이프라인 프레임워크
- `ChromaDB` — 벡터 데이터베이스
- `OpenAI text-embedding-3-small` — 문서 임베딩
- `SQLite` — 대화 기록 영속화

**Pipeline**
```
사용자 입력
    │
    ├──► [ChromaDB] 심리학·임상 문서 검색 (chunk_size=400, overlap=50)
    │         PDF 14개 · 711 chunks
    │
    └──► [PHQ-9 위험 감지] 슬롯 9번 실시간 모니터링
              │
              ▼
         [GPT-4o-mini] RAG 컨텍스트 기반 응답 생성
              │
    ┌─────────┴────────┐
    │                  │
 일반 응답          위기 감지 시
(공감 + 근거)      → 109 자살예방상담전화 안내
```

**Performance (LLM-as-a-Judge, 10점 만점)**

| 지표 | 일반 GPT | RAG 챗봇 | 향상 |
|------|---------|---------|------|
| 공감도 | 7.39 | 8.29 | +0.90 |
| 정확도 | 7.61 | 8.13 | +0.52 |
| 근거 충실도 | 7.27 | 8.03 | +0.76 |
| **평균** | **7.42** | **9.70** | **+2.28** |

> 블라인드 테스트 22명 대상: 10문항 중 7개(70%)에서 RAG 챗봇 우세 확인

---

### 2. Report Generation AI (Fine-tuned LLM)

환자의 일상 데이터를 분석해 의사용 1-page 임상 리포트를 자동 생성합니다.

**Tech Stack**
- `Qwen 2.5-7B` — Base 모델 (Local LLM)
- `QLoRA` — 파인튜닝 기법
- `AI Hub 심리상담 데이터` — 학습 데이터 (71,102개)
- `PHQ-9` — 우울증 자동 채점 척도

**Pipeline**
```
환자 일상 데이터
(감정점수 · 수면 · 복약 · 일기 텍스트)
    │
    ▼
[GPT-4o-mini] PHQ-9 9개 항목 자동 채점
    │   성균관대 논문 피어슨 상관계수 0.8644 근거
    │
    ▼
[Qwen 2.5-7B Fine-tuned] 임상 요약 생성
    │   주요증상 / 위험요인 / 개선요인
    │
    ▼
[ChromaDB] 진료지침 RAG 검색
    │   PHQ-9 총점 기반 동적 쿼리
    │
    ▼
의사용 1-Page 임상 리포트
(PHQ-9 결과 · 임상 요약 · 치료 권고사항)
```

**Fine-tuning Performance**

| 지표 | Base 모델 | Fine-tuned | 향상 |
|------|---------|------------|------|
| ROUGE-1 | 0.045 | 0.275 | **6배** |
| BERTScore F1 | 0.646 | 0.688 | +6.5% |

**학습 데이터**
- AI Hub [심리상담 음성데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71806) 기반
- PHQ-9 슬롯 채점 결과를 임상 리포트 형식으로 출력하도록 학습
- 총 71,102개 (룰 기반 70,602개 + GPT 생성 500개)

---

### 3. Server & Deployment

```
FastAPI (Python)
    │
    ├── POST /ai/chat        # 챗봇 응답
    ├── POST /ai/report      # 진료 리포트 생성
    └── GET  /hospital/*     # 병원 검색

AWS EC2 (nohup 백그라운드 실행)
```

---

### Tech Stack Summary

```
Language  : Python 3.11
Framework : FastAPI, LangChain
LLM       : GPT-4o-mini, Qwen 2.5-7B (QLoRA)
VectorDB  : ChromaDB
Embedding : OpenAI text-embedding-3-small
DB        : SQLite
Infra     : AWS EC2
Eval      : RAGAS, LLM-as-a-Judge, ROUGE, BERTScore
```
