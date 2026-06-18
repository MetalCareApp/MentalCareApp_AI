"""
=============================================================
Re:Mind RAG 챗봇 RAGAS 성능 평가 (일반 GPT vs RAG 챗봇 비교)
=============================================================
평가 지표:
  1. Faithfulness       - 답변이 문서에 근거하는가 (환각 측정)
  2. Answer Relevancy   - 답변이 질문에 관련있는가
  3. Context Precision  - 검색된 문서가 질문에 정확한가
  4. Context Recall     - 필요한 정보가 검색됐는가

실행 전:
  - uvicorn main:app --reload 서버 실행 필요
  - ragas_testset.json 생성 필요
=============================================================
"""

import json
import os
import requests
import time
import numpy as np
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI

load_dotenv(dotenv_path="secrets.txt")

TESTSET_PATH = "./ragas_testset.json"
AI_SERVER    = "http://52.65.221.221:8000"
SESSION_ID   = "ragas_eval_session"
client       = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 메트릭 인스턴스 생성
faithfulness      = Faithfulness()
answer_relevancy  = AnswerRelevancy()
context_precision = ContextPrecision()
context_recall    = ContextRecall()


# ──────────────────────────────────────────────────────────
# 1. 테스트셋 로드
# ──────────────────────────────────────────────────────────
def load_testset() -> list[dict]:
    with open(TESTSET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────
# 2. RAG 챗봇 답변 가져오기
# ──────────────────────────────────────────────────────────
def get_rag_answer(question: str) -> str:
    try:
        res = requests.post(
            f"{AI_SERVER}/ai/chat",
            json={"question": question, "session_id": SESSION_ID},
            timeout=30,
        )
        return res.json().get("answer", "")
    except Exception as e:
        print(f"  RAG 서버 오류: {e}")
        return ""


# ──────────────────────────────────────────────────────────
# 3. 일반 GPT 답변 가져오기 (PDF 없이)
# ──────────────────────────────────────────────────────────
def get_plain_gpt_answer(question: str) -> str:
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 정신건강 상담 AI입니다. 질문에 답변해주세요."},
                {"role": "user",   "content": question}
            ],
            max_tokens=300,
            temperature=0.7,
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        print(f"  GPT 오류: {e}")
        return ""


# ──────────────────────────────────────────────────────────
# 4. AWS 서버에서 컨텍스트 가져오기 (/ai/context 엔드포인트)
# ──────────────────────────────────────────────────────────
def get_contexts(question: str) -> list[str]:
    try:
        res = requests.post(
            f"{AI_SERVER}/ai/context",
            json={"question": question},
            timeout=30,
        )
        return res.json().get("contexts", [])
    except Exception as e:
        print(f"  컨텍스트 검색 오류: {e}")
        return []


# ──────────────────────────────────────────────────────────
# 5. 평가 데이터셋 구성
# ──────────────────────────────────────────────────────────
def build_datasets(testset: list[dict]) -> tuple[Dataset, Dataset]:
    rag_questions, rag_answers, rag_contexts, rag_gt = [], [], [], []
    gpt_questions, gpt_answers, gpt_contexts, gpt_gt = [], [], [], []

    print(f"\n총 {len(testset)}개 질문 평가 중...")
    print()

    for i, qa in enumerate(testset, 1):
        question     = qa["question"]
        ground_truth = qa["answer"]
        fallback_ctx = [qa.get("context", "")]

        print(f"  [{i}/{len(testset)}] {question[:40]}...")

        # RAG 챗봇 답변
        rag_answer = get_rag_answer(question)
        contexts   = get_contexts(question) or fallback_ctx

        # 일반 GPT 답변
        gpt_answer = get_plain_gpt_answer(question)

        if rag_answer and len(rag_answer) > 10:
            rag_questions.append(question)
            rag_answers.append(rag_answer)
            rag_contexts.append(contexts)
            rag_gt.append(ground_truth)

        if gpt_answer and len(gpt_answer) > 10:
            gpt_questions.append(question)
            gpt_answers.append(gpt_answer)
            gpt_contexts.append(contexts)
            gpt_gt.append(ground_truth)

        time.sleep(0.5)

    rag_dataset = Dataset.from_dict({
        "question":     rag_questions,
        "answer":       rag_answers,
        "contexts":     rag_contexts,
        "ground_truth": rag_gt,
    })

    gpt_dataset = Dataset.from_dict({
        "question":     gpt_questions,
        "answer":       gpt_answers,
        "contexts":     gpt_contexts,
        "ground_truth": gpt_gt,
    })

    return rag_dataset, gpt_dataset


# ──────────────────────────────────────────────────────────
# 6. RAGAS 평가 실행
# ──────────────────────────────────────────────────────────
def run_evaluation(dataset: Dataset, label: str) -> dict:
    print(f"\n{label} 평가 중... (약 1~2분)")

    llm_wrapped = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    emb_wrapped = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

    faithfulness.llm            = llm_wrapped
    answer_relevancy.llm        = llm_wrapped
    answer_relevancy.embeddings = emb_wrapped
    context_precision.llm       = llm_wrapped
    context_recall.llm          = llm_wrapped

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    return result


# ──────────────────────────────────────────────────────────
# 7. 비교 결과 출력
# ──────────────────────────────────────────────────────────
def safe_float(val) -> float:
    """list, numpy array, float 전부 처리"""
    if isinstance(val, (list, np.ndarray)):
        arr = [v for v in val if v is not None and not np.isnan(float(v))]
        return float(np.mean(arr)) if arr else 0.0
    if val is None:
        return 0.0
    try:
        f = float(val)
        return 0.0 if np.isnan(f) else f
    except Exception:
        return 0.0


def grade(score: float) -> str:
    if score >= 0.8:
        return "우수"
    if score >= 0.6:
        return "양호"
    return "개선 필요"


def print_comparison(rag_result: dict, gpt_result: dict):
    metrics = [
        ("Faithfulness",      "faithfulness"),
        ("Answer Relevancy",  "answer_relevancy"),
        ("Context Precision", "context_precision"),
        ("Context Recall",    "context_recall"),
    ]

    print()
    print("=" * 65)
    print("  Re:Mind RAG 챗봇 vs 일반 GPT 성능 비교")
    print("=" * 65)
    print(f"  {'지표':<22} {'일반 GPT':>10} {'RAG 챗봇':>10} {'향상도':>10}")
    print("-" * 65)

    total_rag, total_gpt = 0, 0
    output = {}

    for label, key in metrics:
        rag_score = safe_float(rag_result[key])
        gpt_score = safe_float(gpt_result[key])
        diff      = rag_score - gpt_score
        arrow     = "▲" if diff > 0 else "▼"
        total_rag += rag_score
        total_gpt += gpt_score
        output[key] = {"gpt": gpt_score, "rag": rag_score, "diff": diff}
        print(f"  {label:<22} {gpt_score:>10.4f} {rag_score:>10.4f} {arrow}{abs(diff):>9.4f}")

    print("-" * 65)
    avg_rag  = total_rag / 4
    avg_gpt  = total_gpt / 4
    avg_diff = avg_rag - avg_gpt
    arrow    = "▲" if avg_diff > 0 else "▼"
    print(f"  {'평균':<22} {avg_gpt:>10.4f} {avg_rag:>10.4f} {arrow}{abs(avg_diff):>9.4f}")
    print("=" * 65)

    print()
    print("[점수 해석]")
    print("  0.8 이상: 우수  /  0.6~0.8: 양호  /  0.6 미만: 개선 필요")

    print()
    print(f"  {'지표':<22} {'일반 GPT':>12} {'RAG 챗봇':>12}")
    print("-" * 50)
    for label, key in metrics:
        g = grade(safe_float(gpt_result[key]))
        r = grade(safe_float(rag_result[key]))
        print(f"  {label:<22} {g:>12} {r:>12}")

    # JSON 저장
    final = {
        "plain_gpt":   {k: safe_float(gpt_result[v]) for k, v in metrics},
        "rag_chatbot": {k: safe_float(rag_result[v]) for k, v in metrics},
        "improvement": {k: v["diff"] for k, v in output.items()},
        "average": {
            "plain_gpt":   avg_gpt,
            "rag_chatbot": avg_rag,
            "improvement": avg_diff
        }
    }
    with open("./ragas_result.json", "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)
    print(f"\n  결과 저장: ./ragas_result.json")


# ──────────────────────────────────────────────────────────
# 8. 메인
# ──────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  Re:Mind RAG 챗봇 RAGAS 성능 평가")
    print("=" * 65)

    # 서버 상태 확인
    try:
        requests.get(f"{AI_SERVER}/", timeout=5)
        print("AI 서버 연결 확인")
    except Exception:
        print("AI 서버가 꺼져있습니다.")
        print("   uvicorn main:app --reload 먼저 실행해주세요.")
        return

    # 테스트셋 로드
    testset = load_testset()
    print(f"테스트셋 로드: {len(testset)}개")

    # 데이터셋 구성
    rag_dataset, gpt_dataset = build_datasets(testset)

    if len(rag_dataset) == 0:
        print("평가 데이터가 없습니다.")
        return

    # 평가 실행
    rag_result = run_evaluation(rag_dataset, "RAG 챗봇")
    gpt_result = run_evaluation(gpt_dataset, "일반 GPT")

    # 비교 결과 출력
    print_comparison(rag_result, gpt_result)


if __name__ == "__main__":
    main()