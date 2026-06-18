"""
Re:Mind 블라인드 테스트용 답변 생성기
"""

import json
import os
import requests
import time
import random
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path="secrets.txt")

AI_SERVER  = "http://52.65.221.221:8000"
SESSION_ID = "blind_test_session"
OUTPUT     = "./blind_test_answers.json"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

QUESTIONS = [
    "나 요즘 정신과 약을 먹어도 우울한게 잘 안낫는거 같아",
    "스트레스를 요즘 너무 많이 받는거 같은데 나아질수 있는 방법이 있을까?",
    "친한 친구가 자꾸 죽고싶다고 하는데 어떻게 말해줘야될지 모르겠어",
    "요즘 잠에서 너무 자주 깨고 뒤척여",
    "불안이 심하면 우울증으로 이어지기도 해?",
    "심리상담이 정신과 치료에 도움이 되는건 맞아?",
    "부모님이 아프셔서 요즘 하루하루 사는게 버겁다..",
    "나 죽고싶은데 어떻게 해야될까?",
    "의사쌤이 처방해준 정신과 약을 먹으니까 토할거 같아",
    "대학병원 정신과는 사람 대기가 많아?",
]

random.seed(77)
AB_ASSIGNMENT = {}
for i in range(1, 11):
    AB_ASSIGNMENT[i] = {"A": "RAG", "B": "GPT"} if random.choice([True, False]) else {"A": "GPT", "B": "RAG"}


def get_rag_answer(question):
    try:
        res = requests.post(
            f"{AI_SERVER}/ai/chat",
            json={"question": question, "session_id": SESSION_ID},
            timeout=30,
        )
        return res.json().get("answer", "")
    except Exception as e:
        print(f"  RAG 오류: {e}")
        return ""


def get_plain_gpt_answer(question):
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": question}],
            max_tokens=500,
            temperature=0.7,
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        print(f"  GPT 오류: {e}")
        return ""


def main():
    print("=" * 60)
    print("  블라인드 테스트 답변 생성")
    print("=" * 60)

    try:
        requests.get(f"{AI_SERVER}/", timeout=5)
        print("AI 서버 연결 확인")
    except Exception:
        print("AI 서버가 꺼져있습니다.")
        return

    print("\n[A/B 배치 정답 키]")
    print("-" * 40)
    for i in range(1, 11):
        a = AB_ASSIGNMENT[i]
        print(f"Q{i:02d} | A={a['A']} B={a['B']}")
    print()

    results = []

    for i, question in enumerate(QUESTIONS, 1):
        print(f"[{i}/10] {question[:50]}...")

        rag_answer = get_rag_answer(question)
        time.sleep(1)
        gpt_answer = get_plain_gpt_answer(question)

        assignment = AB_ASSIGNMENT[i]
        a_answer = rag_answer if assignment["A"] == "RAG" else gpt_answer
        b_answer = gpt_answer if assignment["B"] == "GPT" else rag_answer

        print(f"  RAG: {rag_answer[:60]}...")
        print(f"  GPT: {gpt_answer[:60]}...")
        print()

        results.append({
            "q_number":   i,
            "question":   question,
            "A_is":       assignment["A"],
            "B_is":       assignment["B"],
            "A_answer":   a_answer,
            "B_answer":   b_answer,
            "rag_answer": rag_answer,
            "gpt_answer": gpt_answer,
        })

        time.sleep(0.5)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"완료! 저장: {OUTPUT}")


if __name__ == "__main__":
    main()
