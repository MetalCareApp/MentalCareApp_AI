import requests

print("Re:Mind 마음 상담 챗봇입니다. 종료하려면 'quit' 입력")
print("-" * 40)

session_id = "user1"

while True:
    question = input("나: ")
    if question.lower() == "quit":
        print("챗봇을 종료합니다.")
        break

    try:
        res = requests.post(
            "http://127.0.0.1:8000/ai/chat",
            json={"question": question, "session_id": session_id}
        )
        data    = res.json()
        answer  = data["answer"]
        is_risk = data["is_risk"]

        # is_risk=True면 터미널에 경고 표시
        if is_risk:
            print("[⚠️  위험 감지 is_risk=True]")

        print(f"상담사: {answer}")
        print()

    except Exception as e:
        print(f"오류 발생: {e}")
        print()
