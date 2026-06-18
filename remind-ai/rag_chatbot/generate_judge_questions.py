"""
Re:Mind LLM-as-a-Judge용 50개 질문 생성기
"""
import json
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path="secrets.txt")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

original_10 = [
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

prompt = f"""아래는 심리상담 AI 챗봇에게 실제 사용자가 보낼 법한 질문 10개입니다.
이와 비슷한 말투와 패턴으로 추가 40개를 만들어주세요.

[원본 10개]
{chr(10).join(f'{i+1}. {q}' for i, q in enumerate(original_10))}

[규칙]
- 실제 사람이 챗봇에게 말하는 것처럼 자연스럽고 구어체로
- 아래 5가지 유형 골고루 포함:
  1. 감정 토로형: "나 요즘 너무 힘들어", "우울한데 이유를 모르겠어" 등
  2. 정보 질문형: "우울증이랑 조울증 차이가 뭐야?", "SSRI가 뭔지 설명해줘" 등
  3. 위기형: "사라지고 싶어", "더 이상 못하겠어" 등 (3~5개 이내)
  4. 약/치료 관련: "약 끊어도 돼?", "상담 얼마나 받아야 해?" 등
  5. 타인 걱정형: "남자친구가 자해를 해", "엄마가 우울증인거 같아" 등
- 중복 없이 다양하게
- 반드시 JSON 배열로만 반환 (다른 텍스트 금지): ["질문1", "질문2", ...]
"""

print("질문 생성 중...")
res = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=2000,
    temperature=0.9,
)

raw = res.choices[0].message.content.strip()
js = raw[raw.find('['):raw.rfind(']')+1]
new_40 = json.loads(js)

all_50 = original_10 + new_40[:40]
print(f"총 {len(all_50)}개 생성 완료!")
print()
for i, q in enumerate(all_50, 1):
    print(f"{i:02d}. {q}")

with open("./judge_questions_50.json", "w", encoding="utf-8") as f:
    json.dump(all_50, f, ensure_ascii=False, indent=2)

print(f"\n저장 완료: judge_questions_50.json")
