import os
from dotenv import load_dotenv
from google import genai
import numpy as np

load_dotenv()  # .env에서 API 키 읽기

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY")) # genai 클라이언트 초기화, API 키는 환경 변수에서 가져옴


result = client.models.embed_content( # embed_content 메서드를 사용하여 텍스트를 벡터로 변환, 모델과 내용을 지정하여 호출
    model="gemini-embedding-001",
    contents="납기일은 언제인가요?"
)
                           

vector = result.embeddings[0].values   # 결과물을 벡터로 추출, embeddings 리스트에서 첫 번째 항목의 values 속성을 가져와 벡터 값을 얻음

print("벡터 차원 수:", len(vector))     # 벡터의 차원 수를 출력, 벡터의 길이를 계산하여 차원 수를 확인 
print("앞 5개 값:", vector[:5])          # 벡터의 앞 5개 값을 출력, 벡터의 처음 5개 요소를 슬라이싱하여 출력하여 벡터의 일부 내용을 확인

# 차원(Dimension) = "단어의 특징을 몇 가지로 설명할 것인가"

def cosine_similarity(a, b): # 코사인 유사도 계산 함수, 두 벡터 a와 b를 입력으로 받아 코사인 유사도를 계산하여 반환하는 함수
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# 질문 1개
question = "납기일은 언제인가요?"

# 후보 문장들 (이 중에서 질문과 가장 가까운 걸 찾을 것)
candidates = [
    "물품 납품 기한은 계약일로부터 30일 이내입니다.",
    "대금은 검수 완료 후 익월 말일에 지급합니다.",
    "오늘 점심은 김치찌개를 먹었습니다.",
]

# 질문 + 후보 전부 한 번에 임베딩
all_texts = [question] + candidates # 질문과 후보 문장들을 하나의 리스트로 합쳐서 임베딩을 한 번에 처리할 수 있도록 하는 것
                                    # 이렇게 하면 모델에게 한 번의 호출로 모든 텍스트를 벡터로 변환할 수 있음 
result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=all_texts
)
vectors = [e.values for e in result.embeddings] # 리스트 컴프리헨션을 사용하여 결과에서 각 텍스트에 대한 벡터를 추출하여 vectors 리스트에 저장
                                                # result.embeddings는 임베딩된 결과가 담긴 리스트이며, 각 항목의 values 속성에서 벡터 값을 가져와서 vectors 리스트에 저장하는 것
                                                # embeddings는 라이브러리(google-genai)를 만들 때, 결괏값을 담는 바구니의 이름을 미리 개발자가 embeddings라고 정해놓은 것

q_vector = vectors[0]          # 질문 벡터
cand_vectors = vectors[1:]     # 후보 벡터들

# 질문 vs 각 후보 유사도 계산
for i in range(len(candidates)):
    text = candidates[i]
    vec = cand_vectors[i]
    score = cosine_similarity(q_vector, vec) # 질문 벡터와 후보 벡터 간의 코사인 유사도를 계산하여 score 변수에 저장, 이 점수는 두 벡터가 얼마나 유사한지를 나타냄
                                             # 1에 가까울수록 유사도가 높음을 의미
    print(f"유사도 {score:.4f}  |  {text}")
