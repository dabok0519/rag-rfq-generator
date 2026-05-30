import os 
from dotenv import load_dotenv
from google import genai

# .env 파일 불러오기 즉 , api 키 불러오기 
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# genai 클라이언트 생성
client = genai.Client(api_key = api_key)


# 첫 호출 
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="안녕! 한 문장으로 자기소개 해줘."
)

print(response.text)