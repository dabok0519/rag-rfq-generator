import os 
from dotenv import load_dotenv  
from google import genai    
from google.genai import types # genai 라이브러리에서 types 모듈을 가져오는 것, 이 모듈은 다양한 데이터 유형과 구조를 정의하는 데 사용


class Chatbot:
    def __init__(self, system_prompt="너는 친절한 파이썬 튜터야.", model_choice = "models/gemini-2.5-flash"):
        load_dotenv()  # .env 파일에서 환경 변수를 로드
        self.api_key = os.getenv("GOOGLE_API_KEY")  # 환경 변수에서 API 키를 가져오는 것 
        self.client = genai.Client(api_key=self.api_key) # genai 클라이언트 초기화
        
        
        self.chat = self.client.chats.create(model=model_choice, config=types.GenerateContentConfig(
                system_instruction=system_prompt
            )) # 챗봇 모델을 지정하여 채팅 세션을 생성
        print("챗봇 준비완료")

    def run(self):
        print("대화를 시작합니다. (끝내려면 'quit' 입력)")
        while True:
            user_input = input("나: ")

            if user_input == "quit":
                print("대화를 종료합니다.")
                break

            try:
                response = self.chat.send_message(user_input)
                print("챗봇:", response.text)
            except Exception as e:
                print("⚠️ 오류가 발생했어요:", e)
                print("(잠시 후 다시 시도해주세요)")

bot = Chatbot()
bot.run()
