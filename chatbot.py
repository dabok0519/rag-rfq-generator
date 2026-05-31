import os 
from dotenv import load_dotenv  
from google import genai    

class Chatbot:
    def __init__(self):
        load_dotenv()  # .env 파일에서 환경 변수를 로드
        self.api_key = os.getenv("GOOGLE_API_KEY")  # 환경 변수에서 API 키를 가져오는 것 
        self.client = genai.Client(api_key=self.api_key) # genai 클라이언트 초기화
        self.chat = self.client.chats.create(model="gemini-2.5-flash") # 챗봇 모델을 지정하여 채팅 세션을 생성
        print("챗봇 준비완료")

    def run(self):
        print("대화를 시작합니다. (끝내려면 'quit' 입력)")
        while True:
            user_input = input("나: ")        # ① 사용자 입력 받기

            if user_input == "quit":           # ② quit이면 종료
                print("대화를 종료합니다.")
                break

            response = self.chat.send_message(user_input)  # ③ AI에 보내기
            print("챗봇:", response.text)       # ④ 답변 출력


bot = Chatbot()
bot.run()
