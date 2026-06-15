import os , json # JSON 패키지import 

from rich.console import Console
from rich.markdown import Markdown

from dotenv import load_dotenv  
from google import genai    
from google.genai import types # genai 라이브러리에서 types 모듈을 가져오는 것, 이 모듈은 다양한 데이터 유형과 구조를 정의하는 데 사용



class Basebot:
    def __init__(self,temperature, system_prompt, model_choice = "models/gemini-2.5-flash"):
        load_dotenv()  # .env 파일에서 환경 변수를 로드

        self.api_key = os.getenv("GOOGLE_API_KEY")  # 환경 변수에서 API 키를 가져오는 것 
        self.client = genai.Client(api_key=self.api_key) # genai 클라이언트 초기화
        
        # GenerateContentConfig: 이것은 SDK가 모델에게 "어떤 방식으로 답변을 생성할지" 알려주기 위해 만들어 둔 설정 객체(Configuration Object)
        self.chat = self.client.chats.create(model=model_choice, config=types.GenerateContentConfig( 
                system_instruction=system_prompt,
                temperature=temperature,
            )) # 챗봇 모델을 지정하여 채팅 세션을 생성
        self.history = [] # 대화의 히스토리를 저장하는 리스트, 사용자가 질문할 때마다 질문과 챗봇의 답변을 이 리스트에 추가하여 대화의 흐름을 유지할 수 있도록 하는 것
        self.console = Console() # 콘솔 객체를 생성하여 터미널에 다양한 스타일의 텍스트를 출력할 수 있도록 하는 것, 예를 들어 오류 메시지를 빨간색으로 표시하거나 중요한 정보를 강조하는 데 사용할 수 있음
        print("챗봇 준비완료")
    
    """ 
    def ask_once(self, Question):
        try:
            response = self.chat.send_message(Question) # 사용자의 질문을 챗봇에게 보내고 응답을 받는 것
            self.console.print("챗봇:", style="bold cyan")
            self.console.print(Markdown(response.text))

        except Exception as e:
            self.console.print("⚠️ 오류가 발생했어요:", style="bold red")
            self.console.print("(잠시 후 다시 시도해주세요)", style="italic")
            return None """ 
    
    def save_history(self): # 대화 히스토리를 JSON 파일로 저장하는 메서드, 대화가 끝난 후에 이 메서드를 호출하여 대화 기록을 "history.json" 파일에 저장할 수 있도록 하는 것
        with open("history.json", "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
        self.console.print("대화 기록을 저장했습니다.", style="bold green")

    def run(self):
        self.console.print("대화를 시작합니다. (끝내려면 'quit' 입력)", style="bold blue")
        while True:
            user_input = input("나: ")
            self.history.append({"role": "user", "content": user_input}) # 각 항목은 역할(role)과 내용(content)을 포함하는 딕셔너리 형태로 저장됨.
                                                                         # 역할은 "user"로 지정되어 사용자의 질문임을 나타냄

            if user_input == "quit":
                self.save_history()  # 대화 종료 시 히스토리 저장
                self.console.print("대화를 종료합니다.", style="bold red")
                break

            try:
                response = self.chat.send_message(user_input)
                self.history.append({"role": "bot", "content": response.text}) # 챗봇의 답변도 히스토리에 저장, 역할은 "bot"으로 지정되어 챗봇의 응답임을 나타냄
                self.console.print("챗봇:", style="bold cyan")
                self.console.print(Markdown(response.text))

            except Exception as e:
                self.console.print("⚠️ 오류가 발생했어요:", style="bold red")
                self.console.print("(잠시 후 다시 시도해주세요)", style="italic")

class Chatbot(Basebot):
    def __init__(self, temperature, system_prompt="너는 친절한 파이썬 튜터야." ):
        super().__init__(temperature, system_prompt) # 부모 클래스인 Basebot의 __init__ 메서드를 호출하여 초기화 작업을 수행






bot = Chatbot(temperature=1.0)  # 온도를 낮춰서 좀 더 일관된 답변을 얻을 수 있도록 설정
bot.run()

"""
bot2 = Chatbot(temperature=1.5)  # 온도를 높여서 좀 더 창의적인 답변을 얻을 수 있도록 설정

print("=== temperature 0.0 (3번 반복) ===")
bot_low = Chatbot(temperature=0.0)
for i in range(3):
    bot_low.ask_once("파이썬에서 리스트를 오름차순 정렬하는 함수 이름만 한 단어로 답해.")

print("\n=== temperature 1.5 (3번 반복) ===")
bot_high = Chatbot(temperature=1.5)
for i in range(3):
    bot_high.ask_once("파이썬에서 리스트를 오름차순 정렬하는 함수 이름만 한 단어로 답해.") """