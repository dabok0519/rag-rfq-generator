# [Gemini 파이썬 튜터 챗봇]

파이썬을 초보자도 알기 쉽게 알려주는 챗봇 생성

## 주요 기능

- 대화를 기억함
- 챗봇 성격을 정할 수 있음
- 에러가 나도 안 죽음

## 기술 스택

- 언어 : Python
- dotenv를 통한 가상환경 설정 
- google 패키지 -  genai , types


## 설치 및 실행 방법

아래 명령어를 순서대로 실행하세요.

```
mkdir hello-ai
cd hello-ai
python -m venv venv
venv\Scripts\activate
pip install google-genai python-dotenv
python chatbot.py
```

실행 전 `.env` 파일에 `GOOGLE_API_KEY`를 넣어야 합니다.
