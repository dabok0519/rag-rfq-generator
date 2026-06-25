# 1) 베이스 이미지 — 로컬 파이썬 3.13에 맞춤
FROM python:3.13-slim

# 2) 상자 안 작업 폴더
WORKDIR /app

# 3) 의존성 목록 먼저 복사 (캐시 활용)
COPY requirements.txt .

# 4) 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# 5) 나머지 코드 전부 복사 (app.py, rfq_graph.py, chroma_db/ 등)
COPY . .

# 6) Streamlit 포트 열기
EXPOSE 8501

# 7) 실행 명령
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]