import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv()

# 데이터를 저장한 것 뿐 문서를 대조할 임베딩 객체는 여전히 필요함. 
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# 저장된 Chroma 불러오기 (from_documents 아님! 새로 임베딩 안 함)
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)
print("불러온 벡터 개수:", vectorstore._collection.count())   # 148 떠야 정상

# Retriever 만들기
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}   # ← 몇 개 가져올까? 어제 n_results 몇이었지?
)

# 검색 루프 (어제 while True CLI 재현)
while True:
    query = input("\n질문 (종료: q): ")
    if query == "q":
        break
    results = retriever.invoke(query)   # ← 한 줄! dict 파헤치기 없음
    for i, doc in enumerate(results):
        print(f"\n--- 결과 {i+1} ---")
        print("내용:", doc.page_content[:200])
        print("출처:", doc.metadata.get("source"), "p.", doc.metadata.get("page"))