import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# 기존 DB 그대로 로드 (재임베딩 금지)
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)
print("벡터 개수:", vectorstore._collection.count())

# 테스트 질문 4개
questions = [
    "대금 지급 기일은 언제인가?",
    "품질보증은 어떤 기준인가?",
    "납기를 어기면 어떻게 되는가?",
    "납기 준수율은 몇 %인가?",
]

# k값 — 이걸 3, 5로 바꿔가며 실험
K = 3
retriever = vectorstore.as_retriever(search_kwargs={"k": K})

for q in questions:
    print("\n" + "=" * 50)
    print(f"질문: {q}")
    docs = retriever.invoke(q)
    for i, doc in enumerate(docs, 1):
        article = doc.metadata.get("article", "조항 미상")                # ← metadata에서 조항명 꺼내기
        print(f"  [{i}] {article}")
        print(f"      {doc.page_content[:200]}")   # ← 앞 몇 글자만 미리보기
