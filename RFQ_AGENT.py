import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv() # .env 파일에 저장된 환경 변수(API 키 등)를 시스템에 로드

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)
print("불러온 벡터 개수:", vectorstore._collection.count())

retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # 유사도가 높은 3개의 chunk 

# === 여기부터 STEP 2 추가 ===

# 생성용 LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
# ??? Q: temperature=0으로 둔 이유는? (아래 질문) -> temperature가 낮을 수록 매우 논리적이고 사실적인 답변을 생성 

def format_context(docs):
    """검색된 Document 리스트를 출처 포함 문자열로 합친다."""
    blocks = []
    for i, doc in enumerate(docs):  # enumerate()는 리스트를 index 번호와 내용으로 깔끔하게 분리해줌 
        source = doc.metadata.get("source", "unknown") # metadata에 source가 있으면 그 값을, 없으면 "unknown"을 반환 -> 
        article = doc.metadata.get("article", "조항 미상")
        block = f"[근거 {i+1}] (출처: {source}, {article})\n{doc.page_content}"
        blocks.append(block)
    return "\n\n".join(blocks)


RFQ_CLAUSE_PROMPT = """너는 구매 계약서를 돕는 어시스턴트다.
아래 [근거 조항]만 사용해서, RFQ에 들어갈 '거래 조건' 항목을 정리하라.

규칙:
- 근거에 있는 내용만 쓸 것. 없으면 지어내지 말 것.
- 각 항목 끝에 출처(조항명)를 표시할 것.
- 품목/수량/납기/단가 같은 건 여기서 다루지 말 것.

[근거 조항]
{context}

[거래 조건]
"""

while True:
    item = input("\n품목 (종료: q): ")
    if item == "q":
        break
    qty = input("수량: ")
    due = input("납기일: ")

    search_query = "대금지급 납기 품질보증 하자담보 지체상금"
    docs = retriever.invoke(search_query)
    context = format_context(docs)
    prompt = RFQ_CLAUSE_PROMPT.format(context=context)

    clause = llm.invoke(prompt).content

    rfq = f"""===== RFQ (견적요청서) 초안 =====

[1. 구매 품목]
- 품목: {item}
- 수량: {qty}
- 납기일: {due}

[2. 거래 조건]
{clause}

[3. 공급사 기재란]
- 단가: 
- 총 견적금액: 
- 회신 기한: 
"""
    print(rfq)