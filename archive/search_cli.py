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


PROMPT_TEMPLATE = """너는 구매 담당자를 돕는 어시스턴트다.
아래 [근거]에 있는 내용만 사용해서 질문에 답하라.

규칙:
- 근거에 없는 내용은 지어내지 말고 "근거에서 찾을 수 없습니다"라고 답하라.
- 답변 끝에 사용한 근거의 출처(문서명, 페이지)를 표시하라.

[근거]
{context}

[답변]

"""

# === while 루프 교체 ===
while True:
    query = input("\n질문 (종료: q): ")
    if query == "q":
        break

    docs = retriever.invoke(query)  # 입력받은 질문(query)을 기반으로 리트리버(retriever)를 통해 관련 문서(docs)들을 검색해 결과를 반환 
    context = format_context(docs) # 검색된 문서들을 LLM이 이해하기 좋은 하나의 텍스트 포맷(문맥, context)으로 결합/가공
    prompt = PROMPT_TEMPLATE.format(context=context, question=query) # 미리 정의해둔 프롬프트 템플릿에 가공된 문맥(context)과 사용자의 질문(query)을 채워 넣음 

    # 일단 LLM 호출 전에 프롬프트부터 눈으로 확인
    print("\n===== 조립된 프롬프트 =====")
    print(prompt)

    answer = llm.invoke(prompt)     # 완성된 프롬프트를 LLM 모델에 전달하여 실행(invoke)하고 결과(답변 객체)를 반환 
    print("\n===== 답변 =====")
    print(answer.content)  # LLM이 생성한 답변 내용 중 실제 텍스트 메시지(content)만 추출하여 출력 
