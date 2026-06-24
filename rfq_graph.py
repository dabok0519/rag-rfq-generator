from typing import TypedDict, List # TypedDict: 딕셔너리(Dictionary)의 키(Key)와 값(Value)의 타입을 고정해 주는 도구
from langchain_core.documents import Document # Document: 랭체인(LangChain) 라이브러리에서 사용하는 문서 객체 형식

class RFQState(TypedDict):
    """
    RFQ(견적 요청서) 생성 워크플로우에서 단계별 데이터를 저장하고 전달하는 상태(State) 클래스
    """
    # 1. 사용자가 입력한 원본 구매요청 데이터
    # 예: {"item": "노트북", "quantity": 50, "due_date": "2026-12-31", "budget": "50,000,000원"}
    purchase_request: dict           # 사용자 구매요청 (품목·수량·납기·예산)
    
    # 2. RAG(검색 증강 생성) 시스템을 통해 사내 규정이나 데이터베이스에서 찾아낸 표준 계약/구매 조항들
    # LangChain의 Document 객체(텍스트 내용 및 메타데이터 포함)들이 리스트 형태로 저장
    retrieved_docs: List[Document]   # 검색된 표준조항

    missing_fields: List[str]        # ← 추가: 빠진 필수항목 목록 (비어있으면 통과)

    # 3. 1번의 요청 조건과 2번의 표준 조항을 결합하여 LLM(거대언어모델)이 최종적으로 작성한 RFQ 초안 텍스트
    rfq_draft: str                   # RFQ 초안


import os
from datetime import datetime, timedelta
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

# 기존 search_cli.py와 동일하게 vectorstore 불러오기
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 고정 검색 쿼리 (A안 — 사용자 입력과 무관)
FIXED_QUERY = "대금지급 납기 품질보증 하자담보 지체상금"


def search_node(state: RFQState) -> dict:
    """표준조항 검색 노드 — 고정 쿼리로 검색해서 retrieved_docs 칸을 채운다."""
    docs = retriever.invoke(FIXED_QUERY)
    return {"retrieved_docs": docs}


def analyze_node(state: RFQState) -> dict: #결과값을 dict(딕셔너리) 형태로 반환 
    """구매요청 분석 노드 — 필수항목이 다 있는지 검증해서 missing_fields 칸을 채운다."""
    req = state["purchase_request"]
    required = ["item", "quantity", "due_date"]

    #  req 안에 required 항목들이 있는지 확인해서, 빠진 것들의 리스트를 만들기
    missing = [k for k in required if k not in req]

    # missing 을 State의 missing_fields에 담아 반환 
    return {"missing_fields": missing}

def route_after_analyze(state: RFQState) -> str:
    """누락 항목 유무로 다음 행선지를 정하는 라우터.
    비었으면 search로(ok), 빠진 게 있으면 멈춤으로(missing)."""
    missing = state["missing_fields"]
    if not missing:
        return "ok"
    else:
        return "missing"


from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# rfq_generate.py에서 그대로 가져옴
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


def format_context(docs):
    """검색된 Document 리스트를 출처 포함 문자열로 합친다. (search_cli.py와 동일)"""
    blocks = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        article = doc.metadata.get("article", "조항 미상")
        blocks.append(f"[근거 {i+1}] (출처: {source}, {article})\n{doc.page_content}")
    return "\n\n".join(blocks)


def generate_node(state: RFQState) -> dict:
    """RFQ 초안 생성 노드 — LLM은 거래조건만, 나머지는 f-string 조립."""
    req = state["purchase_request"]
    docs = state["retrieved_docs"]
    

    # 기존 코드 이외에 외부 의존성을 낮추기 위한 가드레일 추가 
    # 가드 ① 검색 결과가 0건이면 — 근거 없이는 생성하지 않는다
    if not docs: # docs의 내용이 비어있다는 것은 관련 표준 조항을 찾지 못했다는 것 
        return {"rfq_draft": "⚠️ 관련 표준조항을 찾지 못해 RFQ를 생성할 수 없습니다."}

    # 가드 ② LLM 호출을 try/except로 감싼다
    context = format_context(docs)
    prompt = RFQ_CLAUSE_PROMPT.format(context=context)
    try:
        clause = llm.invoke(prompt).content 
    except Exception as e: # api 호출 실패 시에 오류 처리 
        return {"rfq_draft": f"⚠️ 거래조건 생성 중 오류가 발생했습니다: {e}"}





    """# ① LLM으로 거래조건만 생성 (근거 = 검색된 조항)
    context = format_context(docs)
    prompt = RFQ_CLAUSE_PROMPT.format(context=context)
    clause = llm.invoke(prompt).content """

    # ② 출처 목록 만들기 (어떤 조항을 근거로 썼는지)
    sources = ", ".join(doc.metadata.get("article", "조항 미상") for doc in docs)

    # ③ f-string으로 최종 조립 (사용자 입력값은 LLM 안 거치고 직접 박음)
     # 문서 메타 (오늘 날짜 기준 자동 생성)
    today = datetime.now()
    doc_no = f"RFQ-{today.strftime('%Y%m%d')}-001"
    issue_date = today.strftime('%Y-%m-%d')
    reply_due = (today + timedelta(days=7)).strftime('%Y-%m-%d')  # 회신기한: 발행일+7일

    # ③ f-string으로 최종 조립 (사용자 입력값은 LLM 안 거치고 직접 박음)
    rfq = f"""# 📋 견적요청서 (RFQ)

- **문서번호**: {doc_no}
- **발행일자**: {issue_date}
- **회신기한**: {reply_due}

## 1. 구매 요청 정보 

- **품목**: {req.get('item', '-')}
- **수량**: {req.get('quantity', '-')}
- **희망 납기일**: {req.get('due_date', '-')}
- **책정 예산**: {req.get('budget', '-')}

## 2. 거래 조건 (표준계약 근거) 
{clause}

## 3. 공급사 회신란 

- **공급사명**:
- **제안 단가**:
- **총 견적금액**:
- **납품 가능일**:
- **비고**:

## 4. 근거 표준조항 
본 RFQ의 거래조건은 아래 표준조항을 근거로 작성되었습니다.
- {sources}

※ 본 견적요청서는 표준하도급계약서를 근거로 자동 생성된 초안입니다.
※ 회신기한 내 미회신 시 견적 의사가 없는 것으로 간주합니다.
"""
    return {"rfq_draft": rfq}






from langgraph.graph import StateGraph, END

# 1) 그래프 객체 생성 — 설계도(RFQState)를 알려줌
graph = StateGraph(RFQState)

# 2) 노드 등록 — "search"라는 이름으로 search_node 함수를 매단다
graph.add_node("search", search_node)
graph.add_node("analyze", analyze_node)   # ← 추가: 분석 노드 등록
graph.add_node("generate", generate_node)    # ← 추가

# 3) 시작점과 끝점 지정
graph.set_entry_point("analyze")          # ← 변경: 시작점을 analyze로
graph.add_conditional_edges(
    "analyze",            # 어느 노드 다음에 갈림길?
    route_after_analyze,  # 판단할 라우터 함수
    {
        "ok": "search",   # "ok" 반환 → search로
        "missing": END,   # "missing" 반환 → 끝
    },
)
graph.add_edge("search", "generate")         # ← 변경: search 다음은 generate
graph.add_edge("generate", END)              # ← 변경: generate가 끝점


# 4) compile — 실행 가능한 형태로 굳힌다
app = graph.compile()             # ← 여기서 비로소 app 이 생긴다










"""
result = {
    # 1. 처음 입력했던 데이터가 그대로 유지됨
    "purchase_request": {
        "item": "볼트", 
        "quantity": 100, 
        "due_date": "2026-07-30"
    },
    
    # 2. search 노드를 거치면서 새롭게 추가된 데이터 (리스트 형태!)
    "retrieved_docs": [
        Document(page_content="...", metadata={"article": "..."}),  # [1]번 문서
        Document(page_content="...", metadata={"article": "..."}),  # [2]번 문서
        Document(page_content="...", metadata={"article": "..."})   # [3]번 문서 (k=3 이므로 총 3개)
    ]

"""


if __name__ == "__main__":
    result = app.invoke({
        "purchase_request": {"item": "볼트","quantity" : "100", "due_date": "2026-07-30"}  # quantity 일부러 뺌
    })

    print("=== 그릇에 든 칸들 ===")
    print("키 목록:", list(result.keys()))
    print("누락 항목:", result["missing_fields"])

    # 누락 항목이 있으면 여기서 멈춤 — 검색·RFQ 출력 건너뜀
    if result["missing_fields"]:
        print(f"\n  필수 항목이 빠졌습니다: {result['missing_fields']}")
        print("→ 이 항목들을 채워야 RFQ를 생성할 수 있습니다.")
    else:
        # 끝까지 간 경우에만 검색 결과·RFQ 출력
        print("\n=== 검색된 조항 ===")
        for i, doc in enumerate(result["retrieved_docs"], 1):
            article = doc.metadata.get("article", "조항 미상")
            print(f"[{i}] {article}")
            print(doc.page_content[:100], "...\n")
        print("\n" + "=" * 40)
        print(result["rfq_draft"])