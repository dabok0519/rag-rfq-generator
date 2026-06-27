from typing import TypedDict, List  # TypedDict: 딕셔너리의 키/값 타입을 고정해 주는 도구
from langchain_core.documents import Document  # Document: LangChain 표준 문서 객체


class RFQState(TypedDict):
    """RFQ 생성 워크플로우에서 단계별 데이터를 저장하고 전달하는 상태(State) 클래스."""
    # 1. 사용자가 입력한 원본 구매요청 데이터
    #    예: {"item": "노트북", "quantity": 50, "due_date": "2026-12-31", "budget": "50,000,000원"}
    purchase_request: dict           # 사용자 구매요청 (품목·수량·납기·예산)

    # 2. RAG 검색으로 찾아낸 표준계약 조항들 (Document 리스트)
    retrieved_docs: List[Document]   # 검색된 표준조항

    missing_fields: List[str]        # 빠진 필수항목 목록 (비어있으면 통과)

    # 3. 최종 RFQ 초안 텍스트
    rfq_draft: str                   # RFQ 초안

    search_query: str                # 실제 사용된 검색어(들) — 디버깅·투명성용


import os
from datetime import datetime, timedelta
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()  # 로컬: .env에서 키 읽기

# 클라우드(Streamlit Cloud) 대응: .env가 없고 키가 아직 없으면 st.secrets에서 가져온다
if not os.getenv("GOOGLE_API_KEY"):
    try:
        import streamlit as st
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass


# ── 벡터스토어 / 리트리버 ─────────────────────────────────────────────
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)
# 멀티쿼리에서 쿼리당 적게 가져와 합치므로 k는 작게(쿼리별 상위 3건)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# 고정 검색 쿼리 (fallback 전용 — 사용자 입력과 무관, 거래조건 5개 축을 모두 포괄)
FIXED_QUERY = "대금지급 납기 품질보증 하자담보 지체상금"


# ── 품목 유형 분류 (키워드 매칭 — LLM 호출 없이 분류) ────────────────
# LLM 분류는 호출 1회를 더 쓰므로(무료 한도 절약), 키워드 사전 매칭으로 대체한다.
# trade_type은 검색어 생성의 '힌트'로만 쓰이고, 멀티쿼리가 어차피 모든 축을 훑으므로
# 사전에 없는 품목이 "기타"로 떨어져도 최종 RFQ 품질에는 거의 영향이 없다.
TRADE_TYPE_KEYWORDS = {
    "설비": ["노트북", "컴퓨터", "서버", "pc", "모니터", "프린터", "장비", "기계",
             "머신", "설비", "공구", "도구", "장치", "라우터", "스위치"],
    "원자재": ["철강", "철판", "강판", "화학", "원료", "원자재", "자재", "부품",
               "플라스틱", "수지", "도료", "볼트", "너트", "파이프", "케이블"],
    "용역": ["개발", "용역", "컨설팅", "유지보수", "라이선스", "라이센스", "구축",
             "외주", "설계", "감리", "교육", "청소", "경비", "자문", "서비스"],
}


def classify_trade_type(item: str) -> str:
    """품목명을 거래유형으로 분류한다(키워드 매칭). 매칭 없으면 '기타'."""
    text = (item or "").lower()
    for trade_type, keywords in TRADE_TYPE_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return trade_type
    return "기타"


# ── 멀티쿼리 생성 (개선 3) ───────────────────────────────────────────
QUERY_GEN_PROMPT = """너는 구매 계약 검색을 돕는 어시스턴트다.
아래 구매요청 정보를 보고, 관련 표준계약 조항을 찾기 위한 '검색어 묶음'을 생성하라.

규칙:
- 거래조건의 서로 다른 축을 각각 겨냥한 검색어를 줄 단위로 나열할 것.
  (예: 대금지급 조건 / 품질보증·검사 / 하자담보 / 납기·지체상금 등)
- 한 축에 치우치지 말고 여러 축을 고르게 포괄할 것.
- 각 줄은 공백으로 구분된 핵심 키워드만. 계약 본문 문장은 지어내지 말 것.
- 4~6줄, 한 줄당 키워드 3~6개.
- 검색어 외의 설명·번호·기호는 출력하지 말 것.

[구매요청 정보]
- 품목: {item}
- 거래유형: {trade_type}
- 예산: {budget}
- 납기: {due_date}

[검색어 묶음]"""


def build_search_queries(req: dict, trade_type: str) -> List[str]:
    """입력 + 거래유형으로 LLM에게 축별 검색어 여러 개를 생성시킨다.
    실패하거나 비면 빈 리스트를 반환(→ 호출부에서 fallback)."""
    prompt = QUERY_GEN_PROMPT.format(
        item=req.get("item", "-"),
        trade_type=trade_type,
        budget=req.get("budget", "-"),
        due_date=req.get("due_date", "-"),
    )
    try:
        raw = llm.invoke(prompt).content.strip()
    except Exception as e:
        print(f"[build_search_queries] LLM 호출 오류: {e}")
        return []

    # 줄 단위로 쪼개고, 빈 줄·불필요한 머리기호(-, *, 숫자.) 정리
    queries = []
    for line in raw.splitlines():
        line = line.strip().lstrip("-*•0123456789. )").strip()
        if line:
            queries.append(line)
    return queries


# ── 중복 조항 제거 ───────────────────────────────────────────────────
def dedup_by_article(docs: List[Document]) -> List[Document]:
    """같은 article(조항명)은 첫 번째 것만 남긴다. 순서 유지."""
    seen = set()
    result = []
    for doc in docs:
        article = doc.metadata.get("article", "조항 미상")
        if article not in seen:
            result.append(doc)
            seen.add(article)
    return result


# ── 노드: 검색 ───────────────────────────────────────────────────────
def search_node(state: RFQState) -> dict:
    """멀티쿼리 검색 노드.
    품목 분류 → 축별 검색어 생성 → 각 쿼리 검색 결과 합치고 dedup.
    생성 실패 또는 0건이면 고정 쿼리로 fallback."""
    req = state["purchase_request"]

    # ① 품목 유형 분류
    trade_type = classify_trade_type(req.get("item", ""))

    # ② 축별 검색어 여러 개 생성
    queries = build_search_queries(req, trade_type)

    # 가드 ①: 생성 실패(빈 리스트)면 고정 쿼리로 fallback
    if not queries:
        queries = [FIXED_QUERY]

    # ③ 각 쿼리로 검색해 모두 합친 뒤 중복 조항 제거
    merged: List[Document] = []
    for q in queries:
        merged.extend(retriever.invoke(q))
    docs = dedup_by_article(merged)

    # 가드 ②: 그래도 0건이면 고정 쿼리로 재검색
    if not docs:
        queries = [FIXED_QUERY]
        docs = dedup_by_article(retriever.invoke(FIXED_QUERY))

    return {"retrieved_docs": docs, "search_query": " | ".join(queries)}


# ── 노드: 분석(필수항목 검증) ────────────────────────────────────────
def analyze_node(state: RFQState) -> dict:
    """필수항목이 다 있는지 검증해서 missing_fields 칸을 채운다."""
    req = state["purchase_request"]
    required = ["item", "quantity", "due_date"]
    missing = [k for k in required if k not in req]
    return {"missing_fields": missing}


def route_after_analyze(state: RFQState) -> str:
    """누락 항목 유무로 분기. 비었으면 search(ok), 빠지면 멈춤(missing)."""
    return "ok" if not state["missing_fields"] else "missing"


# ── 거래조건 생성 프롬프트 ───────────────────────────────────────────
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


def format_context(docs: List[Document]) -> str:
    """검색된 Document 리스트를 출처 포함 문자열로 합친다."""
    blocks = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        article = doc.metadata.get("article", "조항 미상")
        blocks.append(f"[근거 {i+1}] (출처: {source}, {article})\n{doc.page_content}")
    return "\n\n".join(blocks)


# ── 노드: RFQ 초안 생성 ──────────────────────────────────────────────
def generate_node(state: RFQState) -> dict:
    """RFQ 초안 생성 노드 — LLM은 거래조건만 생성, 나머지 값은 코드가 직접 조립(B안)."""
    req = state["purchase_request"]
    docs = state["retrieved_docs"]

    # 가드 ①: 검색 결과 0건이면 근거 없이 생성하지 않는다
    if not docs:
        return {"rfq_draft": "⚠️ 관련 표준조항을 찾지 못해 RFQ를 생성할 수 없습니다."}

    # 가드 ②: LLM 호출은 try/except로 감싼다 (거래조건만 LLM 생성)
    context = format_context(docs)
    prompt = RFQ_CLAUSE_PROMPT.format(context=context)
    try:
        clause = llm.invoke(prompt).content
    except Exception as e:
        return {"rfq_draft": f"⚠️ 거래조건 생성 중 오류가 발생했습니다: {e}"}

    # 출처 목록 (근거로 쓴 조항)
    sources = ", ".join(doc.metadata.get("article", "조항 미상") for doc in docs)

    # 문서 메타 (오늘 날짜 기준 자동 생성)
    today = datetime.now()
    doc_no = f"RFQ-{today.strftime('%Y%m%d')}-001"
    issue_date = today.strftime('%Y-%m-%d')
    reply_due = (today + timedelta(days=7)).strftime('%Y-%m-%d')  # 회신기한: 발행일+7일

    # f-string 직접 조립 — 사용자 입력값(①)은 LLM을 거치지 않는다(무결성 보장)
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


# ── 그래프 구성 ──────────────────────────────────────────────────────
from langgraph.graph import StateGraph, END

graph = StateGraph(RFQState)

graph.add_node("search", search_node)
graph.add_node("analyze", analyze_node)
graph.add_node("generate", generate_node)

graph.set_entry_point("analyze")
graph.add_conditional_edges(
    "analyze",
    route_after_analyze,
    {
        "ok": "search",     # 필수항목 OK → 검색
        "missing": END,     # 누락 → 종료
    },
)
graph.add_edge("search", "generate")
graph.add_edge("generate", END)

app = graph.compile()


# ── 단독 실행 테스트 ─────────────────────────────────────────────────
if __name__ == "__main__":
    result = app.invoke({
        "purchase_request": {
            "item": "노트북",
            "quantity": "50",
            "due_date": "2026-12-31",
            "budget": "50,000,000원",
        }
    })

    print("=== 키 목록:", list(result.keys()))
    print("=== 누락 항목:", result["missing_fields"])

    if result["missing_fields"]:
        print(f"\n필수 항목이 빠졌습니다: {result['missing_fields']}")
    else:
        print("\n=== 실제 사용된 검색어(축별) ===")
        print(result["search_query"])

        print("\n=== 검색된 조항 ===")
        for i, doc in enumerate(result["retrieved_docs"], 1):
            print(f"[{i}] {doc.metadata.get('article', '조항 미상')}")

        print("\n" + "=" * 40)
        print(result["rfq_draft"])