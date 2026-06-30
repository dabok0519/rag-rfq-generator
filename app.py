import streamlit as st
from rfq_graph import app   # 컴파일된 그래프 가져오기


def format_won(amount):
    """원 단위 정수를 한국식 표기로. 1200000 → '1,200,000원 (120만원)'"""
    try:
        amount = int(amount)
    except (ValueError, TypeError):
        return str(amount)
    exact = f"{amount:,}원"
    if amount >= 10000 and amount % 10000 == 0:
        return f"{exact} ({amount // 10000:,}만원)"
    return exact


st.title("RAG 기반 RFQ 생성기")   # 화면 맨 위 제목

# --- 입력 폼 ---
item = st.text_input("품목", placeholder="예: 볼트, 강판, 노트북, 유지보수 용역")
quantity = st.number_input("수량",min_value = 0 , step = 1)
unit = st.selectbox("단위", ["개(EA)", "kg", "m", "롤", "세트", "식", "건"])
due_date = st.date_input("납기일") 
budget = st.number_input("예산 (원)", min_value=0, step=100000, format="%d")

# --- 생성 버튼 ---
if st.button("RFQ 생성"):
    # 1) 입력값을 dict로 묶기
    purchase_request = {
        "item": item,
        "quantity": f"{quantity} {unit}",   # ← 수량+단위를 코드가 직접 조립
        "due_date": due_date.strftime("%Y-%m-%d"),   # date → 문자열로 변환 
        "budget": format_won(budget),
    }

    # 2) 그래프 호출
    result = app.invoke({"purchase_request": purchase_request})

    # 3) 결과 분기 — 여기를 채워보세요
    # result["missing_fields"]가 비었으면 → st.markdown으로 rfq_draft 출력
    # 뭔가 있으면 → st.warning으로 안내
    if result["missing_fields"]:           # 누락 항목이 존재하면 
        st.warning(f"필수 항목이 빠졌습니다: {result['missing_fields']}")   # 화면에 띄워줌
    else:
        st.markdown(result["rfq_draft"])

         # --- STEP 3: 근거 조항 원문 펼쳐보기 ---
        with st.expander("📎 근거 조항 원문 보기"):
            for doc in result["retrieved_docs"]:
                sub = doc.metadata.get("article", "조항 미상")
                st.markdown(f"**{sub}**")
                st.text(doc.page_content)     # ← st.write에서 변경
                st.divider()
