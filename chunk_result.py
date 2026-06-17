from pypdf import PdfReader
import chromadb
from chunk import chunk_text   # chunk.py 안의 chunk_text를 가져옴

reader = PdfReader("표준하도급계약서.pdf")

text = ""
for page in reader.pages:
    text += page.extract_text()



chunks = chunk_text(text, size=500, overlap=50)   # STEP 2 함수 그대로


client = chromadb.Client()
collection = client.get_or_create_collection(name="pdf_docs")

collection.add(
    documents=chunks,
    ids=[f"c{i}" for i in range(len(chunks))],   # c0, c1, c2... 자동 생성
)

while True:
    question = input("\n질문 (종료하려면 q): ")
    if question == "q":
        break

    results = collection.query(query_texts=[question], n_results=3)

    print("\n--- 관련 청크 top-3 ---")
    for i, doc in enumerate(results["documents"][0], 1): # enumerate는 리스트를 돌 때 순번(번호)을 같이 붙여주는 함수
                                                         # 1번부터 번호를 붙힐거임
                                                         # results["documents"][0]: 반복문을 돌릴 대상(주로 텍스트나 문서들이 담긴 리스트)  
        print(f"\n[{i}] {doc[:200]}...")   # 너무 길면 앞 200자만