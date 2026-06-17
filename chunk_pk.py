import chromadb

client = chromadb.Client() # 새로운  Chroma client 객체 생성 
collection = client.create_collection(name="my_docs") # 새로운 컬렉션 생성 
                                                      # 보관함( 컬렉션 ) 이름 

chunks = [
    "출장비 한도는 1일 10만원이다.",
    "식대는 영수증 제출 시 실비 처리한다.",
    "법인카드는 사전 승인 후 사용 가능하다.",
]

collection.add( #  documents나 ids라는 단어를 정의한 적이 없지만, collection.add(...)라는 기계를 사용할 때 사용하는 규칙 그 자체 
    documents=chunks, # 청크를 실제로 저장할 보관소 
    ids=["c0", "c1", "c2"],   # 각 청크의 고유 이름표 (중복 불가)
)

results = collection.query(
    query_texts=["출장 가면 하루에 얼마까지 써도 돼?"],
    n_results=2,   # 가까운 청크 top 2
)

print(results["documents"])