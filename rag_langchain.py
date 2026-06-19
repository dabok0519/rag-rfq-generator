from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("표준하도급계약서.pdf")   # ← 어제 쓴 PDF 경로
docs = loader.load()          # ← 여기서 뭐가 나올까? -> PDF 내의 텍스트 추출 

# 검증 1: 몇 개가 나왔나?
print(len(docs))

# 검증 2: 첫 번째 요소의 정체는?
print(type(docs[0]))                #  LangChain의 표준 데이터 그릇
print(docs[0].page_content[:200])   # 내용 앞부분 ( 200글자까지만 출력 )
print(docs[0].metadata)             # 현재 page_content에 대한 메타 데이터 출력 


from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # 한 청크 최대 글자 수 => chunk_size=500은 상한선(최대)이지 정확히 맞춰야 할 목표가 아님 
    chunk_overlap=10,     # 청크 간 겹치는 글자 수 
)

chunks = splitter.split_documents(docs)   # ← docs는 STEP 2의 그 43개

# 검증
print(len(chunks))                 # chunk_size를 정해 준 후 총 몇개의 청크로 분리되었나 ? 
print(len(chunks[0].page_content)) # 첫 청크 길이는 얼마인가 ? 
print(chunks[0].metadata)          # ← 청크를 쪼갰을 때 메타데이터는 어떻게 처리되는가 ? 



import os, time
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# 1) 빈 vectorstore 먼저 만들기 (저장만, 임베딩 호출 X)
vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

# 2) 청크를 배치로 끊어서 넣기
batch_size = 50        # ← 한 번에 몇 개? 한도가 분당 100이니 안전하게...
for i in range(0, len(chunks), batch_size):
    batch = chunks[i : i + batch_size]
    vectorstore.add_documents(batch)
    print(f"{i + len(batch)}/{len(chunks)} 저장 완료")
    time.sleep(30)     # ← 다음 배치 전에 몇 초 쉴까? (분당 100 한도 고려)

print("최종:", vectorstore._collection.count())