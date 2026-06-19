
from langchain_community.document_loaders import  Docx2txtLoader
import re # 정규식 패턴 
from langchain_core.documents import Document # # LangChain 표준 데이터 그릇 - 우리가 직접 만들 때 사용

from langchain_text_splitters import RecursiveCharacterTextSplitter   




loader = Docx2txtLoader("표준하도급계약서.docx")   # ← 어제 쓴 PDF 경로
docs = loader.load()          # ← 여기서 뭐가 나올까? -> PDF 내의 텍스트 추출 

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=10,
)


# 검증 1: 몇 개가 나왔나?
print(len(docs))

# 검증 2: 첫 번째 요소의 정체는?

# 1) "제○조(제목)" 위치를 모두 찾기
print(type(docs[0]))                #  LangChain의 표준 데이터 그릇
print(docs[0].page_content[:200])   # 내용 앞부분 ( 200글자까지만 출력 )
print(docs[0].metadata)             # 현재 page_content에 대한 메타 데이터 출력 


# ===== 조항별 분할 (splitter 전에!) =====
full_text = docs[0].page_content   # DOCX는 한 덩어리로 들어옴


# 2) 각 조항 시작~다음 조항 시작 전까지를 한 조각으로 자르기
pattern = r'제\s*\d+\s*조\s*\([^)]*\)'
matches = list(re.finditer(pattern, full_text))  # finditer는 위치(start)까지 줌

article_docs = []
for i, m in enumerate(matches):
    title = m.group()  # 예: '제13조(목적물의 제조등)'
    start = m.start()  # 이 조항이 시작하는 위치
    if i + 1 < len(matches):  # 다음 조항 시작 위치 (마지막이면 끝까지)
        end = matches[i+1].start()
    else:
        end = len(full_text)
    body = full_text[start:end]  # 조항 본문 통째로
    article_docs.append(
        Document(page_content=body, metadata={"source": "표준하도급계약서.docx", "article": title})
    )

print("조항 개수:", len(article_docs))   # 90 근처 떠야 정상


chunks = splitter.split_documents(article_docs)   # docs(통짜) 아니라 article_docs!




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