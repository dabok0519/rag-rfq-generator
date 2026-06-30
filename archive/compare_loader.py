from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader

PDF = "표준하도급계약서.pdf"

# 비교할 페이지 (깨짐 봤던 p.19 근처 — 0부터 시작하니 인덱스 18)
TARGET = 18

# (A) 어제 쓰던 로더
old_docs = PyPDFLoader(PDF).load()
print("===== PyPDFLoader (기존) =====")
print(old_docs[TARGET].page_content[:400])

# (B) 새 로더
new_docs = PyMuPDFLoader(PDF).load()                        # ??? 1: PyMuPDFLoader로 똑같이 load
print("\n===== PyMuPDFLoader (신규) =====")
print(new_docs[TARGET].page_content[:400])
