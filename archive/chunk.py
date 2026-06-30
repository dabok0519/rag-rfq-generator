def chunk_text(text, size, overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap   # ← 핵심: 끝에서 overlap만큼 되돌아가서 다음 시작
    return chunks

text = "오늘은 청킹을 배운다. 청킹은 문서를 잘게 쪼개는 작업이다."
if __name__ == "__main__":          # ← 이 줄 추가
    text = "오늘은 청킹을 배운다. 청킹은 문서를 잘게 쪼개는 작업이다."
    print(chunk_text(text, size=10, overlap=3))
