import math


## 코사인 공식 : cosine(A, B) = (A·B) / (|A| × |B|)

# 1. A·B (dot product) : A와 B의 각 요소를 곱한 후 모두 더한 값

def dot_product(a, b):
    result = 0
    for i in range(len(a)):
	    c = a[i] * b[i]
	    result = result + c 
    return result


# 테스트
print(dot_product([1, 2, 3], [4, 5, 6]))  # 32가 나오면 성공


# 2. |A| (norm) : A 벡터의 크기, 즉 A의 각 요소를 제곱한 후 모두 더한 값에 루트를 씌운 것

def norm(a):
    # a = [3, 4] 이면 5.0 이 나와야 함
    total = 0
    for i in range(len(a)):
        total = total + a[i] * a[i]
    return math.sqrt(total)   # 마지막에 루트 씌우기


# 테스트
print(norm([3, 4]))  # 5.0 이 나오면 성공



# 3. 코사인 유사도 계산 : A와 B의 코사인 유사도는 A·B를 |A|와 |B|의 곱으로 나눈 값
def cosine_similarity(a, b):
    result = dot_product(a, b) / (norm(a) * norm(b))
    return result


# 테스트 1: 방향이 완전히 같은 벡터 → 1.0
print(cosine_similarity([1, 2, 3], [1, 2, 3]))   # 1.0

# 테스트 2: 방향이 같고 크기만 다른 벡터 → 1.0
print(cosine_similarity([1, 2, 3], [2, 4, 6]))   # 1.0 (크기 2배지만 방향 같음!)

# 테스트 3: 직각(완전 무관) → 0.0
print(cosine_similarity([1, 0], [0, 1]))          # 0.0


import numpy as np

# 4. numpy 버전으로 코사인 유사도 계산하기
a = np.array([1, 2, 3])
b = np.array([2, 4, 6])

# numpy 버전 — 한 줄씩 손코딩과 대조
dot = np.dot(a, b)              # 내 dot_product와 같은 역할
norm_a = np.linalg.norm(a)     # 내 norm과 같은 역할
norm_b = np.linalg.norm(b)

cosine = dot / (norm_a * norm_b)
print(cosine)   # 1.0 — 손코딩 결과랑 같아야 함