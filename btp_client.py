import os
import requests
from dotenv import load_dotenv

load_dotenv()

UAA_URL = os.getenv("BTP_UAA_URL")
BASE_URL = os.getenv("BTP_BASE_URL")
CLIENT_ID = os.getenv("BTP_CLIENT_ID")
CLIENT_SECRET = os.getenv("BTP_CLIENT_SECRET")

ODATA_PATH = "/sap/opu/odata4/sap/zsb_pr_request_api/srvd_a2x/sap/zsd_pr_request/0001/PurchaseRequest"


def get_token() -> str:
    """XSUAA client credentials로 액세스 토큰 발급."""
    res = requests.post(
        f"{UAA_URL}/oauth/token",
        data={"grant_type": "client_credentials"},
        auth=(CLIENT_ID, CLIENT_SECRET),
    )
    res.raise_for_status()
    return res.json()["access_token"]


def fetch_purchase_request(req_no: str) -> dict:
    """요청번호로 BTP의 구매요청 1건 조회 → dict 반환."""
    token = get_token()
    # OData V4 키 조회: .../PurchaseRequest('PR-2026-001')
    url = f"{BASE_URL}{ODATA_PATH}('{req_no}')"
    res = requests.get(
        url,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    res.raise_for_status()
    d = res.json()
    # OData V4 단건 응답은 최상위에 필드가 바로 옴
    return {
        "item": d.get("Item"),
        "quantity": d.get("Quantity"),
        "due_date": d.get("DueDate"),
        "budget": d.get("Budget"),
    }


if __name__ == "__main__":
    # 먼저 전체 목록 조회로 연결 확인
    token = get_token()
    res = requests.get(
        f"{BASE_URL}{ODATA_PATH}",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    print("OData 상태코드:", res.status_code)
    print(res.text[:800])

    print("\n--- 단건 조회 테스트 ---")
    print(fetch_purchase_request("PR-2026-001"))