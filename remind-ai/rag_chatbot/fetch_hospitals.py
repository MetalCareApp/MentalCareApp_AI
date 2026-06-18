"""
=============================================================
전국 정신건강의학과 데이터 수집 스크립트 (이어받기 지원)
=============================================================
- 건강보험심사평가원 병원정보 API 활용
- 타임아웃 30초, 실패시 3회 재시도
- START_PAGE로 이어받기 가능
=============================================================
"""

import requests
import sqlite3
import time
from pathlib import Path
from dotenv import load_dotenv
import os
import xml.etree.ElementTree as ET

load_dotenv(dotenv_path="secrets.txt")

API_KEY  = os.getenv("PUBLIC_DATA_API_KEY")
DB_PATH  = "./hospitals.db"
BASE_URL = "http://apis.data.go.kr/B551182/hospInfoServicev2/getHospBasisList"
PSYCH_CODE = "11"

# ★ 이어받기: 중단된 페이지부터 시작 (처음부터면 1)
START_PAGE = 90


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS hospitals (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            address     TEXT,
            city        TEXT,
            district    TEXT,
            dong        TEXT,
            phone       TEXT,
            postal_code TEXT,
            lat         REAL,
            lng         REAL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_city     ON hospitals(city)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_district ON hospitals(district)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_dong     ON hospitals(dong)")
    conn.commit()
    conn.close()
    print("DB 초기화 완료")


def fetch_page(page_no: int, num_of_rows: int = 100) -> list:
    params = {
        "serviceKey": API_KEY,
        "pageNo":     page_no,
        "numOfRows":  num_of_rows,
        "dgsbjtCd":   PSYCH_CODE,
        "_type":      "xml",
    }
    for attempt in range(3):
        try:
            res = requests.get(BASE_URL, params=params, timeout=30)
            res.raise_for_status()
            root = ET.fromstring(res.text)
            items = root.findall(".//item")
            return items
        except Exception as e:
            print(f"  API 오류 (페이지 {page_no}, 시도 {attempt+1}/3): {e}")
            time.sleep(2 ** attempt)
    return []


def get_total_count() -> int:
    params = {
        "serviceKey": API_KEY,
        "pageNo":     1,
        "numOfRows":  1,
        "dgsbjtCd":   PSYCH_CODE,
        "_type":      "xml",
    }
    try:
        res = requests.get(BASE_URL, params=params, timeout=30)
        root = ET.fromstring(res.text)
        total = root.find(".//totalCount")
        return int(total.text) if total is not None else 0
    except Exception as e:
        print(f"  전체 수 조회 실패: {e}")
        return 0


def parse_address(addr: str) -> tuple:
    if not addr:
        return ("", "", "")
    parts = addr.strip().split()
    city     = parts[0] if len(parts) > 0 else ""
    district = parts[1] if len(parts) > 1 else ""
    dong     = parts[2] if len(parts) > 2 else ""
    return city, district, dong


def get_current_count() -> int:
    """현재 DB에 저장된 수"""
    if not os.path.exists(DB_PATH):
        return 0
    conn = sqlite3.connect(DB_PATH)
    count = conn.execute("SELECT COUNT(*) FROM hospitals").fetchone()[0]
    conn.close()
    return count


def save_hospitals(items: list):
    conn = sqlite3.connect(DB_PATH)
    saved = 0
    for item in items:
        def get(tag):
            el = item.find(tag)
            return el.text.strip() if el is not None and el.text else ""

        name    = get("yadmNm")
        address = get("addr")
        phone   = get("telno")
        postal  = get("postNo")
        lat     = get("YPos")
        lng     = get("XPos")

        city, district, dong = parse_address(address)

        if not name:
            continue

        conn.execute("""
            INSERT INTO hospitals (name, address, city, district, dong, phone, postal_code, lat, lng)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (name, address, city, district, dong, phone, postal,
              float(lat) if lat else None,
              float(lng) if lng else None))
        saved += 1

    conn.commit()
    conn.close()
    return saved


def main():
    print("=" * 55)
    print("  전국 정신건강의학과 데이터 수집")
    print("=" * 55)

    if not API_KEY:
        print("PUBLIC_DATA_API_KEY가 secrets.txt에 없습니다.")
        return

    init_db()

    total = get_total_count()
    print(f"  전체 데이터 수: {total}개")

    if total == 0:
        print("데이터가 없습니다. API 키를 확인해주세요.")
        return

    num_of_rows  = 100
    total_pages  = (total // num_of_rows) + 1
    current_count = get_current_count()

    print(f"  현재 저장된 수: {current_count}개")
    print(f"  {START_PAGE}페이지부터 이어서 수집...")
    print()

    total_saved = current_count

    for page in range(START_PAGE, total_pages + 1):
        items = fetch_page(page, num_of_rows)
        if not items:
            print(f"  페이지 {page} 빈 응답 - 스킵")
            continue

        saved = save_hospitals(items)
        total_saved += saved

        if page % 10 == 0:
            print(f"  [{page}/{total_pages}] 저장: {total_saved}개")

        time.sleep(0.1)

    print()
    print("=" * 55)
    print("  완료!")
    print("=" * 55)
    print(f"  저장된 병원 수: {total_saved}개")
    print(f"  DB 경로: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT name, city, district, phone FROM hospitals LIMIT 5"
    ).fetchall()
    conn.close()

    print()
    print("[샘플 데이터]")
    for row in rows:
        print(f"  {row[0]} | {row[1]} {row[2]} | {row[3]}")


if __name__ == "__main__":
    main()
