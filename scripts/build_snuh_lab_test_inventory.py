#!/usr/bin/env python3
"""Build SNUH lab-test metadata inventory from snuhlab.org.

The site exposes class-filtered list pages such as:
http://www.snuhlab.org/checkup/check_list.aspx?ins_class_code=L25&searchfield=TOTAL&searchword=

This script scrapes the list pages for selected 검사분류코드 values and writes a
local CSV/JSON inventory. It does not query the SNUH CDM.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
import re
import time
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlparse
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup


BASE_URL = "http://www.snuhlab.org"
LIST_PATH = "/checkup/check_list.aspx"
DEFAULT_CLASSES = ["L25"]
DEFAULT_OUTPUT_DIR = Path("reports/snuh_lab_test_inventory")

FIELDS = [
    "ins_class_code",
    "source_url",
    "no",
    "검사분류코드",
    "검사분류명",
    "검사항목코드",
    "검사항목명",
    "참고치",
    "단위",
    "기본검체",
    "검체용기",
    "주의사항",
    "시행일",
    "접수마감시간",
    "보고소요시간",
    "문의처",
    "검사방법",
    "검사의의",
    "비고",
    "동의어",
    "오더가능여부",
    "검체용기 이미지",
    "시행여부",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    parser.add_argument("--max-pages", type=int, default=0)
    parser.add_argument("--user-agent", default="Mozilla/5.0 FERMAT Task17 metadata audit")
    return parser.parse_args()


def now_iso():
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def log(message):
    print(f"[{now_iso()}] {message}", flush=True)


def fetch_text(url: str, user_agent: str) -> str:
    req = Request(url, headers={"User-Agent": user_agent})
    with urlopen(req, timeout=30) as response:
        raw = response.read()
    return raw.decode("utf-8", errors="replace")


def list_url(ins_class_code: str, page: int = 1) -> str:
    suffix = (
        f"{LIST_PATH}?page={page}&ins_class_code={ins_class_code}"
        "&searchfield=TOTAL&searchword="
    )
    return urljoin(BASE_URL, suffix)


def clean_text(node) -> str:
    if node is None:
        return ""
    text = node.get_text("\n", strip=True)
    text = html.unescape(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def page_numbers(soup: BeautifulSoup) -> list[int]:
    numbers = {1}
    for link in soup.find_all("a", href=True):
        href = html.unescape(link["href"])
        parsed = urlparse(urljoin(BASE_URL, href))
        if not parsed.path.endswith("check_list.aspx"):
            continue
        page_values = parse_qs(parsed.query).get("page")
        if not page_values:
            continue
        try:
            numbers.add(int(page_values[0]))
        except ValueError:
            pass
    return sorted(numbers)


def discover_last_page(ins_class_code: str, user_agent: str) -> tuple[int, str]:
    url = list_url(ins_class_code, 1)
    text = fetch_text(url, user_agent)
    soup = BeautifulSoup(text, "html.parser")
    pages = page_numbers(soup)
    return max(pages), text


def extract_no(href: str) -> str:
    parsed = urlparse(urljoin(BASE_URL, html.unescape(href)))
    return (parse_qs(parsed.query).get("no") or [""])[0]


def table_fields(table) -> dict[str, str]:
    fields = {}
    for row in table.find_all("tr"):
        th = row.find("th")
        if th is None:
            continue
        label = clean_text(th)
        cells = row.find_all("td")
        if not label or not cells:
            continue
        fields[label] = clean_text(cells[0])
    return fields


def parse_list_page(text: str, ins_class_code: str) -> list[dict[str, str]]:
    soup = BeautifulSoup(text, "html.parser")
    rows = []
    seen = set()
    for link in soup.find_all("a", href=True):
        href = html.unescape(link["href"])
        if "check_view.aspx?no=" not in href:
            continue
        no = extract_no(href)
        if not no or no in seen:
            continue
        seen.add(no)
        table = link.find_next("table")
        if table is None:
            continue
        row = table_fields(table)
        row["ins_class_code"] = ins_class_code
        row["source_url"] = urljoin(BASE_URL, href)
        row["no"] = no
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]):
    extras = sorted({key for row in rows for key in row if key not in FIELDS})
    fieldnames = FIELDS + extras
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, str]]) -> dict:
    by_class = {}
    for row in rows:
        code = row.get("검사분류코드") or row.get("ins_class_code") or ""
        by_class.setdefault(code, {"rows": 0, "item_codes": set(), "orderable_yes": 0, "active": 0})
        entry = by_class[code]
        entry["rows"] += 1
        if row.get("검사항목코드"):
            entry["item_codes"].add(row["검사항목코드"])
        if row.get("오더가능여부") == "예":
            entry["orderable_yes"] += 1
        if row.get("시행여부") == "시행중":
            entry["active"] += 1
    return {
        key: {
            **{k: v for k, v in value.items() if k != "item_codes"},
            "unique_item_codes": len(value["item_codes"]),
        }
        for key, value in sorted(by_class.items())
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    page_manifest = []

    for ins_class_code in args.classes:
        last_page, first_page_text = discover_last_page(ins_class_code, args.user_agent)
        if args.max_pages:
            last_page = min(last_page, args.max_pages)
        log(f"{ins_class_code}: pages=1..{last_page}")

        for page in range(1, last_page + 1):
            text = first_page_text if page == 1 else fetch_text(
                list_url(ins_class_code, page),
                args.user_agent,
            )
            rows = parse_list_page(text, ins_class_code)
            all_rows.extend(rows)
            page_manifest.append(
                {
                    "ins_class_code": ins_class_code,
                    "page": page,
                    "url": list_url(ins_class_code, page),
                    "rows": len(rows),
                }
            )
            log(f"{ins_class_code} page {page}: rows={len(rows)}")
            if args.sleep_seconds:
                time.sleep(args.sleep_seconds)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = args.output_dir / f"snuh_lab_test_inventory_{stamp}.csv"
    json_path = args.output_dir / f"snuh_lab_test_inventory_{stamp}.json"
    manifest_path = args.output_dir / f"snuh_lab_test_inventory_manifest_{stamp}.json"
    latest_csv = args.output_dir / "snuh_lab_test_inventory_latest.csv"
    latest_json = args.output_dir / "snuh_lab_test_inventory_latest.json"
    latest_manifest = args.output_dir / "snuh_lab_test_inventory_manifest_latest.json"

    write_csv(csv_path, all_rows)
    json_path.write_text(json.dumps(all_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "generated_at": now_iso(),
        "classes": args.classes,
        "rows": len(all_rows),
        "summary_by_class": summarize(all_rows),
        "pages": page_manifest,
        "csv": str(csv_path),
        "json": str(json_path),
    }
    manifest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    write_csv(latest_csv, all_rows)
    latest_json.write_text(json.dumps(all_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    latest_manifest.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    log(f"wrote {csv_path}")
    log(f"wrote {json_path}")
    log(f"wrote {manifest_path}")
    print(json.dumps(summary["summary_by_class"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
