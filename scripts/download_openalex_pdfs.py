#!/usr/bin/env python3
"""
Download open-access PDFs for an OpenAlex works CSV.

Usage from repo root:
    pip install pandas requests tqdm
    python scripts/download_openalex_pdfs.py \
        --csv data/metadata/works.csv \
        --out data/raw_pdfs \
        --log data/metadata/openalex_download_log.csv \
        --email your_email@example.com

What it does:
- Reads OpenAlex work IDs from the CSV.
- Calls the OpenAlex API for each work to get OA PDF URLs.
- Downloads only responses that are actually PDFs.
- Writes a full log so you know what downloaded/skipped/failed.
"""

import argparse
import csv
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from tqdm import tqdm


def safe_filename(text: str, max_len: int = 90) -> str:
    text = str(text or "untitled").lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:max_len] or "untitled"


def normalize_openalex_id(value: str) -> str | None:
    if not value or pd.isna(value):
        return None
    value = str(value).strip()
    if value.startswith("https://openalex.org/"):
        return value.rsplit("/", 1)[-1]
    if value.startswith("W"):
        return value
    return None


def get_nested(d: dict, path: list[str]):
    cur = d
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def candidate_pdf_urls(work_json: dict) -> list[str]:
    urls = []

    # Best OpenAlex field
    best_pdf = get_nested(work_json, ["best_oa_location", "pdf_url"])
    best_landing = get_nested(work_json, ["best_oa_location", "landing_page_url"])

    if best_pdf:
        urls.append(best_pdf)

    # Other OA locations may contain PDFs too
    for loc in work_json.get("oa_locations") or []:
        pdf = loc.get("pdf_url")
        landing = loc.get("landing_page_url")
        if pdf:
            urls.append(pdf)
        # Some repositories expose direct PDF-ish URLs in landing fields
        if landing and (".pdf" in landing.lower() or "download" in landing.lower()):
            urls.append(landing)

    # Last resort: sometimes landing URL is actually a downloadable endpoint
    if best_landing and (".pdf" in best_landing.lower() or "download" in best_landing.lower()):
        urls.append(best_landing)

    # Deduplicate while preserving order
    seen = set()
    clean = []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            clean.append(u)

    return clean


def looks_like_pdf(resp: requests.Response) -> bool:
    content_type = resp.headers.get("content-type", "").lower()
    return resp.content[:4] == b"%PDF" or "application/pdf" in content_type or "pdf" in content_type


def download_pdf(url: str, out_path: Path, session: requests.Session) -> tuple[bool, str]:
    try:
        r = session.get(url, timeout=45, allow_redirects=True)
    except Exception as e:
        return False, f"request_error: {e}"

    if r.status_code != 200:
        return False, f"http_{r.status_code}"

    if not looks_like_pdf(r):
        return False, "not_pdf_response"

    out_path.write_bytes(r.content)
    return True, "downloaded"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to OpenAlex CSV export")
    parser.add_argument("--out", default="data/raw_pdfs", help="Folder to save PDFs")
    parser.add_argument("--log", default="data/metadata/openalex_download_log.csv", help="CSV log path")
    parser.add_argument("--email", required=True, help="Your email for OpenAlex polite pool")
    parser.add_argument("--sleep", type=float, default=0.2, help="Delay between OpenAlex requests")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of rows to process")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out)
    log_path = Path(args.log)

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if args.limit:
        df = df.head(args.limit)

    session = requests.Session()
    session.headers.update({
        "User-Agent": f"RAG course project downloader; mailto:{args.email}"
    })

    fieldnames = [
        "row_index",
        "openalex_id",
        "title",
        "year",
        "doi",
        "is_oa",
        "oa_status",
        "pdf_url_attempted",
        "status",
        "filename",
        "error",
    ]

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            title = row.get("display_name", "")
            year = row.get("publication_year", "")
            doi = row.get("doi", "")
            is_oa = row.get("open_access.is_oa", "")
            oa_status = row.get("open_access.oa_status", "")

            work_id = normalize_openalex_id(row.get("id", ""))

            base_log = {
                "row_index": idx,
                "openalex_id": work_id or "",
                "title": title,
                "year": year,
                "doi": doi,
                "is_oa": is_oa,
                "oa_status": oa_status,
                "pdf_url_attempted": "",
                "status": "",
                "filename": "",
                "error": "",
            }

            if not work_id:
                base_log["status"] = "missing_openalex_id"
                writer.writerow(base_log)
                f.flush()
                continue

            # Query OpenAlex work endpoint to retrieve full OA locations/PDF URL.
            api_url = f"https://api.openalex.org/works/{work_id}?mailto={args.email}"
            try:
                api_resp = session.get(api_url, timeout=30)
                if api_resp.status_code != 200:
                    base_log["status"] = f"openalex_http_{api_resp.status_code}"
                    writer.writerow(base_log)
                    f.flush()
                    continue
                work = api_resp.json()
            except Exception as e:
                base_log["status"] = "openalex_request_failed"
                base_log["error"] = str(e)
                writer.writerow(base_log)
                f.flush()
                continue

            urls = candidate_pdf_urls(work)

            if not urls:
                base_log["status"] = "no_pdf_url_in_openalex"
                writer.writerow(base_log)
                f.flush()
                time.sleep(args.sleep)
                continue

            downloaded = False
            last_error = ""

            for attempt_num, pdf_url in enumerate(urls, start=1):
                filename = f"{idx:03d}_{int(float(year)) if str(year) not in ['', 'nan'] else 'no_year'}_{safe_filename(title)}.pdf"
                out_path = out_dir / filename

                ok, status = download_pdf(pdf_url, out_path, session)

                log_row = dict(base_log)
                log_row["pdf_url_attempted"] = pdf_url
                log_row["status"] = status
                log_row["filename"] = filename if ok else ""
                log_row["error"] = "" if ok else status
                writer.writerow(log_row)
                f.flush()

                if ok:
                    downloaded = True
                    break
                else:
                    last_error = status

            time.sleep(args.sleep)

    print(f"\nDone.")
    print(f"PDFs saved to: {out_dir}")
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()
