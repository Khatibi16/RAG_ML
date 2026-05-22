import json
import csv
import re
import requests
from pathlib import Path

INPUT = Path("../data/metadata/apify_google_scholar.json")
OUT_DIR = Path("../data/raw_pdfs")
LOG = Path("../data/metadata/download_log.csv")

OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG.parent.mkdir(parents=True, exist_ok=True)

def safe_name(s):
    s = re.sub(r"[^a-zA-Z0-9]+", "_", str(s).lower()).strip("_")
    return s[:90] or "paper"

with open(INPUT, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(LOG, "w", newline="", encoding="utf-8") as log_file:
    writer = csv.DictWriter(log_file, fieldnames=[
        "title", "year", "source", "url", "status", "filename"
    ])
    writer.writeheader()

    for i, item in enumerate(data):
        title = item.get("title", "")
        year = item.get("year", "")
        source = item.get("source", "")
        url = item.get("link", "")

        filename = ""

        if not url or url == "N/A":
            status = "no_url"

        elif ".pdf" not in url.lower() and "get_pdf" not in url.lower() and "download" not in url.lower():
            status = "skipped_not_direct_pdf"

        else:
            try:
                r = requests.get(url, timeout=30, headers={
                    "User-Agent": "Mozilla/5.0"
                })

                content_type = r.headers.get("content-type", "").lower()

                if r.status_code != 200:
                    status = f"http_{r.status_code}"

                elif r.content[:4] == b"%PDF" or "pdf" in content_type:
                    filename = f"{i:03d}_{year}_{safe_name(title)}.pdf"
                    path = OUT_DIR / filename
                    path.write_bytes(r.content)
                    status = "downloaded"

                else:
                    status = "not_pdf_response"

            except Exception as e:
                status = f"failed_request: {e}"

        writer.writerow({
            "title": title,
            "year": year,
            "source": source,
            "url": url,
            "status": status,
            "filename": filename
        })

        log_file.flush()
        print(f"[{status}] {title}")