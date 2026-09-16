#!/usr/bin/env python3
"""
Downloads the official NetHackWiki MediaWiki XML dump from alt.org.
Supports resuming, chunked streaming, and gzip verification.
"""

import os
import sys
import time
import gzip
import requests

WIKI_DUMP_URL = "https://archive.alt.org/nethackwiki/nethackwiki_current.xml.gz"
TARGET_DIR = os.path.join(os.path.dirname(__file__), "../../data/raw")
TARGET_FILE = os.path.join(TARGET_DIR, "nethackwiki_current.xml.gz")


def download_dump(url: str = WIKI_DUMP_URL, target_path: str = TARGET_FILE) -> str:
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    # Check if complete valid gzip already exists
    if os.path.exists(target_path) and os.path.getsize(target_path) > 30 * 1024 * 1024:
        try:
            with gzip.open(target_path, "rb") as f:
                f.read(1024)
            print(f"[✓] Existing valid wiki dump found: {target_path} ({os.path.getsize(target_path) / (1024*1024):.1f} MB)")
            return target_path
        except Exception:
            print("[!] Existing file corrupted, re-downloading...")
            os.remove(target_path)

    existing_bytes = os.path.getsize(target_path) if os.path.exists(target_path) else 0
    headers = {"User-Agent": "NetHack-Cognitive-OS-ETL/1.0"}
    if existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"
        print(f"[*] Resuming download from byte {existing_bytes}...")
    else:
        print(f"[*] Initiating download from {url}...")

    response = requests.get(url, headers=headers, stream=True, timeout=30)
    
    # Handle server response code
    if response.status_code == 416:  # Range Not Satisfiable -> already complete
        print("[✓] File is already completely downloaded.")
        return target_path
    elif response.status_code not in (200, 206):
        raise RuntimeError(f"HTTP download failed with status {response.status_code}: {response.text[:200]}")

    total_bytes = int(response.headers.get("content-length", 0)) + existing_bytes
    mode = "ab" if existing_bytes > 0 else "wb"

    downloaded = existing_bytes
    start_time = time.time()
    last_log_time = start_time

    with open(target_path, mode) as f:
        for chunk in response.iter_content(chunk_size=1024 * 128):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                now = time.time()
                if now - last_log_time >= 1.0:
                    pct = (downloaded / total_bytes * 100) if total_bytes > 0 else 0
                    speed = (downloaded - existing_bytes) / (1024 * 1024 * (now - start_time) + 1e-5)
                    print(f"[*] Progress: {downloaded / (1024*1024):.1f} MB / {total_bytes / (1024*1024):.1f} MB ({pct:.1f}%) - {speed:.2f} MB/s", end="\r")
                    last_log_time = now

    print(f"\n[✓] Download complete: {target_path} ({downloaded / (1024*1024):.1f} MB in {time.time() - start_time:.1f}s)")

    # Verify gzip integrity
    with gzip.open(target_path, "rb") as f:
        f.read(1024)
    print("[✓] Gzip integrity verified successfully.")
    return target_path


if __name__ == "__main__":
    download_dump()
