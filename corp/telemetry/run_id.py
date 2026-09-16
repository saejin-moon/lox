"""
Run ID Generator: Produces compact 6-character Base-62 hashes of timestamp/date.
"""

import hashlib
import sys
from datetime import datetime, timezone

BASE62_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def generate_base62_id(dt: datetime | None = None) -> str:
    """
    Generates a 6-character Base-62 hash from the specified (or current UTC) datetime.
    
    6 Base-62 characters provide 62^6 = 56,800,235,584 unique permutations (~35.7 bits of entropy),
    ensuring collision-free run identifiers across distributed runs while remaining ultra-compact.
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    
    # High-precision ISO string (including microseconds and timezone)
    date_str = dt.isoformat()
    digest = hashlib.sha256(date_str.encode("utf-8")).digest()
    
    # Read first 8 bytes as big-endian unsigned integer
    val = int.from_bytes(digest[:8], byteorder="big")
    
    chars = []
    for _ in range(6):
        chars.append(BASE62_ALPHABET[val % 62])
        val //= 62
        
    return "".join(chars)


def generate_run_id(prefix: str | None = None, dt: datetime | None = None) -> str:
    """
    Generates a run ID with optional prefix:
    If prefix is provided: '<prefix>_<base62_hash>' (e.g. 'train_7k9aXb')
    If prefix is empty/None: '<base62_hash>' (e.g. '7k9aXb')
    """
    token = generate_base62_id(dt)
    if prefix:
        return f"{prefix}_{token}"
    return token


if __name__ == "__main__":
    prefix = sys.argv[1] if len(sys.argv) > 1 else None
    print(generate_run_id(prefix))
