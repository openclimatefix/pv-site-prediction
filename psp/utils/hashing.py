import hashlib
from typing import Any


def naive_hash(x: Any) -> int:
    return int(hashlib.sha1(str(x).encode()).hexdigest(), 16)
