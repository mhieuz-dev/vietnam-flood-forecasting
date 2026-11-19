"""Khóa tập test bằng mã băm và cho đọc tập test qua một cửa duy nhất có ghi nhật ký."""
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd

from floodlab.config import CAU_HINH_DIR, RESULTS_DIR, ROOT

KHOA = CAU_HINH_DIR / "tap_test_khoa.json"
NHAT_KY = RESULTS_DIR / "nhat_ky_test.log"
COT = ["tile_id", "ngay_t", "vai_tro", "file_nhan"]


def _bam_file(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _bam_tong(dong: list[dict]) -> str:
    return hashlib.sha256(json.dumps(dong, sort_keys=True).encode()).hexdigest()


def khoa_tap_test(ds_mau: pd.DataFrame, path: Path = KHOA) -> str:
    """Ghi (tile_id, ngay_t) của test_a, test_b kèm SHA-256 từng file nhãn; trả băm tổng."""
    test = ds_mau[ds_mau["vai_tro"].isin(["test_a", "test_b"])][COT]
    dong = [{**r, "sha256": _bam_file(r["file_nhan"])}
            for r in test.sort_values(["vai_tro", "tile_id", "ngay_t"]).to_dict("records")]
    tong = _bam_tong(dong)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"bam_tong": tong, "mau": dong}, indent=1, ensure_ascii=False), encoding="utf-8")
    return tong


def _git_hash() -> str:
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip() or os.environ.get("FLOODLAB_GIT", "khong_co_git")  # Kaggle không có .git


def doc_test(ten: Literal["test_a", "test_b"], ly_do: str, path: Path = KHOA) -> pd.DataFrame:
    khoa = json.loads(Path(path).read_text(encoding="utf-8"))
    dong = khoa["mau"]
    if _bam_tong(dong) != khoa["bam_tong"] or any(_bam_file(r["file_nhan"]) != r["sha256"] for r in dong):
        raise RuntimeError("tap test da thay doi")
    NHAT_KY.parent.mkdir(parents=True, exist_ok=True)
    with open(NHAT_KY, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()}\t{_git_hash()}\t{ten}\t{ly_do}\n")
    return pd.DataFrame([r for r in dong if r["vai_tro"] == ten])
