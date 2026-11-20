"""Tải ảnh S1 trạng thái cho đúng các ngày cần của một lưu vực.

Ngày mục tiêu t lấy từ su_kien.chon_ngay_muc_tieu (ngày trong trận, số ngày không lũ cùng mùa bằng số ngày lũ, mọi
ngày test), ghi data/mau/ngay_muc_tieu/{lv}.json cho Task 9. Với mỗi ô, ngày có ảnh = lịch chụp S1 thật trên GEE (lưu
ở data/s1/_lich/{tile}.json) hợp với ngày ô có nhãn GFM; chỉ dùng ngày nhãn thì hụt vì sản phẩm GFM nhiều ngày không
phủ ô dù S1 có chụp. Ảnh cần tải là ảnh mới nhất <= t - L trong 30 ngày, L = 1..3.
Chạy: .venv/bin/python scripts/tai_s1.py --luu_vuc huong [--chi_dem]
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import ee
import pandas as pd
from pyproj import Transformer
from shapely.ops import transform

from floodlab import s1, su_kien
from floodlab.config import CRS, DATA_DIR, EE_PROJECT
from floodlab.luoi import luoi_o


def _ngay(f):
    s = f.stem
    return date(int(s[:4]), int(s[4:6]), int(s[6:8]))


_SANG_4326 = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True).transform


def _lich(tile_id, o_utm, tu: date, den: date) -> list[date]:
    """Ngày chụp S1 thật của ô trong [tu, den], lưu json để chạy lại không hỏi GEE lần nữa."""
    p = s1.THU_MUC / "_lich" / f"{tile_id}.json"
    if p.exists():
        j = json.loads(p.read_text())
        if j["tu"] <= tu.isoformat() and j["den"] >= den.isoformat():
            return [date.fromisoformat(n) for n in j["ngay"]]
    ngay = s1.ngay_chup(transform(_SANG_4326, o_utm), tu, den)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"tu": tu.isoformat(), "den": den.isoformat(), "ngay": [n.isoformat() for n in ngay]}))
    return ngay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--luu_vuc", required=True)
    ap.add_argument("--chi_dem", action="store_true")
    a = ap.parse_args()
    lv = a.luu_vuc
    d = pd.read_csv(DATA_DIR / "nhan" / "_ty_le" / f"{lv}.csv", parse_dates=["ngay"])
    d = su_kien.bo_thang_cay_lua(d, lv)
    nguong = su_kien.nguong_tran(d[d.loai.isin(["mua_mua", "mua_kho"])])
    muc_tieu = su_kien.chon_ngay_muc_tieu(d, nguong, seed=0)
    ra = DATA_DIR / "mau" / "ngay_muc_tieu" / f"{lv}.json"
    ra.parent.mkdir(parents=True, exist_ok=True)
    ra.write_text(json.dumps({"nguong": nguong, "ngay": {n.isoformat(): v for n, v in sorted(muc_tieu.items())}},
                             indent=1))
    ee.Initialize(project=EE_PROJECT)
    o = luoi_o(lv)
    tu, den = min(muc_tieu) - timedelta(days=33), max(muc_tieu)
    with ThreadPoolExecutor(6) as ex:
        lich = dict(zip(o.tile_id, ex.map(lambda t: _lich(t.tile_id, t.geometry, tu, den), o.itertuples())))
    viec, them = [], 0
    for t in o.itertuples():
        nhan = {_ngay(f) for f in (DATA_DIR / "nhan" / t.tile_id).glob("*.npy")}
        co = nhan | set(lich[t.tile_id])
        mt = [n for n in muc_tieu if n in nhan]
        can = s1.ngay_trang_thai_can(sorted(co), mt)
        them += len(can - s1.ngay_trang_thai_can(sorted(nhan), mt))
        viec += [(t.tile_id, t.geometry, n) for n in sorted(can)]
    da_co = sum((s1.THU_MUC / tid / f"{n:%Y%m%d}.npy").exists() for tid, _, n in viec)
    print(f"{lv}: {len(muc_tieu)} ngày mục tiêu {pd.Series(muc_tieu).value_counts().to_dict()}, "
          f"{len(viec)} lượt ô-ngày S1 (đã có {da_co}, {them} lượt nhờ lịch chụp GEE)", flush=True)
    if a.chi_dem:
        return
    t_bd, xong, rong = time.time(), 0, 0
    with ThreadPoolExecutor(6) as ex:
        for p in ex.map(lambda v: s1.tai(*v), viec):
            xong += 1
            rong += p is None
            if xong % 200 == 0:
                print(f"  {xong}/{len(viec)} ({rong} không có ảnh, {time.time() - t_bd:.0f} s)", flush=True)
    print(f"{lv} xong: {xong} lượt, {rong} không có ảnh, {time.time() - t_bd:.0f} s", flush=True)


if __name__ == "__main__":
    main()
