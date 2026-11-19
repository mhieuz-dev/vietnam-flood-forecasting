import json

import numpy as np
import pandas as pd
import pytest

from floodlab import tong_hop
from floodlab.config import ROOT


def _json(p, d):
    p.write_text(json.dumps(d), encoding="utf-8")
    return str(p)


def test_doc_khoa_va_tinh_lai(tmp_path):
    a = _json(tmp_path / "a.json", {"tong": {"csi": 0.2, "ci": [0.2, 0.1, 0.3]}})
    b = _json(tmp_path / "b.json", {"tong": {"csi": 0.4, "ci": [0.4, 0.3, 0.5]}})
    assert tong_hop.doc_khoa(json.load(open(a)), "tong/ci/2") == 0.3
    assert tong_hop.tinh("gt", [a], "tong/csi") == 0.2
    assert tong_hop.tinh("tb", [a, b], "tong/csi") == pytest.approx(0.3)
    assert tong_hop.tinh("dlc", [a, b], "tong/csi") == pytest.approx(np.std([0.2, 0.4], ddof=1))


def test_o_so_kem_nguon_va_kiem_khop(tmp_path, monkeypatch):
    monkeypatch.setattr(tong_hop, "ROOT", tmp_path)
    a = _json(tmp_path / "a.json", {"tong": {"csi": 0.24151}})
    sl = tong_hop.SoLieu()
    o = sl.o("x/csi", "gt", [a], "tong/csi")
    assert "0.242" in o and "% nguon: x/csi" in o
    f_sl = tmp_path / "tong_hop.json"
    sl.luu(f_sl)
    bang = tmp_path / "b.tex"
    bang.write_text("\\begin{tabular}{lr}\n\\toprule\nMô hình & CSI \\\\\n\\midrule\nB3 &\n" + o + " \\\\\n"
                    "\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
    assert tong_hop.kiem_bang(bang, f_sl) == []
    bang.write_text(bang.read_text().replace("0.242", "0.250"), encoding="utf-8")
    assert tong_hop.kiem_bang(bang, f_sl)  # số bị sửa tay
    bang.write_text(bang.read_text().replace("B3 &", "B3 & 0.5 &"), encoding="utf-8")
    assert any("không có nguồn" in e for e in tong_hop.kiem_bang(bang, f_sl))
    a2 = _json(tmp_path / "a.json", {"tong": {"csi": 0.3}})
    assert tong_hop.kiem_nguon(f_sl) and a2  # json gốc đổi thì tong_hop.json lệch


def _dem(n_o, tp, fp, fn):
    return pd.DataFrame({"tile_id": [f"o{i}" for i in range(n_o)], "tp": tp, "fp": fp, "fn": fn})


def test_bootstrap_cap():
    rng = np.random.default_rng(0)
    n = 200
    tp = rng.integers(50, 100, n)
    tot = _dem(n, tp, rng.integers(0, 10, n), rng.integers(0, 10, n))
    kem = _dem(n, tp // 3, rng.integers(40, 80, n), rng.integers(40, 80, n))
    r = tong_hop.bootstrap_cap(tot, kem, B=500, seed=0)
    assert r["hieu"] > 0 and r["p"] < 0.01 and r["thap"] > 0
    r = tong_hop.bootstrap_cap(tot, tot.copy(), B=500, seed=0)
    assert r["hieu"] == 0 and r["p"] == 1.0


def test_holm():
    assert tong_hop.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert tong_hop.holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])


def test_moi_so_trong_bang_co_nguon():
    f_sl = ROOT / "results" / "tong_hop.json"
    bang = sorted((ROOT / "bai_bao" / "bang").glob("*.tex")) + [ROOT / "bai_bao" / "so_lieu.tex"]
    assert f_sl.exists() and bang, "chưa chạy scripts/tong_hop.py"
    loi = [e for b in bang for e in tong_hop.kiem_bang(b, f_sl)] + tong_hop.kiem_nguon(f_sl)
    assert loi == [], loi[:10]


def test_macro_so_lieu_duoc_kiem(tmp_path, monkeypatch):
    monkeypatch.setattr(tong_hop, "ROOT", tmp_path)
    a = _json(tmp_path / "a.json", {"tong": {"csi": 0.24151}})
    sl = tong_hop.SoLieu()
    m = "\\newcommand{\\sX}{" + sl.o("x/csi", "gt", [a], "tong/csi").replace("% nguon", "}% nguon")
    sl.luu(tmp_path / "th.json")
    f = tmp_path / "so_lieu.tex"
    f.write_text(m + "\n", encoding="utf-8")
    assert tong_hop.kiem_bang(f, tmp_path / "th.json") == []
    f.write_text(m.replace("0.242", "0.300") + "\n\\newcommand{\\sY}{0.5}\n", encoding="utf-8")
    assert len(tong_hop.kiem_bang(f, tmp_path / "th.json")) == 2


def _seed(rng, n, tp_mu, nhieu_seed):
    """Một seed giả: tp quanh tp_mu nhân hệ số dao động theo seed (nhieu_seed), fp/fn ngẫu nhiên theo ô."""
    k = 1 + rng.normal(0, nhieu_seed)
    return _dem(n, (rng.integers(40, 60, n) * tp_mu * k).astype(int), rng.integers(20, 40, n), rng.integers(20, 40, n))


def test_bootstrap_hai_tang_rong_hon_mot_tang_khi_seed_dao_dong():
    rng = np.random.default_rng(1)
    n = 300
    a = [_seed(rng, n, 1.0, 0.25) for _ in range(3)]
    b = [_seed(rng, n, 1.0, 0.25) for _ in range(3)]
    mot = tong_hop.bootstrap_cap(pd.concat(a), pd.concat(b), B=1000, seed=0)
    hai = tong_hop.bootstrap_cap_phan_cap(a, b, B=1000, seed=0)
    assert hai["hieu"] == pytest.approx(mot["hieu"])  # ước lượng điểm như nhau (gộp mọi seed)
    assert (hai["cao"] - hai["thap"]) > 1.5 * (mot["cao"] - mot["thap"])


def test_bootstrap_hai_tang_ghep_seed_va_tat_dinh():
    rng = np.random.default_rng(2)
    n = 200
    a = [_seed(rng, n, 1.0, 0.0) for _ in range(3)]
    r = tong_hop.bootstrap_cap_phan_cap(a, a, B=500, seed=0, ghep_seed=True)
    assert r["hieu"] == 0 and r["p"] == 1.0  # cùng mô hình, ghép seed: hiệu luôn 0
    tot = [_dem(n, rng.integers(80, 100, n), rng.integers(0, 5, n), rng.integers(0, 5, n))]  # mô hình tất định 1 lần chạy
    r = tong_hop.bootstrap_cap_phan_cap(tot, a, B=500, seed=0)
    assert r["hieu"] > 0 and r["thap"] > 0 and r["p"] < 0.01


def test_khoi_o():
    assert tong_hop.khoi_o("hong_thai_binh_116_496", 1) == "hong_thai_binh_116_496"
    assert tong_hop.khoi_o("hong_thai_binh_116_496", 5) == tong_hop.khoi_o("hong_thai_binh_119_499", 5)
    assert tong_hop.khoi_o("hong_thai_binh_116_496", 5) != tong_hop.khoi_o("hong_thai_binh_121_496", 5)
    assert tong_hop.khoi_o("ca_104_472", 2) != tong_hop.khoi_o("gianh_104_472", 2)
