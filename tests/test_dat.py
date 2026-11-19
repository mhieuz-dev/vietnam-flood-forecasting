import numpy as np
import pandas as pd

from floodlab import dat
from floodlab.config import CAU_HINH_DIR


def test_nhom_thuy_van_cat_la_A():
    assert dat.nhom_thuy_van(np.array([5.0]), np.array([92.0]))[0] == "A"


def test_nhom_thuy_van_theo_ross():
    # (sét %, cát %) -> lớp kết cấu USDA -> nhóm theo Ross et al. 2018 Bảng 1
    ca = [((5, 80), "B"),    # thịt pha cát (loamy sand)
          ((15, 65), "B"),   # cát pha thịt (sandy loam)
          ((20, 40), "C"),   # thịt (loam)
          ((30, 30), "C"),   # sét pha thịt (clay loam)
          ((50, 20), "D"),   # sét (clay)
          ((45, 5), "D")]    # sét pha bụi (silty clay)
    for (set_, cat), mong in ca:
        assert dat.nhom_thuy_van(np.array([float(set_)]), np.array([float(cat)]))[0] == mong, (set_, cat)


def test_cn_tra_bang():
    bang = pd.read_csv(CAU_HINH_DIR / "cn_bang.csv")
    for _, r in bang.iterrows():
        for nhom in "ABCD":
            got = dat.cn(np.array([nhom]), np.array([r.worldcover]))[0]
            assert got == r[nhom], (r.worldcover, nhom)


def test_nhom_pedon_lay_tang_kem_thoat_nhat():
    # cát ở tầng mặt nhưng sét ở tầng sâu -> pedon nhóm D
    assert dat.nhom_pedon([np.array(["A"]), np.array(["D"]), np.array(["B"])])[0] == "D"


def test_lap_dat_thieu_bang_lang_gieng():
    a = np.array([[10.0, -3276.8], [-3276.8, 30.0]])
    out, so = dat.lap_thieu(a)
    assert so == 2 and not np.isnan(out).any() and out.min() >= 10.0
