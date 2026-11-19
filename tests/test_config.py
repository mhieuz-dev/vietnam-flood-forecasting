from floodlab import config


def test_hang_so_khop_spec():
    assert config.TILE_M == config.TILE_PX * config.PIXEL_M == 4480
    assert config.CRS == "EPSG:32648"
    assert config.LEADS == (1, 2, 3)
    assert config.SEEDS == (0, 1, 2)
    assert config.SO_NGAY_QUAN_TRAC == 31
    assert config.NHAN_BO_QUA == 255
    assert config.LUU_VUC_KIEM_DINH in config.LUU_VUC
    assert len(config.LUU_VUC) == 6
    assert config.CUA_SO == {
        "train": ("2016-01-01", "2023-12-31"),
        "test_a": ("2024-09-01", "2024-09-30"),
        "test_b": ("2025-10-01", "2025-11-30"),
    }
    assert config.KHUNG_MUA_THUONG_NGUON == (97.0, 8.0, 110.0, 26.0)
    assert config.API_K == 0.90


def test_root_ghi_de_bang_bien_moi_truong(monkeypatch, tmp_path):
    import importlib

    monkeypatch.setenv("FLOODLAB_ROOT", str(tmp_path))
    c = importlib.reload(config)
    try:
        assert c.ROOT == tmp_path and c.DATA_DIR == tmp_path / "data" and c.CAU_HINH_DIR == tmp_path / "cau_hinh"
    finally:
        monkeypatch.delenv("FLOODLAB_ROOT")
        importlib.reload(config)
