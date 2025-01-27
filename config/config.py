from pathlib import Path

CONFIG = {
    "BASE_DIR": Path(__file__).resolve().parent.parent,
    "DATA_PATH": Path(__file__).resolve().parent.parent / "data" / "data_house.csv",
    "TEST_SIZE": 0.2,
}