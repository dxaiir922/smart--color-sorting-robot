import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "color_training.csv"
MODEL = ROOT / "data" / "color_model.json"
FEATURES = ["red", "green", "blue", "size", "is_box"]

groups = {}
with DATASET.open(newline="", encoding="utf-8") as file:
    for row in csv.DictReader(file):
        groups.setdefault(row["label"], []).append([float(row[name]) for name in FEATURES])

centroids = {}
for label, rows in groups.items():
    centroids[label] = [
        sum(row[i] for row in rows) / len(rows)
        for i in range(len(FEATURES))
    ]

MODEL.write_text(json.dumps({"features": FEATURES, "centroids": centroids}, indent=2), encoding="utf-8")
print("saved", MODEL)
