

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

sys.path.insert(0, str(SRC_DIR))

CONFIG_DIR = ROOT_DIR / "config"
sys.path.insert(0, str(CONFIG_DIR))

from train import train_feature_extraction, train_fine_tuning
from evaluate import evaluate

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


CONFIG_PATH = "config/mobilenetv3_smoke_test.yaml"


model, base, history_fe, train_ds, val_ds, test_ds, class_names = train_feature_extraction(preprocess_input,
    config_path=CONFIG_PATH
)

model, history_ft, test_ds = train_fine_tuning(
    model,
    base,
    train_ds,
    val_ds,
    test_ds,
    config_path=CONFIG_PATH
)

results = evaluate(
    model,
    test_ds,
    class_names
)

print("Smoke test completed.")
print("Accuracy:", results["accuracy"])
print("Top-2 accuracy:", results["top2_accuracy"])
print(results["classification_report"])