

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

sys.path.insert(0, str(SRC_DIR))

CONFIG_DIR = ROOT_DIR / "config"
sys.path.insert(0, str(CONFIG_DIR))

MODELS_DIR = ROOT_DIR / "models"
sys.path.insert(0, str(MODELS_DIR))

import tensorflow as tf
from tensorflow import keras
import json
from predict import load_and_preprocess_image, get_top_k_predictions,predict_image
from eurosat_classifier.config import load_config
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

CONFIG_PATH = "config/mobilenetv3_smoke_test.yaml"

print(sys.path)
config = load_config(CONFIG_PATH) #

model_path = config["artifacts"]["ft_best_model_name"]
loaded_model = keras.models.load_model(model_path)

class_name_path=config["artifacts"]["class_names"]
with open(class_name_path, "r") as f:
    class_names = json.load(f)

if class_names:
    demo_image_path = config["data"]["image_test"]
    image_loaded = load_and_preprocess_image(demo_image_path, config["data"]["image_size"], preprocess_input)
    predictions = predict_image(loaded_model, image_loaded)
    k_predictions = get_top_k_predictions(predictions, class_names)
    print(k_predictions)

else:
    raise RuntimeError("Class names not defined!")