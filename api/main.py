import sys
from pathlib import Path




ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

sys.path.insert(0, str(SRC_DIR))

CONFIG_DIR = ROOT_DIR / "config"
sys.path.insert(0, str(CONFIG_DIR))

UPLOAD_DIR = ROOT_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(UPLOAD_DIR))

from eurosat_classifier.config import load_config
import os

from tensorflow import keras
import json
from fastapi import FastAPI,HTTPException, UploadFile
from predict import load_and_preprocess_image, get_top_k_predictions,predict_image
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

CONFIG_PATH = "config/mobilenetv3_smoke_test.yaml"

config = load_config(CONFIG_PATH)

# Chargement du config
model_path = ROOT_DIR/config["artifacts"]["ft_best_model_name"]

#Chargement du modèle
loaded_model = keras.models.load_model(model_path)

# chargement class_names.json
class_name_path=ROOT_DIR/config["artifacts"]["class_names"]
with open(class_name_path, "r") as f:
    class_names = json.load(f)


app = FastAPI()


@app.get("/health")
async def health():
    return {"status": "ok"}



@app.get("/predict-demo")
async def predict_demo():
    if class_names:
        demo_image_path = ROOT_DIR/config["data"]["image_test"]
        image_loaded = load_and_preprocess_image(demo_image_path, config["data"]["image_size"], preprocess_input)
        predictions = predict_image(loaded_model, image_loaded)
        k_predictions = get_top_k_predictions(predictions, class_names)
        print(k_predictions)
        return k_predictions
    else:
         raise HTTPException(
            status_code=503,
            detail="ClassName not found"
        )

@app.post("/predict")
async def predict(file : UploadFile):
    file_location = UPLOAD_DIR/file.filename

    # Save uploaded file locally
    with open(file_location, "wb+") as file_object:
        file_object.write(await file.read())

    # load_and process
    image_loaded = load_and_preprocess_image(file_location,config["data"]["image_size"], preprocess_input)

    predictions = predict_image(loaded_model, image_loaded)
    k_predictions = get_top_k_predictions(predictions, class_names)
    try:
        os.remove(file_location)
    except OSError as e:
        raise HTTPException(
            status_code=500,
            detail=f"File {file.filename} not remove"
        )

    return k_predictions