import tensorflow as tf
import numpy as np

def load_and_preprocess_image(image_path, image_size, preprocess_input):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=image_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Ajouter dimension batch
    img_array = preprocess_input(img_array)  # Normalisation MobileNetV3
    return img_array


def predict_image(model, img_array):
    predictions = model.predict(img_array, verbose=0)
    return predictions

def get_top_k_predictions(predictions, class_names, k=3):
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    
    # Top k prédictions
    topk_idx = np.argsort(predictions[0])[-k:][::-1]
    return {
        'predicted_class': class_names[predicted_class_idx],
        'confidence': float(confidence),
        'all_probabilities': predictions[0],
        'topk': [(class_names[i], float(predictions[0][i])) for i in topk_idx]
    }