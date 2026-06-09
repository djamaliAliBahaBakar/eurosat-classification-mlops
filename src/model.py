
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV3Small

def build_mobilenetv3_model(input_shape, num_classes):
    base_model = MobileNetV3Small(
        input_shape=input_shape,
        include_top=False,  # Exclure la couche de classification ImageNet
        weights='imagenet',  # Charger les poids pré-entraînés sur ImageNet
        pooling='avg'  # Global Average Pooling à la fin
    )

    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)

    # Classifier avec régularisation modérée
    x = layers.Dropout(0.3)(x)  # Dropout modéré
    x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs, name='MobileNetV3_classifier')
    return model, base_model

    

