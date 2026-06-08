import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



def build_data_augmentation():
    data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomFlip("vertical"),
    layers.RandomRotation(0.1),  # Rotation jusqu'à 30%
    layers.RandomZoom(0.15),
    ], name="data_augmentation")
    return data_augmentation


def  prepare_dataset(ds,  data_augmentation, preprocess_input, augment=False, shuffle=True ):
    # Normalisation MobileNetV2
    normalization = lambda x, y: (preprocess_input(x), y)
    ds = ds.map(normalization, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Augmentation (seulement pour le train)
    if augment:
        augmentation = lambda x, y: (data_augmentation(x, training=True), y)
        ds = ds.map(augmentation, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Shuffle pour le training
    if shuffle:
        ds = ds.shuffle(1000)
    
    # Optimisation des performances
    #ds = ds.cache()  # Cache en mémoire
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)  # Prefetch pour accélérer
    
    return ds

