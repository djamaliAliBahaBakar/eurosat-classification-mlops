from model import build_mobilenetv3_model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from preprocessing import prepare_dataset,build_data_augmentation
from data import load_dataset
from eurosat_classifier.config import load_config



def build_callbacks():
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,  # Patience raisonnable
        restore_best_weights=True,
        mode='max',
        verbose=1
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-7,
        verbose=1
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'best_model_frozen.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=0
    )
    return early_stopping, reduce_lr, checkpoint


def compile_model(model):
    model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # LR standard
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_accuracy')])

    

def train_feature_extraction(input_shape):
    EPOCHS_FROZEN = 40
    config = load_config("configs/mobilenetv3_config.yaml")
    batch_size = config["data"]["batch_size"]
    train_ds, val_ds, test_ds, class_names= load_dataset(dataset_name="apollo2506/eurosat-dataset", image_size=(224, 224), batch_size=batch_size, seed=42, sub_dir_path="EuroSAT")
    data_augmentation= build_data_augmentation()
    train_ds_prepared = prepare_dataset(train_ds,data_augmentation, augment=True, shuffle=True )
    val_ds_prepared = prepare_dataset(val_ds, data_augmentation, augment=False, shuffle=False )

    model, base = build_mobilenetv3_model(input_shape, len(class_names))
    base.trainable = False


    compile_model(model)
    early_stopping, reduce_lr, checkpoint = build_callbacks()

    history = model.fit(
        train_ds_prepared,
        validation_data=val_ds_prepared,
        epochs=EPOCHS_FROZEN,
        callbacks=[early_stopping, reduce_lr, checkpoint],
        verbose=1
    )
    return model, base, history, test_ds, class_names
