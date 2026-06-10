from model import build_mobilenetv3_model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from preprocessing import prepare_dataset,build_data_augmentation
from data import load_dataset
from eurosat_classifier.config import load_config



def build_callbacks(early_stop_patience, reduce_patience,min_lr, best_model_name):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=early_stop_patience,  # Patience raisonnable
        restore_best_weights=True,
        mode='max',
        verbose=1
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=reduce_patience,
        min_lr=min_lr,
        verbose=1
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        best_model_name,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=0
    )
    return early_stopping, reduce_lr, checkpoint



def compile_model(model, learning_rate):
    model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),  
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_accuracy'), 'precision', 'recall'])

    

def train_feature_extraction():

    config = load_config("configs/mobilenetv3_config.yaml")
    batch_size = config["data"]["batch_size"]
    input_shape = config["dataset"]["input_shape"]
    train_ds, val_ds, test_ds, class_names= load_dataset(dataset_name=config["dataset"]["name"], image_size=config["data"]["image_size"], batch_size=batch_size, seed=config["dataset"]["seed"], sub_dir_path=config["dataset"]["sub_dir"])
    data_augmentation= build_data_augmentation()
    train_ds_prepared = prepare_dataset(train_ds,data_augmentation, augment=True, shuffle=True )
    val_ds_prepared = prepare_dataset(val_ds, data_augmentation, augment=False, shuffle=False )
    test_ds_prepared = prepare_dataset(test_ds, data_augmentation, augment=False, shuffle=False )

    model, base = build_mobilenetv3_model(input_shape, len(class_names))
    base.trainable = False


    compile_model(model, config["training"]["learning_rate_frozen"])
    early_stopping, reduce_lr, checkpoint = build_callbacks(config["training"]["fe_early_stop_patience"], config["training"]["fe_reduce_patience"], config["training"]["fe_min_lr"], config["artifacts"]["fe_best_model_name"])

    history = model.fit(
        train_ds_prepared,
        validation_data=val_ds_prepared,
        epochs=config["training"]["epochs_frozen"],
        callbacks=[early_stopping, reduce_lr, checkpoint],
        verbose=1
    )
    return model, base, history, train_ds_prepared, val_ds_prepared, test_ds_prepared




def train_fine_tuning(model, base, train_ds_prepared, val_ds_prepared, test_ds_prepared):

    base.trainable = True
    
    config = load_config("configs/mobilenetv3_config.yaml")
    fine_tune_at = len(base.layers) - config["fine_tuning"]["fine_tune_last_n_layers"]

    for i, layer in enumerate(base.layers):
        if i < fine_tune_at:
            layer.trainable = False
        else:
            layer.trainable = True
    
    compile_model(model, config["training"]["learning_rate_finetune"])

    early_stopping_ft, reduce_lr_ft, checkpoint_ft = build_callbacks(config["training"]["ft_early_stop_patience"], config["training"]["ft_reduce_patience"], config["training"]["ft_min_lr"], config["artifacts"]["ft_best_model_name"])
    history_finetuned = model.fit(
        train_ds_prepared,
        validation_data=val_ds_prepared,
        epochs=config["training"]["epochs_finetune"],
        callbacks=[early_stopping_ft, reduce_lr_ft, checkpoint_ft],
        verbose=1
    )

    return model, history_finetuned, test_ds_prepared

