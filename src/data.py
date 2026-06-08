import kagglehub
import os
import tensorflow as tf



def load_dataset(dataset_name, image_size, batch_size, seed, sub_dir_path):
    path = kagglehub.dataset_download(dataset_name) # "apollo2506/eurosat-dataset"

    data_dir = os.path.join(path, sub_dir_path) # "EuroSAT"
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=seed,  #42
        image_size=image_size,
        batch_size=batch_size
    )

    temp_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size
)



    temp_ds = temp_ds.shuffle(1000, seed=42)

    # Nombre de batchs
    temp_batches = tf.data.experimental.cardinality(temp_ds).numpy()

    # 50% VAL / 50% TEST
    val_ds = temp_ds.take(temp_batches // 2)
    test_ds = temp_ds.skip(temp_batches // 2)
    class_names = train_ds.class_names
    return train_ds, val_ds, test_ds, class_names
