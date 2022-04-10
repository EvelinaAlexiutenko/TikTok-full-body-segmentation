import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping

from focal_loss import BinaryFocalLoss

from model import build_unet
from dice_metric import dice_metric
from data import load_dataset, tf_dataset
from main import CWD

if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(gpus[0], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")

    """ Hyperparamaters """
    dataset_path = os.path.join(CWD, "segmentation_full_body_tik_tok_2615_img")
    input_shape = (256, 256, 3)
    batch_size = 2
    epochs = 10
    lr = 1e-4
    model_path = "weights/best.h5"
    csv_path = "data.csv"

    """ Load the dataset """
    (train_x, train_y), (test_x, test_y) = load_dataset(dataset_path)
    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Test: {len(test_x)} - {len(test_y)}")

    train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
    test_dataset = tf_dataset(test_x, test_y, batch=batch_size)

    """ Model """
    model = build_unet(input_shape)
    model.compile(
        loss=BinaryFocalLoss(2),
        optimizer=tf.keras.optimizers.Adam(lr),
        metrics=[
            dice_metric,
            tf.keras.metrics.MeanIoU(num_classes=2),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.Precision()
        ]
    )

    # model.summary()

    callbacks = [
        ModelCheckpoint(model_path, monitor="val_loss", verbose=1),
        ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.1, verbose=1),
        CSVLogger(csv_path),
        EarlyStopping(monitor="val_loss", patience=10)
    ]

    train_steps = len(train_x) // batch_size
    if len(train_x) % batch_size != 0:
        train_steps += 1

    test_steps = len(test_x) // batch_size
    if len(test_x) % batch_size != 0:
        test_steps += 1

    model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=epochs,
        steps_per_epoch=train_steps,
        validation_steps=test_steps,
        callbacks=callbacks
    )
