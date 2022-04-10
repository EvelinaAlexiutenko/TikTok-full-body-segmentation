import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import metrics
from tensorflow.python.keras.metrics import MeanIoU


def compute_iou(model, a, b, n_classes=2):
    IoU_values = []
    for img in range(0, a.shape[0]):
        temp_img = a[img]
        ground_truth = b[img]
        temp_img_input = np.expand_dims(temp_img, 0)
        prediction = (model.predict(temp_img_input)[
                      0, :, :, 0] > 0.5).astype(np.uint8)

        IoU = MeanIoU(num_classes=n_classes)
        IoU.update_state(ground_truth[:, :, 0], prediction)
        IoU = IoU.result().numpy()
        IoU_values.append(IoU)
    return IoU_values
