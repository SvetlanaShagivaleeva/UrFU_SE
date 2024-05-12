import io

import streamlit as st
from PIL import Image
import torch
import numpy as np
import cv2


int_to_classes = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bs",
    7: "train",
    8: "trck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    12: "stop sign",
    13: "parking meter",
    14: "bench",
    15: "bird",
    16: "cat",
    17: "dog",
    18: "horse",
    19: "sheep",
    20: "cow",
    21: "elephant",
    22: "bear",
    23: "zebra",
    24: "giraffe",
    25: "backpack",
    26: "mbrella",
    27: "handbag",
    28: "tie",
    29: "sitcase",
    30: "frisbee",
    31: "skis",
    32: "snowboard",
    33: "sports ball",
    34: "kite",
    35: "baseball bat",
    36: "baseball glove",
    37: "skateboard",
    38: "srfboard",
    39: "tennis racket",
    40: "bottle",
    41: "wine glass",
    42: "cp",
    43: "fork",
    44: "knife",
    45: "spoon",
    46: "bowl",
    47: "banana",
    48: "apple",
    49: "sandwich",
    50: "orange",
    51: "broccoli",
    52: "carrot",
    53: "hot dog",
    54: "pizza",
    55: "dont",
    56: "cake",
    57: "chair",
    58: "coch",
    59: "potted plant",
    60: "bed",
    61: "dining table",
    62: "toilet",
    63: "tv",
    64: "laptop",
    65: "mose",
    66: "remote",
    67: "keyboard",
    68: "cell phone",
    69: "microwave",
    70: "oven",
    71: "toaster",
    72: "sink",
    73: "refrigerator",
    74: "book",
    75: "clock",
    76: "vase",
    77: "scissors",
    78: "teddy bear",
    79: "hair drier",
    80: "toothbrsh",
}


def load_model():
    model = model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    return model


def load_image():
    uploaded_file = st.file_uploader(label="Выберите изображения для детекции")
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None


def show_prediction(img, preds):
    img = np.array(img)
    for data in preds.xyxy[0]:
        x0, y0, xk, yk, score, id_class = data
        x0, y0, xk, yk, id_class = int(x0), int(y0), int(xk), int(yk), int(id_class)
        img = cv2.rectangle(img, (x0, y0), (xk, yk), (255, 0, 0), 2)
        img = cv2.putText(
            img,
            f"{int_to_classes[id_class + 1]}",
            (x0 + 90, y0 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
        img = cv2.putText(
            img,
            f"{score:.2}:",
            (x0, y0 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    st.image(img)


if __name__ == "__main__":
    model = load_model()

    st.title("Детекция объектов yolov5")
    img = load_image()
    result = st.button("Задетектить изображение")
    if result:
        preds = model(img.copy())
        st.write("**Результаты:**")
        show_prediction(img, preds)
