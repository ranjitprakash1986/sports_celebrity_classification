# imports
import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt
import pywt
import os
import shutil
from wavelet import w2d
import json
import joblib
import base64


def load_artifacts():
    """Loads the ML save model"""
    print("Loading the artifacts")
    global __class_name_to_number
    global __class_number_to_name
    global __model

    __model = None

    # Load the json file
    with open("./artifacts/celeb_code.json", "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v: k for k, v in __class_name_to_number.items()}
        print("Loaded dictionary of classes")

    # Load the saved classification model
    if __model is None:
        with open("./artifacts/celebrity_clf_logreg.pkl", "rb") as f:
            __model = joblib.load(f)
        print("Model loaded")


# Convert the new image to a b64 file type
def test_get_b64_image():
    """Convert the new image to a b64 file type"""
    with open("b64.txt") as f:
        return f.read()


# Convert the image from b64 to cv2 form
def get_cv2_from_b64(b64str):
    """
    credit: https://stackoverflow.com/questions/33754935/read-a-base-64-encoded-image-from-memory-using-opencv-python-library
    :param uri:
    :return:
    """
    encoded_data = b64str.split(",")[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    print("In cv2_fromb64:", img.shape)
    return img


# Classify the new image
def classify_image(img_b64, file_path=None):
    """Classify the new image"""
    imgs = get_cropped_image(img_b64, file_path)
    if imgs is None:
        print("No face detected")
        return
    print("This is imgs shape before the scaling", imgs.shape)

    result = []

    scale_size = 64
    scaled_img_raw = cv2.resize(imgs, (scale_size, scale_size))
    print("shape of scaled_img_raw after resizing: ", scaled_img_raw.shape)

    # wavelet transform
    # print(img.shape)
    im_har = w2d(imgs, "db1", 5)
    scaled_img_har = cv2.resize(im_har, (scale_size, scale_size))
    print("shape of scaled_img_har after resizing: ", scaled_img_har.shape)
    # Create a combined numpy vector of the raw and har image as feature
    stacked_image = np.vstack(
        (
            scaled_img_raw.reshape(scale_size * scale_size * 3, 1),
            scaled_img_har.reshape(scale_size * scale_size, 1),
        )
    )
    print("shape of stacked image before reshaping is: ", stacked_image.shape)
    # Final featurized image vetor
    final = stacked_image.reshape(1, len(stacked_image)).astype(float)

    result.append(
        {
            "class": __class_number_to_name[__model.predict(final)[0]],
            "classification_probability": np.round(__model.predict_proba(final).squeeze(), 2),
            "class_dictionary": __class_name_to_number,
        }
    )

    return result


# Convert the class code to celebrity name using the dcitionary
def get_celeb_name(code):
    return __class_number_to_name(code)


# Detect the face and two eyes and crop the image
def get_cropped_image(img_b64, image_path=None):
    """Detect the face and two eyes and crop the image"""

    # declare the cascade models
    face_cascade = cv2.CascadeClassifier(
        "./opencv/haarcascades/haarcascade_frontalface_default.xml"
    )
    eye_cascade = cv2.CascadeClassifier("./opencv/haarcascades/haarcascade_eye.xml")

    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_from_b64(img_b64)

    # detect the face coordinates from gray image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)

    if faces is None:
        return None

    for (y, x, w, h) in faces:
        # mark face zone with rectangle
        face_img = cv2.rectangle(img, (y, x), (y + w, x + h), (255, 0, 0), 2)
        face_gray = cv2.rectangle(gray, (y, x), (y + w, x + h), (255, 0, 0), 2)
        # plt.imshow(face_img)

        # crop to rectangular zone
        roi_color = face_img[x : x + h, y : y + w]
        roi_gray = face_gray[x : x + h, y : y + w]

        # plt.imshow(roi_color)
        # plt.imshow(roi_gray)

        # detect eyes in roi grayscale image
        eyes_gray = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes_gray) >= 2:
            print("This is returned from cropped_image func:", roi_color.shape)
            return roi_color
        else:
            return None


if __name__ == "__main__":
    load_artifacts()
    # Testing on a sample image as b64 input
    # output = classify_image(test_get_b64_image())

    # Testing by passing a regular image from a file path
    output = classify_image(None, file_path="./Test/ronaldo7.jpg")
    print(output)
