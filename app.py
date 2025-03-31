import tensorflow as tf  # TensorFlow 버전 확인용
from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
import os  # 파일 경로 확인용

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Check TensorFlow version
required_version = (2, 4)
current_version = tuple(map(int, tf.__version__.split('.')[:2]))
if current_version < required_version:
    print(f"Error: TensorFlow version must be >= {required_version[0]}.{required_version[1]}. "
          f"Current version: {tf.__version__}")
    print("Please upgrade TensorFlow using: pip install --upgrade tensorflow")
    exit()

# Load the model
model_path = "model/keras_Model.h5"
if not os.path.exists(model_path):
    print(f"Error: Model file not found at '{model_path}'. Please check the file path.")
    exit()

model = load_model(model_path, compile=False)

# Load the labels
labels_path = "model/labels.txt"
if not os.path.exists(labels_path):
    print(f"Error: Labels file not found at '{labels_path}'. Please check the file path.")
    exit()

class_names = open(labels_path, "r").readlines()


camera = cv2.VideoCapture(0)  # 0은 기본 웹캠

if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()

    # 이미지가 제대로 읽어졌는지 확인
    if not ret:
        print("Failed to grab image")
        break

    # Resize the raw image into (224-height, 224-width) pixels
    image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Make the image a numpy array and reshape it to the models input shape.
    image_input = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image_input = (image_input / 127.5) - 1

    # Predict the model
    prediction = model.predict(image_input)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score in the console
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # Display prediction and confidence score on the image
    text = f"Class: {class_name[2:-1]}, Conf: {np.round(confidence_score * 100)}%"
    
    # Make sure the text is readable (yellow text)
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    # Show the image in a window (This is done only once now)
    cv2.imshow("Webcam Image", image)

    # Listen to the keyboard for presses
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the ESC key on your keyboard
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
# 원격지의 ipcamera도 이용 가능

# CAMERA can be 0 or 1 based on default camera of your computer
# camera = cv2.VideoCapture(0)

# # Check if the camera opened successfully
# if not camera.isOpened():
#     print("Error: Could not access the camera.")
#     exit()

# while True:
#     # Grab the webcamera's image.
#     ret, image = camera.read()

#     # Check if the frame was successfully captured
#     if not ret:
#         print("Error: Failed to capture image from camera.")
#         break

#     # Resize the raw image into (224-height,224-width) pixels
#     image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

#     # Show the image in a window
#     cv2.imshow("Webcam Image", image)

#     # Make the image a numpy array and reshape it to the models input shape.
#     image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

#     # Normalize the image array
#     image = (image / 127.5) - 1

#     # Predicts the model
#     prediction = model.predict(image)
#     index = np.argmax(prediction)
#     class_name = class_names[index]
#     confidence_score = prediction[0][index]

#     # Print prediction and confidence score
#     print("Class:", class_name[2:], end="")
#     print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

#     # Listen to the keyboard for presses.
#     keyboard_input = cv2.waitKey(1)

#     # 27 is the ASCII for the esc key on your keyboard.
#     if keyboard_input == 27:
#         break

# camera.release()
# cv2.destroyAllWindows()
