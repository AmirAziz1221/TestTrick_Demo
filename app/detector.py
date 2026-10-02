"""MediaPipe Face Landmarker wrapper. The model is loaded ONCE at server start."""
from __future__ import annotations  # Enables postponed evaluation of type annotations

import threading  # Used for running tasks in separate threads
import urllib.request  # Used for downloading files/data from URLs
from dataclasses import dataclass  # Used to create simple data-holding classes

import cv2  # OpenCV: used for image/video processing and computer vision
import numpy as np  # NumPy: used for numerical operations and working with image arrays
import mediapipe as mp  # MediaPipe: used for computer vision and pose/hand/face tracking
from mediapipe.tasks import python as mp_python  # MediaPipe Tasks: used to configure and run MediaPipe models
from mediapipe.tasks.python import vision  # MediaPipe Vision: provides vision tasks such as pose, object, and image detection

from .config import MODEL_PATH, MODEL_URL  # Imports the model file path and model download URL from the project config


@dataclass  # Dataclass: creates a simple class mainly used to store structured data
class FrameData:  # Stores the data/results obtained from one video frame
    faces: list  # Stores face landmark data; each face contains a (478, 3) array of normalized coordinates
    blends: list  # Stores blendshape results; each face has a dictionary of scores such as eye-look, blink, smile, etc.
    brightness: float  # Stores the brightness value of the current frame as a floating-point number

def ensure_model():
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading face_landmarker.task -> {MODEL_PATH}")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


class FaceService:
    def __init__(self, num_faces: int = 2):
        opts = vision.FaceLandmarkerOptions(  # MediaPipe Vision: configures the Face Landmarker model
            base_options=mp_python.BaseOptions(model_asset_path=str(ensure_model())),  # MediaPipe Tasks: provides the model file path; ensure_model() downloads/checks the model
            running_mode=vision.RunningMode.IMAGE,  # MediaPipe Vision: processes each image independently; tracking is not useful because frames are ~700 ms apart
            num_faces=num_faces,  # MediaPipe Vision: sets the maximum number of faces to detect; 2 allows detection of a second person
            output_face_blendshapes=True,  # MediaPipe Vision: enables facial expression/blendshape scores such as blinking and eye movement
            min_face_detection_confidence=0.5,  # MediaPipe Vision: minimum confidence required to detect a face
            min_face_presence_confidence=0.5,  # MediaPipe Vision: minimum confidence required to consider the detected face present
        )

        self._lm = vision.FaceLandmarker.create_from_options(opts)  # MediaPipe Vision: creates the Face Landmarker using the configured options
        self._lock = threading.Lock()  # Python threading: prevents multiple threads from using the model at the same time

    def analyze(self, jpeg: bytes) -> FrameData | None:
        bgr = cv2.imdecode(  # OpenCV: decodes JPEG bytes into an image array
            np.frombuffer(jpeg, np.uint8),  # NumPy: converts JPEG bytes into a uint8 array that OpenCV can read
            cv2.IMREAD_COLOR  # OpenCV: tells imdecode() to load the image as a color image
        )

        if bgr is None:  # Python: checks whether OpenCV successfully decoded the image
            return None  # Returns nothing if the JPEG/image could not be decoded

        brightness = float(  # Python: converts the calculated brightness value into a float
            cv2.mean(  # OpenCV: calculates the average pixel value
                cv2.cvtColor(  # OpenCV: converts the image from one color format to another
                    cv2.resize(bgr, (64, 48)),  # OpenCV: resizes the image to 64x48 for faster brightness calculation
                    cv2.COLOR_BGR2GRAY  # OpenCV: converts the BGR image into grayscale
                )
            )[0]  # Gets the grayscale mean value representing the average brightness
        )

        img = mp.Image(  # MediaPipe: creates an image object that MediaPipe can process
            image_format=mp.ImageFormat.SRGB,  # MediaPipe: specifies that the image uses the SRGB color format
            data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # OpenCV: converts BGR format from OpenCV to RGB format required by MediaPipe
        )

        with self._lock:  # Python threading: safely accesses the Face Landmarker model
            res = self._lm.detect(img)  # MediaPipe Face Landmarker: detects faces and facial landmarks in the image

        faces = [  # Python list comprehension: creates a list containing landmark data for every detected face
            np.array(  # NumPy: converts each face's landmark coordinates into a NumPy array
                [(p.x, p.y, p.z) for p in f],  # Python: extracts x, y, and z coordinates from every facial landmark
                dtype=np.float32  # NumPy: stores coordinates as 32-bit floating-point values
            )
            for f in res.face_landmarks  # MediaPipe: loops through all detected faces and their landmarks
        ]

        blends = [  # Python list comprehension: creates a list of blendshape dictionaries for each detected face
            {c.category_name: c.score for c in b}  # Python: creates a dictionary containing blendshape name and confidence score
            for b in (res.face_blendshapes or [])  # MediaPipe: loops through detected facial blendshapes; uses an empty list if none exist
        ]

        return FrameData(faces, blends, brightness)  # Returns all face landmarks, blendshapes, and brightness as a FrameData object


if __name__ == "__main__":  # Python: runs this block only when this file is executed directly
    ensure_model()  # Checks whether the MediaPipe model exists and downloads/prepares it if necessary
    print("Model ready.")  # Python: displays a confirmation message that the model is ready