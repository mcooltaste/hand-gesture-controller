import time
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import pandas as pd

from feature_extraction import extract_features


PROJECT_ROOT = Path(__file__).resolve().parent.parent
HAND_LANDMARKER_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
CLASSIFIER_PATH = PROJECT_ROOT / "models" / "gesture_classifier.joblib"

CONFIDENCE_THRESHOLD = 0.60

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


def load_classifier():
    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(
            f"Classifier not found at {CLASSIFIER_PATH}. Run train_model.py first."
        )

    model_bundle = joblib.load(CLASSIFIER_PATH)

    if "classifier" not in model_bundle or "feature_columns" not in model_bundle:
        raise ValueError("The saved classifier file has an unexpected format.")

    return model_bundle["classifier"], model_bundle["feature_columns"]


def predict_gesture(classifier, feature_columns, features):
    # A DataFrame preserves the same feature names and order used for training.
    sample = pd.DataFrame([features], columns=feature_columns)
    probabilities = classifier.predict_proba(sample)[0]
    best_index = probabilities.argmax()
    confidence = float(probabilities[best_index])
    label = classifier.classes_[best_index]

    if confidence < CONFIDENCE_THRESHOLD:
        label = "unknown"

    return label, confidence


def draw_hand_landmarks(frame, hand_landmarks):
    height, width, _ = frame.shape
    points = [
        (int(landmark.x * width), int(landmark.y * height))
        for landmark in hand_landmarks
    ]

    for start_index, end_index in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start_index],
            points[end_index],
            (255, 255, 255),
            2,
        )

    for point in points:
        cv2.circle(frame, point, 5, (0, 255, 0), -1)


def draw_prediction(frame, label, confidence):
    cv2.rectangle(frame, (20, 20), (600, 130), (230, 230, 230), -1)
    cv2.putText(
        frame,
        f"Gesture: {label}",
        (35, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2,
    )
    cv2.putText(
        frame,
        f"Confidence: {confidence:.1%}",
        (35, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2,
    )


def main():
    classifier, feature_columns = load_classifier()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_LANDMARKER_PATH)),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    window_name = "Real-Time Gesture Recognition"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    try:
        with HandLandmarker.create_from_options(options) as landmarker:
            start_time = time.perf_counter()
            last_timestamp_ms = -1

            while True:
                ret, frame = cap.read()

                if not ret:
                    print("Error: Could not read frame.")
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )

                timestamp_ms = int(
                    (time.perf_counter() - start_time) * 1000
                )

                if timestamp_ms <= last_timestamp_ms:
                    timestamp_ms = last_timestamp_ms + 1

                last_timestamp_ms = timestamp_ms
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                label = "no hand"
                confidence = 0.0

                if result.hand_landmarks:
                    hand_landmarks = result.hand_landmarks[0]
                    features = extract_features(hand_landmarks)

                    if features is not None:
                        label, confidence = predict_gesture(
                            classifier,
                            feature_columns,
                            features,
                        )

                    draw_hand_landmarks(frame, hand_landmarks)

                draw_prediction(frame, label, confidence)
                cv2.imshow(window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                window_closed = cv2.getWindowProperty(
                    window_name,
                    cv2.WND_PROP_VISIBLE,
                ) < 1

                if key == ord("q") or window_closed:
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
