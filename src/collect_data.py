import csv
import time
from pathlib import Path

import cv2
import mediapipe as mp

from feature_extraction import extract_features


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "gesture_samples.csv"

GESTURES = (
    "open_palm",
    "closed_fist",
    "thumbs_up",
    "thumbs_down",
    "peace",
    "point_up",
)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def choose_gesture():
    print("Choose the gesture you want to collect:")

    for number, gesture in enumerate(GESTURES, start=1):
        print(f"  {number}: {gesture}")

    while True:
        choice = input("Gesture number: ").strip()

        if choice.isdigit():
            index = int(choice) - 1

            if 0 <= index < len(GESTURES):
                return GESTURES[index]

        print("Please enter one of the listed numbers.")


def save_sample(label, features):
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_is_new = not DATA_PATH.exists() or DATA_PATH.stat().st_size == 0

    with DATA_PATH.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)

        if file_is_new:
            header = ["label"] + [f"feature_{i}" for i in range(63)]
            writer.writerow(header)

        writer.writerow([label, *features.tolist()])


def count_samples(label):
    if not DATA_PATH.exists():
        return 0

    with DATA_PATH.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        return sum(row["label"] == label for row in reader)


def draw_status(frame, label, sample_count, hand_detected):
    detection_text = "Hand detected" if hand_detected else "No hand detected"
    detection_color = (0, 255, 0) if hand_detected else (0, 0, 255)

    cv2.putText(
        frame,
        f"Gesture: {label}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2,
    )
    cv2.putText(
        frame,
        f"Saved samples: {sample_count}",
        (30, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2,
    )
    cv2.putText(
        frame,
        detection_text,
        (30, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        detection_color,
        2,
    )
    cv2.putText(
        frame,
        "SPACE: save sample    Q: quit",
        (30, 185),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
    )


def main():
    gesture = choose_gesture()
    sample_count = count_samples(gesture)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    window_name = "Gesture Data Collector"
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

                features = None

                if result.hand_landmarks:
                    features = extract_features(result.hand_landmarks[0])

                draw_status(
                    frame,
                    gesture,
                    sample_count,
                    features is not None,
                )
                cv2.imshow(window_name, frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                if key == ord(" "):
                    if features is None:
                        print("No sample saved: no hand was detected.")
                    else:
                        save_sample(gesture, features)
                        sample_count += 1
                        print(f"Saved {gesture} sample {sample_count}.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
