import time

import cv2
import mediapipe as mp
from feature_extraction import extract_features

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

def main():
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path="models/hand_landmarker.task"
        ),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with HandLandmarker.create_from_options(options) as landmarker:

        start_time = time.perf_counter()
        last_timestamp_ms = -1

        cv2.namedWindow("Hand Landmarker Test", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Hand Landmarker Test", 960, 540)

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Could not read frame.")
                break

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp_ms = int(
                (time.perf_counter() - start_time) * 1000
            )

            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1

            last_timestamp_ms = timestamp_ms

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            if result.hand_landmarks:
                for hand_landmarks in result.hand_landmarks:
                    draw_hand_landmarks(
                        frame,
                        hand_landmarks
                    )

                    features = extract_features(
                        hand_landmarks
                    )

                    if features is not None:
                        print(features.shape)

            cv2.imshow(
                "Hand Landmarker Test",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

def draw_hand_landmarks(frame, hand_landmarks):
    height, width, _ = frame.shape

    points = []

    for landmark in hand_landmarks:
        pixel_x = int(landmark.x * width)
        pixel_y = int(landmark.y * height)

        points.append((pixel_x, pixel_y))

    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start_idx],
            points[end_idx],
            (255, 255, 255),
            2
        )

    for pixel_x, pixel_y in points:
        cv2.circle(
            frame,
            (pixel_x, pixel_y),
            5,
            (0, 255, 0),
            -1
        )

if __name__ == "__main__":
    main()