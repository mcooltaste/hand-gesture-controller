import numpy as np

def extract_features(hand_landmarks):
    landmarks = np.array(
        [
            [landmark.x, landmark.y, landmark.z]
            for landmark in hand_landmarks
        ],
        dtype=np.float32
    )

    # Make wrist the origin.
    wrist = landmarks[0]
    landmarks = landmarks - wrist

    # Normalize for hand size / distance from camera.
    distances = np.linalg.norm(
        landmarks,
        axis=1
    )

    scale = np.max(distances)

    if scale == 0:
        return None

    landmarks = landmarks / scale

    # Convert 21x3 matrix into 63-element vector.
    features = landmarks.flatten()

    return features