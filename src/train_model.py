from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "gesture_samples.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "gesture_classifier.joblib"

EXPECTED_FEATURE_COUNT = 63
TEST_SIZE = 0.20
RANDOM_STATE = 42


def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Run collect_data.py first."
        )

    dataset = pd.read_csv(DATA_PATH)

    if "label" not in dataset.columns:
        raise ValueError("Dataset is missing the 'label' column.")

    feature_columns = [
        column for column in dataset.columns if column.startswith("feature_")
    ]

    if len(feature_columns) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} feature columns, "
            f"but found {len(feature_columns)}."
        )

    if dataset[["label", *feature_columns]].isnull().any().any():
        raise ValueError("Dataset contains missing labels or feature values.")

    return dataset, feature_columns


def main():
    dataset, feature_columns = load_dataset()

    # X contains the input measurements; y contains the answers to learn.
    X = dataset[feature_columns]
    y = dataset["label"]

    print(f"Loaded {len(dataset)} samples.")
    print("\nSamples per gesture:")
    print(y.value_counts().sort_index().to_string())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    classifier = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nTest accuracy: {accuracy:.2%}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions))

    labels = sorted(y.unique())
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    matrix_table = pd.DataFrame(
        matrix,
        index=[f"actual_{label}" for label in labels],
        columns=[f"predicted_{label}" for label in labels],
    )

    print("Confusion matrix:")
    print(matrix_table.to_string())

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model_bundle = {
        "classifier": classifier,
        "feature_columns": feature_columns,
    }
    joblib.dump(model_bundle, MODEL_PATH)

    print(f"\nSaved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
