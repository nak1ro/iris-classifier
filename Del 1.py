"""
Delivery 1 - Train & Save the Model
Multi-class classifier on Iris dataset using scikit-learn
"""

import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler


def main():
    iris = load_iris()
    X, y = iris.data, iris.target

    print(f"\nDataset shape: {X.shape}")
    print(f"\nClasses: {iris.target_names}")
    print(f"\nFeatures: {iris.feature_names}")

    print("\nSplitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"\nTest set: {X_test.shape[0]} samples")

    print("\nScaling features")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }

    best_accuracy = 0
    best_model = None
    best_name = ""

    print("Training classifiers")
    for name, classifier in classifiers.items():
        print(f"\nTraining {name}")

        classifier.fit(X_train_scaled, y_train)

        y_pred = classifier.predict(X_test_scaled)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = classifier
            best_name = name

    print(f"\nBest model: {best_name} with accuracy: {best_accuracy:.4f}")

    print(f"\nDetailed evaluation of {best_name}:")
    y_pred_best = best_model.predict(X_test_scaled)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_best, target_names=iris.target_names))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_best))

    print(f"\nSaving {best_name} model as model.joblib...")
    model_data = {
        'model': best_model,
        'scaler': scaler,
        'feature_names': iris.feature_names,
        'target_names': iris.target_names,
        'accuracy': best_accuracy
    }

    joblib.dump(model_data, 'model.joblib')
    print("\nModel saved successfully!")

    print(f"\nFinal Accuracy Score: {best_accuracy:.4f}")


if __name__ == "__main__":
    main()
