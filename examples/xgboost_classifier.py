"""
Example: XGBoost classification with Deeploi.

Shows deployment of an XGBoost classifier model.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import xgboost as xgb
from deeploi import package, load
import os


def main():
    """Run the breast cancer classification example with XGBoost."""
    
    # Load data
    print("Loading breast cancer dataset...")
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    print("Training XGBClassifier...")
    model = xgb.XGBClassifier(
        n_estimators=10,
        max_depth=5,
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Test accuracy: {score:.3f}")
    
    # Create package
    print("\nCreating Deeploi package...")
    pkg = package(model, X_train)
    print(f"Package: {pkg}")
    
    # Make predictions with probabilities
    print("\nMaking predictions on test set...")
    response = pkg.predict(X_test[:3])
    print(f"Predictions (first 3): {response.predictions}")
    
    print("\nMaking predictions with probabilities...")
    response_with_proba = pkg.predict(X_test[:3], include_probabilities=True)
    print(f"Predictions: {response_with_proba.predictions}")
    print(f"Probabilities: {response_with_proba.probabilities}")
    
    # Save
    artifact_dir = "artifacts/cancer_xgb"
    print(f"\nSaving to {artifact_dir}...")
    pkg.save(artifact_dir)
    print("Saved!")
    
    # List files
    if os.path.exists(artifact_dir):
        print(f"Artifact contents: {os.listdir(artifact_dir)}")
    
    # Load
    print(f"\nLoading from {artifact_dir}...")
    loaded_pkg = load(artifact_dir)
    print(f"Loaded: {loaded_pkg}")
    
    # Predict with loaded model
    print("Making predictions with loaded model...")
    response = loaded_pkg.predict(X_test[:3], include_probabilities=True)
    print(f"Predictions: {response.predictions}")
    print(f"Probabilities: {response.probabilities}")
    
    print("\nTo serve this model, run:")
    print(f"  loaded_pkg.serve(port=8002)")


if __name__ == "__main__":
    main()
