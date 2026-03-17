"""
Example: scikit-learn regression with Deeploi.

Shows package deployment for a regression task.
"""

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from deeploi import package, load
import os


def main():
    """Run the diabetes regression example."""
    
    # Load data
    print("Loading diabetes dataset...")
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    print("Training RandomForestRegressor...")
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Test R² score: {score:.3f}")
    
    # Create package
    print("\nCreating Deeploi package...")
    pkg = package(model, X_train)
    print(f"Package: {pkg}")
    
    # Make predictions
    print("\nMaking predictions on test set...")
    response = pkg.predict(X_test[:5])
    print(f"Predictions (first 5): {response.predictions}")
    
    # Save
    artifact_dir = "artifacts/diabetes_rf"
    print(f"\nSaving to {artifact_dir}...")
    pkg.save(artifact_dir)
    print("Saved!")
    
    # Load
    print(f"Loading from {artifact_dir}...")
    loaded_pkg = load(artifact_dir)
    print(f"Loaded: {loaded_pkg}")
    
    # Predict with loaded model
    print("Making predictions with loaded model...")
    response = loaded_pkg.predict(X_test[:5])
    print(f"Predictions: {response.predictions}")
    
    print("\nTo serve this model, run:")
    print(f"  pkg.serve(port=8001)")


if __name__ == "__main__":
    main()
