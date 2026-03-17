"""
Example: scikit-learn classification with Deeploi.

Shows three usage patterns:
1. One-liner: deploy()
2. Package + save/load: package()
3. API serving and prediction
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from deeploi import deploy, package, load
import os


def main():
    """Run the Iris classification example."""
    
    # Load data
    print("Loading iris dataset...")
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Test accuracy: {score:.3f}")
    
    # Example 1: One-liner deploy
    print("\n" + "="*60)
    print("EXAMPLE 1: One-liner deployment")
    print("="*60)
    print("deploy(model, sample=X_train)")
    print("\nThis would start a server at http://127.0.0.1:8000")
    print("(Commented out to avoid blocking. Uncomment to try:)")
    # deploy(model, sample=X_train, port=8000)
    
    # Example 2: Package, save, load
    print("\n" + "="*60)
    print("EXAMPLE 2: Package, save, and load")
    print("="*60)
    
    print("Creating package...")
    pkg = package(model, X_train)
    print(f"Package: {pkg}")
    
    print("Making predictions...")
    response = pkg.predict(X_test[:5])
    print(f"Predictions: {response.predictions}")
    
    print("\nMaking predictions with probabilities...")
    response_with_proba = pkg.predict(X_test[:5], include_probabilities=True)
    print(f"Predictions: {response_with_proba.predictions}")
    print(f"Probabilities: {response_with_proba.probabilities}")
    
    # Save
    artifact_dir = "artifacts/iris_rf"
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
    response = loaded_pkg.predict(X_test[:5], include_probabilities=True)
    print(f"Predictions: {response.predictions}")
    print(f"Probabilities: {response.probabilities}")
    
    # Example 3: Serving
    print("\n" + "="*60)
    print("EXAMPLE 3: Creating FastAPI app (not started)")
    print("="*60)
    print("pkg.serve(port=8000)")
    print("\nTo test locally, run:")
    print('  curl -X POST http://127.0.0.1:8000/predict \\')
    print('    -H "Content-Type: application/json" \\')
    print("    -d '{\"records\": [{\"sepal length (cm)\": 5.1, " +
          "\"sepal width (cm)\": 3.5, \"petal length (cm)\": 1.4, " +
          "\"petal width (cm)\": 0.2}]}'")


if __name__ == "__main__":
    main()
