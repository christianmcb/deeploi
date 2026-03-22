"""Minimal one-line CatBoost classifier deployment example."""

from catboost import CatBoostClassifier
from sklearn.datasets import load_iris

from deeploi import deploy

iris = load_iris(as_frame=True)
X, y = iris.data, iris.target

model = CatBoostClassifier(iterations=50, random_seed=42, verbose=False)
model.fit(X, y)

# One-line deployment
deploy(model)
