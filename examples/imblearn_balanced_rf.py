"""Minimal one-line imbalanced-learn meta-estimator deployment example."""

from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.datasets import load_iris

from deeploi import deploy

iris = load_iris(as_frame=True)
X, y = iris.data, iris.target

model = BalancedRandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X, y)

# One-line deployment
deploy(model)
