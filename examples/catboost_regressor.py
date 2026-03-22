"""Minimal one-line CatBoost regressor deployment example."""

from catboost import CatBoostRegressor
from sklearn.datasets import load_diabetes

from deeploi import deploy

X, y = load_diabetes(return_X_y=True, as_frame=True)

model = CatBoostRegressor(iterations=80, random_seed=42, verbose=False)
model.fit(X, y)

# One-line deployment
deploy(model)
