"""Minimal one-line NGBoost regressor deployment example."""

from ngboost import NGBRegressor
from sklearn.datasets import load_diabetes

from deeploi import deploy

X, y = load_diabetes(return_X_y=True, as_frame=True)

model = NGBRegressor(verbose=False)
model.fit(X, y)

# One-line deployment
deploy(model)
