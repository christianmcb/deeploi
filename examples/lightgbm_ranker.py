"""Minimal one-line LightGBM ranker deployment example."""

import lightgbm as lgb
from sklearn.datasets import load_diabetes

from deeploi import deploy

X, y = load_diabetes(return_X_y=True, as_frame=True)

model = lgb.LGBMRanker(n_estimators=50, random_state=42)
model.fit(X, y, group=[len(X)])

# One-line deployment
deploy(model)
