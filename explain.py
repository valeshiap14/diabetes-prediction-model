"""Explain the fitted estimator on the same transformed features as inference.

SHAP versions and estimator output conventions vary. A failed explanation is
reported to the UI; it never changes the underlying model prediction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def explain_prediction(pipeline, row: pd.DataFrame) -> pd.DataFrame:
    import shap

    prep = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    transformed = prep.transform(row)
    names = list(prep.get_feature_names_out())
    explainer = shap.TreeExplainer(model)
    explanation = explainer(transformed)
    vals = np.asarray(explanation.values)
    # Binary models sometimes return a (rows, features, classes) tensor.
    if vals.ndim == 3:
        vals = vals[:, :, 1]
    if vals.ndim != 2 or vals.shape[0] != 1 or vals.shape[1] != len(names):
        raise ValueError("Unexpected SHAP output dimensionality")
    return pd.DataFrame({"Feature": names, "Contribution": vals[0]}).sort_values(
        "Contribution", key=lambda x: x.abs(), ascending=False
    )
