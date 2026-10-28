"""Phase 3: does the scale of the sample weights change the fitted logistic regression? (PHASE3_SETUP.md item 18)

Same synthetic data (numpy default_rng(0), 500 rows, 5 features), weights w and 10 w, StandardScaler(sample_weight)
followed by LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, sample_weight). Prints the scikit-learn
version, whether the scaler and the coefficients are identical, and the largest coefficient difference.

Usage:  python src/phase3/check_weight_scale.py   -> results/phase3_weight_scale_check.json
"""
import json
import os

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def fit(X, y, w):
    sc = StandardScaler().fit(X, sample_weight=w)
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000).fit(sc.transform(X), y, sample_weight=w)
    return sc, lr


def main():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 5))
    y = (X @ [1, -0.5, 0.3, 0, 0.2] + rng.normal(size=500) > 0).astype(int)
    w = rng.uniform(0.1, 2, 500)
    (s1, l1), (s10, l10) = fit(X, y, w), fit(X, y, 10 * w)
    out = {"sklearn": sklearn.__version__,
           "scaler_identical": bool(np.allclose(s1.mean_, s10.mean_) and np.allclose(s1.scale_, s10.scale_)),
           "coef_identical": bool(np.allclose(l1.coef_, l10.coef_, atol=1e-8)),
           "max_abs_coef_difference": float(np.abs(l1.coef_ - l10.coef_).max()),
           "coef_w": l1.coef_.ravel().tolist(), "coef_10w": l10.coef_.ravel().tolist(),
           "intercept_w": float(l1.intercept_[0]), "intercept_10w": float(l10.intercept_[0])}
    with open(os.path.join(REPO, "results", "phase3_weight_scale_check.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
