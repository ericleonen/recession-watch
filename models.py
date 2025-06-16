from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

lin_reg = (
    "Linear Regression",
    make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty="l1",
            class_weight="balanced",
            solver="liblinear",
            random_state=42
        )
    ),
    {
        "logisticregression__tol": [1e-5, 1e-4, 1e-3],
        "logisticregression__max_iter": [100, 500, 1000],
    }
)

svm = (
    "Support Vector Machine", 
    make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        )
    ), 
    {
        "svc__C": [0.001, 0.01, 0.1, 1],
        "svc__max_iter": [100, 500, 1000]
    }
)