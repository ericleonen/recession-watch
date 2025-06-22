from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

MODELS = {
    "Logistic regression": {
        "pipeline": Pipeline(steps=[
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(
                penalty="l1",
                class_weight="balanced",
                solver="liblinear",
                random_state=42
            ))
        ]),
        "parameter_grid": {
            "tol": [1e-5, 1e-4, 1e-3],
            "max_iter": [100, 500, 1000],
        }
    },
    "SVM": {
        "pipeline": Pipeline(steps=[
            ("scale", StandardScaler()),
            ("classifier", SVC(
                kernel="rbf",
                probability=True,
                random_state=42
            ))
        ]), 
        "parameter_grid": {
            "C": [0.001, 0.01, 0.1, 1],
            "max_iter": [100, 500, 1000]
        }
    },
    "Random forest": {
        "pipeline": Pipeline(steps=[
            ("classifier", RandomForestClassifier(
                class_weight="balanced",
                random_state=42
            ))
        ]),
        "parameter_grid": {
            "n_estimators": [100, 200, 500],
            "max_depth": [4, 8, 12, None],
            "min_samples_split": [2, 4, 8],
            "min_samples_leaf": [1, 4, 16],
            "max_features": ["sqrt", 0.5, 0.7, 1.0],
            "max_leaf_nodes": [4, 8, 16, None],
            "max_samples": [0.5, 0.7, 1.0]
        }
    }
}