from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from sklearn.utils import compute_sample_weight
from sklearn import clone
from sklearn.metrics import average_precision_score, roc_auc_score, fbeta_score, accuracy_score, precision_score, recall_score
from models import models

METRICS = [
    "Average precision",
    "Weighted average precision",
    "ROC AUC",
    "Accuracy",
    "Weighted accuracy",
    "Precision",
    "Weighted precision",
    "Recall",
    "Weighted recall"
]

class RecessionPredictor:
    """
    Helps compare and choose models for the RecessionWatch classification problem.
    """

    def __init__(self, selected_models: list[str] = []):
        """
        Initializes a RecessionPredictor with the given list of models.
        """
        self.selected_models = selected_models

    def fit(self, X: pd.DataFrame, y: pd.Series, metric: str = "ROC AUC"):
        trained_models = {
            model_name: self._tune_train_model(X, y, models[model_name][0], models[model_name][1])
            for model_name in self.selected_models
        }

        self._create_model_table(X, y, trained_models)
        best_model_name = self._select_model_name(metric)
        best_pipeline = trained_models[best_model_name]

        self.best_model = {
            "name": best_model_name,
            "pipeline": best_pipeline
        }

    def predict_proba(self, X: pd.DataFrame):
        return pd.Series(self.best_model["pipeline"].predict_proba(X)[:, 1], index=X.index)

    def _create_model_table(self, X: pd.DataFrame, y: pd.Series, trained_models: dict):
        self.model_table = pd.concat([
            self._eval_model(X, y, name, pipeline) for name, pipeline in trained_models.items()
        ], axis=1)
    
    def _select_model_name(self, metric: str) -> str:
        return self.model_table.columns[np.argmax(self.model_table.loc[metric, :])]

    def _tune_train_model(
            self,
            X: pd.DataFrame, 
            y: pd.Series, 
            pipeline: Pipeline, 
            param_grid: dict
    ) -> tuple[Pipeline, float]:
        """
        Tunes the given model, defined by a Pipeline and a tuneable parameter grid dict, to
        maximize average precision under 5-fold walk-forward optimization with the given data.
        Returns a tuned model.
        """
        tscv = TimeSeriesSplit(n_splits=5)

        search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            scoring="average_precision",
            cv=tscv
        )
        search.fit(X, y)

        return search.best_estimator_

    def _eval_model(
            self,
            X: pd.DataFrame, 
            y: pd.Series, 
            name: str,
            pipeline: Pipeline
    ) -> pd.Series:
        """
        Evaluates the given tuned model and returning a Series of metrics: average precision,
        weighted average precision, ROC AUC, accuracy, weighted accuracy, precision, weighted
        precision, recall, and weighted recall.
        """
        pipeline = clone(pipeline)
        tscv = TimeSeriesSplit(n_splits=5)
        results = pd.Series(np.zeros(9), index=METRICS, name=name)

        for train_index, test_index in tscv.split(X):
            X_train = X.iloc[train_index]
            y_train = y.iloc[train_index]
            X_test = X.iloc[test_index]
            y_test = y.iloc[test_index]
            sample_weights_test = compute_sample_weight(class_weight="balanced", y=y_test)

            pipeline.fit(X_train, y_train)
            probas = pipeline.predict_proba(X_test)[:, 1]
            preds = probas >= 0.5

            results["Average precision"] += average_precision_score(y_test, probas)
            results["Weighted average precision"] += average_precision_score(
                y_test, 
                probas, 
                sample_weight=sample_weights_test
            )
            results["ROC AUC"] += roc_auc_score(y_test, probas)
            results["Accuracy"] += accuracy_score(y_test, preds)
            results["Weighted accuracy"] += accuracy_score(
                y_test, 
                preds, 
                sample_weight=sample_weights_test
            )
            results["Precision"] += precision_score(y_test, preds, zero_division=0)
            results["Weighted precision"] += precision_score(
                y_test, 
                preds, 
                sample_weight=sample_weights_test,
                zero_division=0
            )
            results["Recall"] += recall_score(y_test, preds)
            results["Weighted recall"] += recall_score(
                y_test, 
                preds, 
                sample_weight=sample_weights_test
            )

        return results / 5