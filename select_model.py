from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from sklearn.utils import compute_sample_weight
from sklearn import clone
from sklearn.metrics import fbeta_score, average_precision_score, roc_auc_score, fbeta_score, accuracy_score, precision_score, recall_score

class ModelSelector:
    """
    Helps compare and choose models for the RecessionWatch classification problem.
    """

    def __init__(self, models: list[tuple[str, Pipeline, dict]] = []):
        """
        Initializes a ModelSelector with the given list of models. Each model is specified as
        a name, a Pipeline, and a tuneable parameter grid dict.
        """
        self.models = models
        
    def add_model(self, name: str, pipeline: Pipeline, param_grid: dict):
        """
        Adds the specified model, defined by a name, a Pipeline, and a tuneable parameter grid
        dict, to the list of possible models.
        """
        self.models.append((name, pipeline, param_grid))

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.trained_models = {
            name: self._tune_model(X, y, pipeline, param_grid)
            for name, pipeline, param_grid in self.models
        }

    def compare_models(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return pd.concat([
            self._eval_model(X, y, name) for name in self.trained_models.keys()
        ], axis=1)
    
    def select_model(self, X: pd.DataFrame, y: pd.Series, metric: str = "ROC AUC") -> str:
        metrics = self.compare_models(X, y)

        return metrics.columns[np.argmax(metrics.loc[metric, :])]

    def _tune_model(
            self,
            X: pd.DataFrame, 
            y: pd.Series, 
            pipeline: Pipeline, 
            param_grid: dict
    ) -> tuple[Pipeline, float]:
        """
        Tunes the given model, defined by a Pipeline and a tuneable parameter grid dict, to
        maximize average precision under 5-fold walk-forward optimization with the given data. Then
        finds the best threshold for optimizing F2 score. Returns a tuned model and its threshold.
        """
        tscv = TimeSeriesSplit(n_splits=5)

        search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            scoring="average_precision",
            cv=tscv
        )
        search.fit(X, y)

        best_model = search.best_estimator_
        
        probas = best_model.predict_proba(X)[:, 1]
        thresholds = np.linspace(0, 1, 101)
        f2_scores = [fbeta_score(y, probas >= t, beta=2) for t in thresholds]
        best_threshold = thresholds[np.argmax(f2_scores)]

        return best_model, best_threshold

    def _eval_model(
            self,
            X: pd.DataFrame, 
            y: pd.Series, 
            name: str
    ) -> Pipeline:
        """
        Evaluates the given tuned model and returning a Series of metrics: average precision,
        weighted average precision, ROC AUC, accuracy, weighted accuracy, precision, weighted
        precision, recall, and weighted recall.
        """
        model, threshold = self.trained_models[name]
        model = clone(model)
        tscv = TimeSeriesSplit(n_splits=5)
        results = pd.Series(np.zeros(9), index=[
            "Average Precision",
            "Weighted Average Precision",
            "ROC AUC",
            "Accuracy",
            "Weighted Accuracy",
            "Precision",
            "Weighted Precision",
            "Recall",
            "Weighted Recall"
        ], name=name)

        for train_index, test_index in tscv.split(X):
            X_train = X.iloc[train_index]
            y_train = y.iloc[train_index]
            X_test = X.iloc[test_index]
            y_test = y.iloc[test_index]
            sample_weights_test = compute_sample_weight(class_weight="balanced", y=y_test)

            model.fit(X_train, y_train)
            probas = model.predict_proba(X_test)[:, 1]
            preds = probas >= threshold

            results["Average Precision"] += average_precision_score(y_test, probas)
            results["Weighted Average Precision"] += average_precision_score(
                y_test, 
                probas, 
                sample_weight=sample_weights_test
            )
            results["ROC AUC"] += roc_auc_score(y_test, probas)
            results["Accuracy"] += accuracy_score(y_test, preds)
            results["Weighted Accuracy"] += accuracy_score(
                y_test, 
                preds, 
                sample_weight=sample_weights_test
            )
            results["Precision"] += precision_score(y_test, preds, zero_division=0)
            results["Weighted Precision"] += precision_score(
                y_test, 
                preds, 
                sample_weight=sample_weights_test,
                zero_division=0
            )
            results["Recall"] += recall_score(y_test, preds)
            results["Weighted Recall"] += recall_score(
                y_test, 
                preds, 
                sample_weight=sample_weights_test
            )

        return results / 5