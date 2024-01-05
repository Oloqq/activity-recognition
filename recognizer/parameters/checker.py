from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from ..raw_samples import load_raw_suite
import models.ML as ml
import models.utils as models_utils
import numpy as np
import json
import time

from itertools import product

def make_grid(param_dict):
    # Extract keys and values
    keys = param_dict.keys()
    values = product(*param_dict.values())

    # Create combinations of parameter values
    return [dict(zip(keys, v)) for v in values]

def search(summary, resample_f):
    path_train = "data/train"
    path_test = "data/test"

    train_suite = load_raw_suite(path_train)
    test_suite = load_raw_suite(path_test)
    train_suite.resample(resample_f)
    test_suite.resample(resample_f)
    X_train, y_train = ml.load_data(train_suite)
    X_test, y_test = ml.load_data(test_suite)

    def do_grid(classifier, parameters):
        key = classifier.__class__.__name__
        grid: list[dict] = make_grid(parameters)
        results = []

        print(f"grid: {key} {parameters} -> {grid}")
        for param in grid:
            print(f"param: {param}")
            cls = classifier.__class__(**param)
            cls.fit(X_train, y_train)
            print(f"param: {param}")
            y_pred = cls.predict(X_test)
            accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
            results.append({
                "classifier": key,
                "params": param,
                "accuracy": accuracy
            })
        print(f"results: {results}")
        json.dump(results, open(f"parameter_comparison/{key}_resample={resample_f}_{time.time()}.json", "w"), indent=4)
        summary.extend(results)

    # do_grid(SVC(), {'kernel': ['linear'], 'C': [1, 2]})
    do_grid(KNeighborsClassifier(), {
        'n_neighbors': [1, 2, 3, 10],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
        })

    do_grid(SVC(), {'kernel': ['linear', 'rbf'], 'C': [1, 10, 12, 15, 20]})

    do_grid(DecisionTreeClassifier(), {
        'max_depth': [2, 5, 8, 12, 13, 14, 15],
        'min_samples_split': [2, 4, 6],
        'criterion': ["gini", "entropy", "log_loss"],
        'splitter': ["random"]
        })

    # do_grid(RandomForestClassifier(), {'max_depth': [10, 20, 30], 'n_estimators': [5, 10, 20]})

def main():
    summary = []
    resample_f = 100
    search(summary, resample_f)
    json.dump(summary, open(f"parameter_comparison/summary_with_resample={resample_f}_{time.time()}.json", "w"), indent=4)