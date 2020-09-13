# -*- coding: utf-8 -*-
import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from models import BernoulliNaiveBayes as BNB
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import GridSearchCV
from data import project_dir, processed_path
from train import (
    models_path,
    num_bnb_features,
    get_train_test_data,
    report,
)

reports_path = project_dir / 'reports'

def main():
    logger = logging.getLogger(__name__)
    logger.info('predicting train and test data')

    data_names = (
        'X_train.json',
        'X_test.json',
        'y_train.json',
    )
    X_train, X_test, y_train = get_train_test_data(processed_path, data_names)

    model_names = [
        'BernoulliNaiveBayes.pkl',
        'LogisticRegression.pkl',
        'SupportVectorMachine.pkl'
    ]
    models = get_models(models_path, model_names)

    logger.info('scoring training predictions')
    predict_train(models, X_train, y_train)

    logger.info('making predictions on test data')
    predict_test(models, [name.split('.')[0] for name in model_names], X_test)

def predict_train(models, X_train, y_train):
    for model in models:
        if isinstance(model, GridSearchCV):
            report(model.cv_results_)
        else:
            X = X_train if not isintance(model, BNB) else X_train[:num_bnb_features]
            y_train_pred = model.predict(X)
            print(metrics.classification_report(
                y_train, y_train_pred))

def predict_test(models, model_names, X_test):
    for model, name in zip(models, model_names):
        y_test_pred = model.predict(X_test)

        # Export to CSV file
        pd.DataFrame(y_test_pred,
            columns=['Category']).to_csv(reports_path / (name + '.csv'))

def get_models(input_path, filenames):
    return [get_model(input_path / filename) for filename in filenames]

def get_model(path):
    return pickle.load(open(path, 'rb'))

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
