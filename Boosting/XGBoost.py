import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score

def ExtremeGradientBoosting(args):

    result_df = pd.DataFrame(columns = ['Seed', 'n_estimators', 'Accuracy', 'F1-Score'])

    data = pd.read_csv(args.data_path + args.data_type)

    X_data = data.iloc[:, :-1]
    y_data = data.iloc[:, -1]

    if args.data_type == 'WineQuality.csv':
        y_data -= 3

    elif args.data_type == 'Glass.csv':
        y_data -= 1
        
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_data)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_data, test_size = 0.2, shuffle = True, random_state = args.seed)

    model = XGBClassifier()

    xgb_param_grid = {
        'n_estimators' : args.gbm_estimators,
        'learning_rate' : args.xgb_lr,
        'max_depth' : args.xgb_depth
    }

    xgb_grid = GridSearchCV(model, param_grid = xgb_param_grid, scoring='accuracy', n_jobs = -1, verbose = 1)
    xgb_grid.fit(X_train, y_train)

    print('Best Accuracy : {0:.4f}'.format(xgb_grid.best_score_))
    print('Best Parameters :', xgb_grid.best_params_)

    result_df = pd.DataFrame(xgb_grid.cv_results_)
    result_df.sort_values(by=['rank_test_score'], inplace=True)

    result_df[['params', 'mean_test_score', 'rank_test_score']].head(5)

    return result_df        