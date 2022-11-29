import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score

def GradientBoosting(args):

    result_df = pd.DataFrame(columns = ['Seed', 'n_estimators', 'Accuracy', 'F1-Score'])


    for n_estimators in args.gbm_estimators:
        for seed in args.seed_list:

            data = pd.read_csv(args.data_path + args.data_type)

            X_data = data.iloc[:, :-1]
            y_data = data.iloc[:, -1]

            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X_data)

            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_data, test_size = 0.2, shuffle = True, random_state = seed)

            model = GradientBoostingClassifier(n_estimators=n_estimators,
                                                    random_state = seed)

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_pred, y_test)
            f1score = f1_score(y_pred, y_test, average='weighted')
            
            print('Decision Tree n_estimators =', n_estimators)
            print('Accuracy :', accuracy, 'F1-Score :', f1score)

            result = {
                        'Seed' : seed,
                        'n_estimators' : n_estimators,
                        'Accuracy' : accuracy,
                        'F1-Score' : f1score}
            
            result = pd.DataFrame([result])

            result_df = pd.concat([result_df, result], ignore_index = True)

    return result_df