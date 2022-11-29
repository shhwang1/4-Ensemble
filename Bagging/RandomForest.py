import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def Random_Forest(args):

    result_df = pd.DataFrame(columns = ['Seed', 'Depth', 'Accuracy', 'F1-Score'])

    for depth in args.max_depth_list:
        for seed in args.seed_list:

            data = pd.read_csv(args.data_path + args.data_type)

            X_data = data.iloc[:, :-1]
            y_data = data.iloc[:, -1]

            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X_data)

            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_data, test_size = 0.2, shuffle = True, random_state = seed)

            model = RandomForestClassifier(max_depth = depth,
                                            n_estimators=args.n_estimators,
                                            random_state = seed)
                                            
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_pred, y_test)
            f1score = f1_score(y_pred, y_test, average='weighted')
            
            print('Decision Tree max_depth =', depth)
            print('Accuracy :', accuracy, 'F1-Score :', f1score)

            result = {
                        'Seed' : seed,
                        'Depth' : depth,
                        'Accuracy' : accuracy,
                        'F1-Score' : f1score}
            
            result = pd.DataFrame([result])

            result_df = pd.concat([result_df, result], ignore_index = True)
    
    return result_df