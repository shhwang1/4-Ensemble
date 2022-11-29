import argparse

def Parser1():
    parser = argparse.ArgumentParser(description='4_Ensemble')

    # data type
    parser.add_argument('--data-path', type=str, default='./data/')
    parser.add_argument('--data-type', type=str, default='WineQuality.csv',
                        choices = ['Banking.csv', 'Breast.csv', 'Diabetes.csv', 'Glass.csv', 'PersonalLoan.csv', 'Stellar.csv', 'WineQuality.csv'])                       
    parser.add_argument('--seed-list', type=list, default=[2022, 2023, 2024, 2025, 2026])              
    parser.add_argument('--seed', type=int, default=2022)   
    # Choose methods
    parser.add_argument('--method', type=str, default='XGBoost',
                        choices = ['DecisionTree', 'RandomForest', 'Adaboost', 'GBM', 'XGBoost'])

    # Hyperparameters for Anomaly Detection
    parser.add_argument('--bagging', type=list, default=[True, False]) 
    parser.add_argument('--boosting', type=list, default=[True, False])
    parser.add_argument('--max-depth-list', type=list, default=[10, 20, 30, 40, 50])
    parser.add_argument('--n-estimators', type=int, default=50)
    parser.add_argument('--gbm-estimators', type=list, default=[100, 200, 300, 400, 500])
    parser.add_argument('--xgb-lr', type=list, default=[0.01, 0.05, 0.1, 0.15, 0.2])
    parser.add_argument('--xgb-depth', type=list, default=[4, 6, 8, 10, 12])
    parser.add_argument('--split', type=int, default=0.2)   

    return parser