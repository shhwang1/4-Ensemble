# Ensemble Tutorial

## Table of Contents

#### 0. Overview of Ensemble
___
### Bagging
#### 1. Decision Tree (DT)
#### 2. Random Forest (RF)
***
### Boosting
#### 3.Adaptive Boosting (AdaBoost)
#### 4. Gradient Boosting Machine (GBM)
#### 5. Extreme Gradient Boosting (XGBoost)
___

## 0. Overview of Ensemble
![image](https://user-images.githubusercontent.com/115224653/204128272-c65bf7d7-a25e-491b-a06a-cfde1837ac0f.png)
### - What is "Ensemble?"
Ensemble is a French word for unity and harmony. It is mainly used in music to mean concerto on various instruments. 

A large number of small instrumental sounds are harmonized to create a more magnificent and beautiful sound. 

Of course, you shouldn't, but one tiny mistake can be buried in another sound.    

Ensemble in machine learning is similar. Several weak learners gather to form a stronger strong learner through voting. 

Since there are many models, even if the prediction is misaligned in one model, it is somewhat corrected. That is, a more generalized model is completed.   

The two goals of the ensemble are as follows.

### 1. How do we ensure diversity?
### 2. How do you combine different results from different models?
___
## Dataset

We use 7 datasets for Classification (Banking, Breast, Diabetes, Glass, PersonalLoan, Stellar, Winequality)   

Banking dataset : <https://www.kaggle.com/datasets/rashmiranu/banking-dataset-classification>     
Breast dataset : <https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data>   
Diabetes dataset : <https://www.kaggle.com/datasets/mathchi/diabetes-data-set>   
Glass datset : <https://www.kaggle.com/datasets/uciml/glass>      
PersonalLoan dataset : <https://www.kaggle.com/datasets/teertha/personal-loan-modeling>   
Steallr dataset : <https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17>   
WineQuality dataset : <https://archive.ics.uci.edu/ml/datasets/wine+quality>   

In all methods, data is used in the following form.
``` C
import argparse

def Parser1():
    parser = argparse.ArgumentParser(description='4_Ensemble')

    # data type
    parser.add_argument('--data-path', type=str, default='./data/')
    parser.add_argument('--data-type', type=str, default='Banking.csv',
                        choices = ['Banking.csv', 'Breast.csv', 'Diabetes.csv', 'Glass.csv', 'PersonalLoan.csv', 'Steallr.csv', 'WineQuality.csv'])            
                        
data = pd.read_csv(args.data_path + args.data_type)

X_data = data.iloc[:, :-1]
y_data = data.iloc[:, -1]
```
___

# Bagging

## What is "Bagging"?   
   

![Bagging](https://www.simplilearn.com/ice9/free_resources_article_thumb/Bagging.PNG)

Bagging is one of the enamble methods that increases the performance of the model.  

If you look at the above picture, you can see that the same data with the same color is duplicated in bootstrap.   

 Like this, Bagging creates multiple subset datasets while allowing duplicate extraction from the original dataset, which is called "Bootstrap", and that is why Bagging is also called Bootstrap aggregating.  

 Bagging is characterized by learning models in parallel using Bootstrap, and with this, Bagging is used to deal with bias-variance trade-offs and reduces the variance of a prediction model.

### Bagging is effective when using base learners with high model complexity.   

Bagging avoids overfitting of data and is used for both regression and classification models, specifically for decision tree algorithms.   

This tutorial covers the case of using Decision Tree and Random Forest as base learners by applying Bagging.   

In the analysis part, an ablation study is conducted on the performance difference between when bagging is applied and when not applied.
___
   
## 1. Decision Tree (DT)

<p align="center"><img src="https://regenerativetoday.com/wp-content/uploads/2022/04/dt.png" width="650" height="400"></p> 

Decision tree analyzes the data and represents a pattern that exists between them as a combination of predictable rules and is called a decision tree because it looks like a 'tree'.

The above example is a binary classification problem that determines yes=1 if the working conditions for the new job are satisfied and no=0 if not satisfied.

As shown in the picture above, the initial point is called the root node and the corresponding number of data decreases as the branch continues.

Decision trees are known for their high predictive performance compared to computational complexity.

In addition, it has the strength of explanatory power in units of variables, but decision trees are likely to work well only on specific data because the decision boundary is perpendicular to the data axis.

The model that emerged to overcome this problem is Random Forest, a technique that improves predictive performance by combining the results by creating multiple decision trees for the same data.
___
## Python Code
``` py
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def Decision_Tree(args):

    result_df = pd.DataFrame(columns = ['Bagging', 'Seed', 'Depth', 'Accuracy', 'F1-Score'])

    for bagging in args.bagging:
        for depth in args.max_depth_list:
            for seed in args.seed_list:

                data = pd.read_csv(args.data_path + args.data_type)

                X_data = data.iloc[:, :-1]
                y_data = data.iloc[:, -1]

                scaler = MinMaxScaler()
                X_scaled = scaler.fit_transform(X_data)

                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_data, test_size = 0.2, shuffle = True, random_state = seed)

                if bagging == True:
                    model = BaggingClassifier(base_estimator = DecisionTreeClassifier(max_depth = depth),
                                                                n_estimators=args.n_estimators,
                                                                random_state = seed)
                else:
                    model = DecisionTreeClassifier(max_depth=depth)

                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                accuracy = accuracy_score(y_pred, y_test)
                f1score = f1_score(y_pred, y_test, average='weighted')
                
                print('Decision Tree max_depth =', depth)
                print('Accuracy :', accuracy, 'F1-Score :', f1score)

                result = {'Bagging' : bagging,
                            'Seed' : seed,
                            'Depth' : depth,
                            'Accuracy' : accuracy,
                            'F1-Score' : f1score}
                
                result = pd.DataFrame([result])

                result_df = pd.concat([result_df, result], ignore_index = True)
    
    return result_df
```
Unlike the previous tutorials, this tutorial conducted repeated experiments. 

A total of five random seed values were allocated and repeated. 

In the middle, the args.bagging part is for comparing the performance when bagging is applied and when not applied. 

In the Decision Tree part, the experimental construction is how much performance has improved when bagging is applied. 

This part will be examined in the analysis part later.

___

## 2. Random Forest
<p align="center"><img src="https://i0.wp.com/thaddeus-segura.com/wp-content/uploads/2020/09/rfvsdt.png?fit=933%2C278&ssl=1" width="900" height="300"></p>

Random Forest is an enamble model that improves predictive power by reducing the correlation of individual trees by taking advantage of existing bagging and adding a process of randomly selecting variables.   

Let's take a closer look at the above definition below.

### 1. Random Forest is an ensemble model using Bagging.
Random forest basically uses Bagging. Therefore, Random Forest will also take over the effect of lowering dispersion while maintaining the bias, which is the advantage of Bagging.

### 2. Random Forest improves prediction by reducing the correlation of individual trees through the process of randomly selecting variables.
Random Forest uses the Bootstrap sample dataset to create several individual trees. In Breiman et al.'s 'Random Forest' paper, it is proved that a smaller correlation between individual trees results in a smaller generalization error of the random forest. In other words, reducing the correlation of individual trees means that the predictive power of the random forest is improved.

### It is important to randomly select candidates for variables to separate individual trees!
___

#### Python Code
``` py
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
```
Random Forest part will cover the comparison of performance with the case of Decision Tree without Bagging and Decision Tree with Bagging, not the experiment of tuning the hyperparameter.
___
# Boosting

## What is "Boosting"?   
   

![Boosting](https://velog.velcdn.com/images/iguv/post/3bc28cf5-23f8-4c8b-b509-52c25382f564/image.png)

Boosting is an ensemble method that combines weak learners with poor performance to build a good performance model, and the sequentially generated weak learners compensate for the shortcomings of the previous step in each step.

The training method depends on the type of boosting process called the boosting algorithm. 

However, the algorithm trains the boosting model by following the following general steps:

1. The boosting algorithm assigns the same weight to each data sample. It supplies data to the first machine model, called the basic algorithm. The basic algorithm allows you to make predictions for each data sample.

2. The boosting algorithm evaluates model predictions and increases the weight of samples with more serious errors. It also assigns weights based on model performance. Models that produce excellent predictions have a significant impact on the final decision.

3. The algorithm moves the weighted data to the next decision tree.

4. The algorithm repeats steps 2 and 3 until the training error instance falls below a certain threshold.

### Boosting is effective when using base learners with low model complexity!   
___
## 3. Adaptive Boosting (AdaBoost)

<p align="center"><img src="https://cdn-images-1.medium.com/max/800/1*7TF0GggFTqetjxqU5cnuqA.jpeg" width="750" height="300"></p> 

The Adaboost algorithm is a classification-based machine learning model, a method of synthesizing one strong classifier that performs better by weight modification by building and combining a large number of weak classifiers with slightly lower predictive performance. 

The Adaboost model has the advantage of repeatedly modifying and combining weights through mistakes in weak classifiers, and not compromising predictive performance due to less overfitting of the learning data.

In other words, it is the principle of generating a final strong classifier by adding the product of the weight of the weak classifier and the value of the weak classifier.
___
#### Python Code
``` py
import numpy as np
import pandas as pd
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score

def AdaptiveBoosting(args):

    result_df = pd.DataFrame(columns = ['Boosting', 'Seed', 'Depth', 'Accuracy', 'F1-Score'])

    for boosting in args.boosting:
        for depth in args.max_depth_list:
            for seed in args.seed_list:

                data = pd.read_csv(args.data_path + args.data_type)

                X_data = data.iloc[:, :-1]
                y_data = data.iloc[:, -1]

                scaler = MinMaxScaler()
                X_scaled = scaler.fit_transform(X_data)

                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_data, test_size = 0.2, shuffle = True, random_state = seed)

                if boosting == True:
                    model = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth = depth),
                                                                n_estimators=args.n_estimators,
                                                                random_state = seed)
                else:
                    model = DecisionTreeClassifier(max_depth=depth)

                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                accuracy = accuracy_score(y_pred, y_test)
                f1score = f1_score(y_pred, y_test, average='weighted')
                
                print('Decision Tree max_depth =', depth)
                print('Accuracy :', accuracy, 'F1-Score :', f1score)

                result = {'Boosting' : boosting,
                            'Seed' : seed,
                            'Depth' : depth,
                            'Accuracy' : accuracy,
                            'F1-Score' : f1score}
                
                result = pd.DataFrame([result])

                result_df = pd.concat([result_df, result], ignore_index = True)
    
    return result_df
```
In the middle, the args.boosting part is for comparing the performance when boosting is applied and when not applied. Like the Decision Tree part, the results of the Decision Tree with Adaptive Boosting and the results of the Decision Tree without Adaptive Boosting will be compared in the analysis part.
___
## 4. Gradient Boosting Machine (GBM)

<p align="center"><img src="https://www.akira.ai/hubfs/Imported_Blog_Media/akira-ai-gradient-boosting-ml-technique.png" width="750" height="300"></p> 

Gradient Boosting Machines (GBM) is a way to understand the concept of Boosting as an optimization method called Gradient Descent.

As I explained before, boosting is a method of adding and sequential learning multiple trees and synthesizing the results.

GBM, like AdaBoost, is an algorithm of the Boosting family, so it complements the residual of the previous learner by creating a weak learner sequentially, but the two have different methods of complementing the residual.

AdaBoost addresses previously misclassified data by giving more weight to well-classified learners.

By comparison, GBM is a method of updating predictions by fitting weak learners to the residuals themselves and adding the predicted residuals to the previous predictions.
___
#### Python Code
``` py
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
```
In the experimental part of Gradient Boosting Machines, we adjust the n_estimators hyperparameter and compare the resulting performance changes while adjusting the model complexity. 

This part will be examined in the analysis part.
___
## 5. eXtra Gradient Boost (XGBoost)
![xgboost](https://www.researchgate.net/publication/345327934/figure/fig3/AS:1022810793209856@1620868504478/Flow-chart-of-XGBoost.png)

Gradient Boost is a representative algorithm implemented using the Boosting technique.

The library that implements this algorithm to support parallel learning is eXtra Gradient Boost (XGBost).

It supports both Regression and Classification problems, and is a popular algorithm with good performance and resource efficiency.

Because it learns through parallel processing, the classification speed is faster than that of general GBM.

In addition, in the case of standard GBM, there is no overfitting regulation function, but XGBoost itself has strong durability as an overfitting regulation function.

It has an Early Stopping function, offers a variety of options, and is easy to customize.

___


### Python Code
``` py
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
```
In the experimental part of XGBoost, grid search for hyperparameters n_estimators, learning_rate, and max_depth was conducted.

Let's compare the best performance when each dataset has a hyperparameter.


___

## Analysis


## [Experiment 1.] Decision Tree - DT Performance Comparison by Bagging Application

In the Python code section of the decision tree, it was possible to set whether to apply bagging through the arg.bagging argument.

It is intended to understand the effect of model complexity and bagging on performance by comparing the performance according to the pre-set max_depth value with the application of bagging.

As for the performance evaluation metric, accuracy and F1 score were used as in the previous tutorial.

First of all, the table below is a performance table of the decision tree without bagging. 

### All experiments are the results of five repeated experiments by changing the seed value.

|  Accuracy  | Dataset              |  Depth = 10 |   Depth = 20 |   Depth = 30 |  Depth = 40 |   Depth = 50 |
|:----------:|:--------------------:|:------:|:--------:|:--------:|:--------:|:--------:|
|          1 | Banking            |    **0.8345 (±0.008)** |        0.7881 (±0.005)|         0.7820 (±0.003)|        0.7824 (±0.004) |        0.7830 (±0.008)|
|          2 | Breast                 |    0.9368 (±0.022) |        0.9404 (±0.010)|         **0.9456 (±0.012)**|        0.9404 (±0.019) |        0.9404 (±0.030)|
|          3 | Diabetes                 |    **0.7013 (±0.025)**|        0.6857 (±0.040) |        0.6922 (±0.044) |        0.6896 (±0.036) |        0.6935 (±0.037) |
|          4 | Glass                 |   **0.6837 (±0.066)** |        0.6698 (±0.045) |        0.6744 (±0.033) |        0.6514 (±0.057) |        0.6514 (±0.004) |
|          5 | PersonalLoan                 |    0.9822 (±0.002)|        0.9814 (±0.001) |        0.9820 (±0.002) |        **0.9824 (±0.003)** |        0.9814 (±0.002) | 
|          6 | Stellar                 |    **0.9676 (±0.002)** |        0.9614 (±0.002) |        0.9599 (±0.002) |        0.9609 (±0.002) |        0.9608 (±0.002) | 
|          7 | WineQuality                 |    0.6069 (±0.027) |        0.6150 (±0.032) |        **0.6175 (±0.026)** |        0.6138 (±0.026) |        0.6125 (±0.025)  |

|  F1-Score  | Dataset              |  Depth = 10 |   Depth = 20 |   Depth = 30 |  Depth = 40 |   Depth = 50 |
|:----------:|:--------------------:|:------:|:--------:|:--------:|:--------:|:--------:|
|          1 | Banking            |    **0.8607 (±0.003)** |        0.8494 (±0.003)|         0.8476 (±0.003)|        0.8475 (±0.003) |        0.8475 (±0.003)|
|          2 | Breast                 |    **0.9647 (±0.007)** |        0.9631 (±0.065)|         0.9632 (±0.065)|        0.9632 (±0.065) |        0.9632 (±0.065)|
|          3 | Diabetes                 |    **0.7688 (±0.017)**|        0.7649 (±0.020) |        0.7649 (±0.020) |        0.7649 (±0.020) |        0.7649 (±0.020) |
|          4 | Glass                 |   **0.7395 (±0.037)** |        0.7256 (±0.027) |        0.7256 (±0.027) |        0.7256 (±0.027) |        0.7256 (±0.027) |
|          5 | PersonalLoan                 |    **0.9884 (±0.003)**|        0.9880 (±0.003) |        0.9880 (±0.003) |        0.9880 (±0.003) |        0.9880 (±0.003) | 
|          6 | Stellar                 |    0.9761 (±0.002) |        **0.9767 (±0.002)** |        0.9766 (±0.002) |        0.9766 (±0.002) |        0.9766 (±0.002) | 
|          7 | WineQuality                 |    0.6725 (±0.021) |        0.6763 (±0.022) |        **0.6788 (±0.025)** |        0.6787 (±0.025) |        0.6787 (±0.025)  |

Analyzing the experimental results can be summarized as follows.

#### 1. Compared to when bagging was not applied, performance was improved on all datasets when applied.
#### 2. When the model complexity is high, the performance of bagging is generally good. However, the performance was rather good when the max_depth value related to the model complexity was lower than when it was high.
#### 3. Setting the max_depth hyperparameter value seems to have an important effect on performance.
#### 4. The deviation between repeated experiments was low.
___

### [Experiment 2.] Comparison of k-NN anomaly detection performance by neighbors hyperparameter changes
Like the Local Outlier Factor, the role of k-NN anonymous detection is also important for hyperparameter K. Experimentally check whether the role of K, which was insignificant in the Local Outlier Factor, is different in k-NN anonymous detection. The change pattern of K was configured in the same way as the Local Outlier Factor.

|  Accuracy  | Dataset              |  K = 5 |   K = 10 |   K = 15 |   K = 20 |   K = 30 |   K = 50 |
|:----------:|:--------------------:|:------:|:--------:|:--------:|:--------:|:--------:|:--------:|
|          1 | Cardiotocogrpahy            |    0.9044 |        0.9049 |         0.9066|        0.9082 |        **0.9131**|        0.9121|
|          2 | Glass                 |    **0.9533** |        0.9439|       0.9393 |        0.9392 |        0.9392|        0.9252|
|          3 | Lympho                 |    0.9594 |        0.9594 |        0.9662 |        0.9662 |        0.9797 |        **0.9865** |
|          4 | Seismic                 |    0.9342 |        0.9342 |       **0.9342** |        0.9299 |        0.9276 |        0.9249 |
|          5 | Shuttle                 |    0.9284 |        0.9284 |        0.9284 |        0.9285 |        0.9285 |        0.9285 |
|          6 | Annthyroid                 |    0.9258 |        0.9257 |        0.9256 |        0.9257 |        0.9254 |        **0.9259** |
|          7 | Mammography                 |    0.9767 |        0.9767 |        0.9767 |        **0.9768** |        0.9766 |        0.9767 |

|  F1-Score  | Dataset              |  K = 5 |   K = 10 |   K = 15 |   K = 20 |   K = 30 |   K = 50 |
|:----------:|:--------------------:|:------:|:--------:|:--------:|:--------:|:--------:|:--------:|
|          1 | Cardiotocogrpahy            |    0.9498 |        0.9500 |         0.9508|        0.9517 |        **0.9541**|        0.9533|
|          2 | Glass                 |    **0.9761** |        0.9712|        0.9687 |        0.9685 |        0.9685|        0.9610 |
|          3 | Lympho                 |    0.9793 |        0.9793 |        0.9827 |        0.9827 |        0.9895 |        **0.9930** |
|          4 | Seismic                 |    0.9659 |        0.9659 |        **0.9660** |        0.9637 |        0.9624 |        0.9609 |
|          5 | Shuttle                 |    0.9629 |        0.9629 |        0.9629 |        0.9629 |        0.9629 |        0.9629 |
|          6 | Annthyroid                 |    0.9614 |        0.9614 |        0.9614 |        0.9614 |        0.9612 |        **0.9615** |
|          7 | Mammography                 |    0.9882 |        0.9882 |        0.9882 |        0.9882 |        0.9881 |       **0.9882** |

Analyzing the experimental results can be summarized as follows.

#### 1. It seems that the effect is weaker in k-NN than the weak effect of K in the Local Outlier Factor.
#### 2. Cardiotography datasets have lower performance than other datasets. This seems to be a characteristic of a dataset that should not be approached based on distance (K).
#### 3. K's role seems to be meaningful only when the threshold for determining the outsider is clearly established.

___
### [Experiment 3.] Effects of Masking on Auto-encoder
In Auto-encoder, the MAD_Score described above is set to threshold. In this experiment, the performance when masking is applied and when not applied is compared. If masking is applied, the ratio of masking, args.Compare the masking_ratio in 0.1, 0.2, 0.3, and 0.4 cases. Epoch is 300, batch size is 128.

#### 1) No Masking Case
|  Accuracy  | Dataset              |  Base |
|:----------:|:--------------------:|:------:|
|          1 | Cardiotocogrpahy            |    0.7554 |
|          2 | Glass                 |    0.7600 |
|          3 | Lympho                 |    0.9429 |
|          4 | Seismic                 |    0.7320 |
|          5 | Shuttle                 |    0.9865 |
|          6 | Annthyroid                 |    0.7479 |
|          7 | Mammography                 |    0.9100 |

|  F1-Score  | Dataset              |  Base |
|:----------:|:--------------------:|:------:|
|          1 | Cardiotocogrpahy            |    0.8211 |
|          2 | Glass                 |    0.8500 |
|          3 | Lympho                 |    0.9666 |
|          4 | Seismic                 |    0.8262 |
|          5 | Shuttle                 |    0.9907 |
|          6 | Annthyroid                 |    0.8448 |
|          7 | Mammography                 |    0.9519 |

And the following is the result table of applying masking.

|  Accuracy  | Dataset              |  Base |   Masking ratio = 0.1 |   Masking ratio = 0.2 |   Masking ratio = 0.3 |   Masking ratio = 0.4 |
|:----------:|:--------------------:|:------:|:--------:|:--------:|:--------:|:--------:|
|          1 | Cardiotocogrpahy            |    0.7554 |        0.7377 |         **0.8560**|        0.7949 |        0.7929|
|          2 | Glass                 |    0.7600 |        0.8000|       0.7600 |        **0.8600** |        0.7400|
|          3 | Lympho                 |    **0.9429** |        0.8857 |        0.8000 |        0.8857 |        0.9143 |
|          4 | Seismic                 |    0.7320 |       **0.7810**|       0.6554|        0.7795|        0.7688|
|          5 | Shuttle                 |    0.9865|        0.9854|        0.9842|        0.8383|        **0.9868**|
|          6 | Annthyroid                 |    0.7479 |        **0.7602**|        0.7190|        0.7126|        0.7093|
|          7 | Mammography                 |    0.9100 |        **0.9104**|        0.8990|        0.8810|        0.8936|

|  F1-Score  | Dataset              |  Base |   Masking ratio = 0.1 |   Masking ratio = 0.2 |   Masking ratio = 0.3 |   Masking ratio = 0.4 |
|:----------:|:--------------------:|:------:|:--------:|:--------:|:--------:|:--------:|
|          1 | Cardiotocogrpahy            |    0.8211 |        0.8261 |         **0.9004**|        0.8628 |        0.8575|
|          2 | Glass                 |    0.8500 |        0.8863|       0.8537 |        **0.9195**|        0.8395|
|          3 | Lympho                 |    **0.9666** |        0.9333 |        0.8814 |        0.9333 |        0.9474 |
|          4 | Seismic                 |    0.8262 |        **0.8563**|       0.7508|        0.8569|        0.8527|
|          5 | Shuttle                 |    0.9907|        0.9900|        0.9891|        0.8993|        **0.9916**|
|          6 | Annthyroid                 |    0.8448 |        **0.8497**|        0.8289|        0.8275|        0.8283|
|          7 | Mammography                 |    **0.9519** |        0.9497|        0.9428|        0.9318|        0.9438|

Analyzing the experimental results can be summarized as follows.

#### 1. In the dataset (Cardio, Glass), where the accuracy was relatively low, the masking effect appeared quite a bit.
#### 2. The effect was insignificant on the highly accurate datasets (Shuttle, Lympho, and Mammography).
#### 3. It was judged that it was worth trying masking on a dataset where the threshold was not accurately built.
___
### [Experiment 4.] Comparison of Isolation Forest anomaly detection performance by n_esitimators hyperparameter changes
The most important hyperparameter in the code of the Isolation Forest methodology is n_estimators, which represent the number of trees. In the case of max_sample hyperparameter, it can be set to 'auto', so it is utilized. The experimental table below shows the results according to the change in the number of n_estimators.

|  Accuracy  | Dataset              |  n_estimators = 10 |   n_estimators = 50 |   n_estimators = 100 |   n_estimators = 200 |   n_estimators = 400 |
|:----------:|:--------------------:|:------:|:--------:|:--------:|:--------:|:--------:|
|          1 | Cardiotocogrpahy            |    0.9093|        0.9132|         **0.9143**|        0.9143|        0.9143|
|          2 | Glass                 |    0.9532|        **0.9533**|       0.9533|        0.9533|        0.9533|
|          3 | Lympho                 |    0.9730|        **0.9730**|        0.9730|        0.9730|        0.9730|
|          4 | Seismic                 |    0.9257|       **0.9280**|       0.9280|        0.9280|        0.9280|
|          5 | Shuttle                 |    0.9382|        **0.9384**|        0.9384|        **0.9419**|        0.9419|
|          6 | Annthyroid                 |    0.9328|        0.9283|        **0.9286**|        0.9283|        0.9283|
|          7 | Mammography                 |    0.9747 |        **0.9784**|        0.9757|        0.9747|        0.9751|

|  F1-Score  | Dataset              |  n_estimators = 10 |   n_estimators = 50 |   n_estimators = 100 |   n_estimators = 200 |   n_estimators = 400 |
|:----------:|:--------------------:|:------:|:--------:|:--------:|:--------:|:--------:|
|          1 | Cardiotocogrpahy            |    0.9521|        0.9541|         **0.9547**|        0.9547|        0.9547|
|          2 | Glass                 |    **0.9760**|        0.9760|       0.9760|        0.9760|        0.9760|
|          3 | Lympho                 |    **0.9861**|        0.9861|        0.9861|        0.9861|        0.9861|
|          4 | Seismic                 |    0.9614|       0.9626|       0.9626|        **0.9280**|        0.9626|
|          5 | Shuttle                 |    0.9678|        0.9679|        0.9679|        **0.9688**|        0.9688|
|          6 | Annthyroid                 |    **0.9649**|        0.9626|        0.9627|        0.9626|        0.9626|
|          7 | Mammography                 |    0.9871 |        **0.9890**|        0.9876|        0.9871|        0.9874|

Analyzing the experimental results can be summarized as follows.

#### 1. If the number of samples in the dataset(Glass, Lympho) is small, it is useless to increase n_estimators.
#### 2. In the case of Shuttle dataset with the largest number of samples, there was a change in performance until the n_estimators reached 200.
#### 3. Even if the number of samples was enough, increasing n_estimators did not mean improving performance(Annthyroid, Mammography).

___
## Conclusion

#### 1) For anomaly detection, it is very important to determine the threshold that determines whether it is an outlier.
#### 2) In the case of Isolation Forest, the percentage of the outlier is received as an argument and the threshold is determined by itself. 
#### 3) The IF's overall performance was good. However, in some cases, the Auto-encoder, which arbitrarily set the threshold value, showed better performance than Isolation Forest.
#### 4) I think that research on thresholds suitable for datasets and methodologies used is an essential field.
___

### Reference

- Business Analytics, Korea university (IME-654) https://www.youtube.com/watch?v=vlkbVgdPXc4&t=1588s
- https://www.simplilearn.com/tutorials/machine-learning-tutorial/bagging-in-machine-learning
- https://regenerativetoday.com/simple-explanation-on-how-decision-tree-algorithm-makes-decisions/The-anomaly-detection-and-the-classification-learning-schemas_fig1_282309055
- 
- 
- 
- 
- 


|  Accuracy  | Dataset              |  K = 5 |   K = 10 |   K = 15 |   K = 20 |   K = 30 |   K = 50 |
|:----------:|:--------------------:|:------:|:--------:|:--------:|:--------:|:--------:|:--------:|
|          1 | Banking            |    0. |        0. |         0.|        0.8990 |        0.|        0.|
|          2 | Breast                 |    0. |        0.|        0. |        0. |        0.|        0. |
|          3 | Diabetes                 |    0. |        0. |        0. |        0. |        0. |        0. |
|          4 | Glass                 |    0. |        0. |        0. |        0. |        0. |        0. |
|          5 | PersonalLoan                 |    0. |        0. |        0. |        0. |        0. |        0. |
|          6 | Stellar                 |    0. |        0. |        0. |        0. |        0. |        0. |
|          7 | WineQuality                 |    0. |        0. |        0. |        0. |        0. |        0. |
