from args import Parser1
from Bagging.DecisionTree import Decision_Tree
from Bagging.RandomForest import Random_Forest
from Boosting.AdaBoost import AdaptiveBoosting
from Boosting.GBM import GradientBoosting
from Boosting.XGBoost import ExtremeGradientBoosting

def build_model():
    parser = Parser1()
    args = parser.parse_args()

    if args.method == 'DecisionTree':
        result = Decision_Tree(args)
    elif args.method == 'RandomForest':
        result = Random_Forest(args)
    elif args.method == 'Adaboost':
        result = AdaptiveBoosting(args)
    elif args.method == 'GBM':
        result = GradientBoosting(args)    
    else:
        result = ExtremeGradientBoosting(args)

    print(result)

    result.to_csv('./result/' + args.method + '_' + args.data_type)
    
    return result

def main():
    build_model()

if __name__ == '__main__':
    main()
