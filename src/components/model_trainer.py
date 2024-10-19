import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomeException
from src.logger import logging

from src.utils import save_object,model_evaluate


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("Arifact", "model.pkl")

class ModelTrainer:
    def __init__(self):
         self.model_trainer_config = ModelTrainerConfig()

    def initiate_modl_trainer(self, train_array, test_array):
        try:
            logging.info("split traning and test data")
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            logging.info("Obtaining Preprocessing object")

            preprocessing_obj = self.get

            logging.info("preprocessing object Assifneed")

            models= {

                "Random Forest" : RandomForestRegressor(),
                "Desigion Tree": DecisionTreeRegressor(),
                "Grediant Boosting": GradientBoostingRegressor(),
                "Linear Regrssion" : LinearRegression(),
                "KNighbour Classifier": KNeighborsRegressor(),
                "XGBclassifier": XGBRegressor(),
                "CatBoosting Classifier" : CatBoostRegressor(),
                "AdaBoost Classifier": AdaBoostRegressor()
            }

            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report: dict= model_evaluate(x_train= x_train, y_train= y_train , x_test= x_test, y_test= y_test, models = models,param=params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            logging.info("best model found in both traing and test data")
            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj= best_model
            )
            predicted = best_model.predict(x_test)

            r2_squire = r2_score(y_test, predicted)
            return r2_squire


        except Exception as e:
            raise CustomeException(e,sys)


