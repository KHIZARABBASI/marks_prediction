import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomeException
from src.logger import logging
from src.utils import save_object
import os


@dataclass
class DataTransformationConfig:
    preprocessor_file_ob_path = os.path.join("Arifact","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self ):
        try:
            numarical_column = ['reading_score', 'writing_score']
            categorical_column = ['gender',
                                'race_ethnicity',
                                'parental_level_of_education',
                                'lunch',
                                'test_preparation_course',
                                ]
            
            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy= "median")),
                    ("scaler", StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy = "most_frequent")),
                    ("encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean= False))
                ]
            )
            logging.info(f"Categorical columns: {categorical_column}")
            logging.info(f"Numerical columns: {numarical_column}")

            preprocessor = ColumnTransformer(
                [
                    ("numarical_pipeline", num_pipeline, numarical_column),
                    ("cat_pipeline", cat_pipeline, categorical_column)

                ]
            )

            return preprocessor
        
            
        except Exception as e:
            CustomeException(e, sys)


    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read Train and Test data Completed")

            logging.info("Obtaining Preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            logging.info("preprocessing object Assifneed")

            target_column_name = "math_score"
            numarical_column = ['reading_score', 'writing_score']

            input_feature_train_df = train_df.drop(columns= [target_column_name], axis= 1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_file_ob_path,
                obj=preprocessing_obj

            )

        

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_file_ob_path,
            )

        except Exception as e:
            raise CustomeException(e,sys)

