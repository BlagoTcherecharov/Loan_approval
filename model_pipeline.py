import pandas as pd
from models.nn_model import NnModel
from models.random_forest_model import RFModel
from models.boosting_model import BoostingModel
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class ModelPipeline:
    def __init__(self):
        self.df = None
        self.preprocessor = None

        self.X = self.Y = None

    def pipeline(self):
        self.__get_data()
        self.__preprocess_data()
        self.__model_training()

    def __model_training(self):
        rf_model = RFModel(self.preprocessor, self.X, self.Y)
        boost_model = BoostingModel(self.preprocessor, self.X, self.Y)
        nn_model = NnModel(self.preprocessor, self.X, self.Y)

        rf_model.random_forest_model()
        boost_model.boosting_model()
        nn_model.neural_network_model()

    def __preprocess_data(self):
        numeric_features = ['Age', 'Income', 'Credit_Score', 'Loan_Amount', 'Loan_Term']
        categorical_features = ['Employment_Status']

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                ("cat", OneHotEncoder(), categorical_features),
            ]
        )

        self.X = self.df[numeric_features + categorical_features]

    def __get_data(self):
        data = pd.read_csv("loan_approval_dataset.csv")

        # Drop duplicates and null rows just in case (Dataset should be cleaned already)
        self.df = data.drop_duplicates().dropna()

        self.Y = self.df['Loan_Approved']

if __name__ == "__main__":
    pipeline = ModelPipeline()

    pipeline.pipeline()