import joblib
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

class NnModel:
    def __init__(self, preprocessor, X, Y):
        self.preprocessor = preprocessor
        self.X = X
        self.Y = Y

    def neural_network_model(self):
        # Declaring pipeline for model training (Model is weaker because of overfitting / Small dataset)
        clf = Pipeline(
            steps=[("preprocessor", self.preprocessor), ("mlp", MLPClassifier(max_iter=500, early_stopping=True))]
        )

        # Split into training and test samples
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state=0)

        # Train model
        clf.fit(X_train, y_train)

        # Save model
        joblib.dump(clf, "./model_files/neural_network.pkl")

        # Model performance metrics
        print("model score: %.3f" % clf.score(X_test, y_test))
        print(f"Confusion matrix: \n{confusion_matrix(y_test, clf.predict(X_test))}")
        print(classification_report(y_test, clf.predict(X_test), zero_division=0))
