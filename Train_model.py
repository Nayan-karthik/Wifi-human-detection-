import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import joblib
from preprocess import preprocess_data

def train():
    data = preprocess_data("data/raw/sample_csi.csv")

    X = data[['filtered']]
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = MLPClassifier(hidden_layer_sizes=(10,10), max_iter=500)
    model.fit(X_train, y_train)

    joblib.dump(model, "models/model.pkl")

    print("Model trained and saved!")

if __name__ == "__main__":
    train()
