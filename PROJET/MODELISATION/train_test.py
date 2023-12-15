import pandas as pd
from sklearn.model_selection import train_test_split

def train_test(data):
    X = data.drop(columns=["salaire_1"])
    y = data["salaire_1"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0,stratify=y)