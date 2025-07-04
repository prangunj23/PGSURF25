import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer

DOTA2 = pd.read_csv('../../cmb_prosocial_labeled.csv')
EMBEDDINGS_MODEL = SentenceTransformer('all-mpnet-base-v2')
C = 10
DEG = 2
GAMMA = 'scale'
KERNEL = 'rbf'


def preprocessing(df=DOTA2):
    df_unclear_removed = df[df['action'] != 'UNCLEAR']
    df_unclear_removed['action'] = df['action'].replace({'NOT-PROSOCIAL': -1, 'PROSOCIAL': 1})
    return df

def train(X, y, split):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split)
    model = svm.SVC(kernel=KERNEL)
    model.fit(X_train, y_train)

    return model

def eval(model, X, y):
    y_pred = model.predict(X)
    return y_pred

