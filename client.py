import numpy as np
import requests
import pandas as pd
import pickle


def check_preds(clf=None, X_test=None):
    if X_test is None:
        X_test = pd.read_csv('X_test.csv')
    y_pred = np.loadtxt('preds.csv')
    if clf is None:
        clf = pickle.load(open('churn_model.pkl', 'rb'))
    assert (clf.predict(X_test) == y_pred).all()

def main():
    url = 'http://ec2-35-156-238-216.eu-central-1.compute.amazonaws.com:8080/predict'
    X_test = pd.read_csv('X_test.csv')
    y_pred = np.loadtxt('preds.csv')

    samps = X_test.iloc[:5]

    for i in range(5):
        params = samps.iloc[i].to_dict()

        r = requests.get(url, params=params)
        res = int(r.content)
        if res == y_pred[i]:
            print(f'Sample {i} correctly predicted')
        else:
            print(f'{y_pred} is different from the server prediction {res} ')


if __name__ == '__main__':
    check_preds()
    main()
