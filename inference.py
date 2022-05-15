import pandas as pd
import numpy as np
import pickle
from flask import Flask
from flask import request

app = Flask(__name__)
clf = pickle.load(open('churn_model.pkl', 'rb'))


# def check_preds(clf=None, X_test=None):
#     if X_test is None:
#         X_test = pd.read_csv('X_test.csv')
#     y_pred = np.loadtxt('preds.csv')
#     if clf is None:
#         clf = pickle.load(open('churn_model.pkl', 'rb'))
#     assert (clf.predict(X_test) == y_pred).all()


@app.route('/predict')
def predict():
    samp = np.array(
        [request.args.get('is_male', type=float), request.args.get('num_inters', type=float),
         request.args.get('late_on_payment', type=float),
         request.args.get('age', type=float), request.args.get('years_in_contract', type=float)]).reshape(1, -1)
    pred = clf.predict(samp)[0]
    return str(pred)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080,debug=True)
