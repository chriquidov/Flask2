import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


def main():
    df = pd.read_csv(r'C:\Users\chriq\PycharmProjects\Flask_ex\cellular_churn_greece.csv')

    X = df.loc[:, df.columns != "churned"]
    y = df.churned

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.7, random_state=42)

    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred))

    print(f'accuracy: {(y_pred == y_test).mean()}')

    pickle.dump(clf, open('churn_model.pkl', 'wb'))

    X_test.to_csv('X_test.csv', index=False)

    np.savetxt('preds.csv', y_pred)


if __name__ == '__main__':
    main()
