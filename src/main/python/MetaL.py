import numpy as np
import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from time import time


def read_dataframe(filename):
    col_names = ["Elevation", "Aspect", "Slope",
                 "Horizontal_Distance_To_Hydrology",
                 "Vertical_Distance_To_Hydrology",
                 "Horizontal_Distance_To_Roadways",
                 "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
                 "Horizontal_Distance_To_Fire_Points"]
    col_names.extend("WA_{}".format(i) for i in range(4))
    col_names.extend("ST_{}".format(i) for i in range(40))
    col_names.append("Cover_Type")
    dataframe = pandas.read_csv(filename, names=col_names)
    return dataframe


def logistic_regression(dataframe):
    X, y = predictors_labels(dataframe)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=0)

    regression = LogisticRegression(n_jobs=-1)
    regression.fit(X=X_train, y=y_train)
    print('Logistic regression accuracy:')
    print(regression.score(X=X_test, y=y_test))


def decision_tree(dataframe):
    X, y = predictors_labels(dataframe)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=0)

    tree_classifier = DecisionTreeClassifier()
    tree_classifier.fit(X=X_train, y=y_train)
    print('Decision Tree accuracy:')
    print(tree_classifier.score(X=X_test, y=y_test))


def predictors_labels(dataframe):
    X = dataframe.iloc[:, :-1].values
    y = dataframe.iloc[:, -1:].values
    y = np.ravel(y)
    return X, y


def timeit(function):
    start_time = time()
    function()
    end_time = time()
    print('Execution time %d' % (end_time - start_time))


def main():
    dataframe = read_dataframe("../../../resources/covtype/covtype.data.gz")
    foo = lambda: decision_tree(dataframe)

    timeit('logistic_regression(dataframe)', number=1)
    # timeit('decision_tree(dataframe', number=1)


if __name__ == "__main__":
    main()
