import numpy as np
import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


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

    regression = LogisticRegression(n_jobs=-1)
    regression.fit(X=X, y=y)
    print('Logistic regression accuracy:')
    print(regression.score(X=X, y=y))


def decision_tree(dataframe):
    X, y = predictors_labels(dataframe)

    tree_classifier = DecisionTreeClassifier()
    tree_classifier.fit(X=X, y=y)
    print('Decision Tree accuracy:')
    print(tree_classifier.score(X=X, y=y))


def predictors_labels(dataframe):
    X = dataframe.iloc[:, :-1].values
    y = dataframe.iloc[:, -1:].values
    y = np.ravel(y)
    return X, y


def main():
    dataframe = read_dataframe("../../../resources/covtype/covtype.data.gz")
    logistic_regression(dataframe)
    decision_tree(dataframe)


if __name__ == "__main__":
    main()
