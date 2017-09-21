import numpy as np
import pandas
from sklearn.linear_model import LogisticRegression


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


def do_logistic_regression(dataframe):
    X = dataframe.iloc[:, :-1].values
    y = dataframe.iloc[:, -1:].values
    y = np.ravel(y)

    regression = LogisticRegression(n_jobs=-1)
    # regression.set_params(dataframe.columns.tolist())
    regression.fit(X=X, y=y)
    print(regression.score(X=X, y=y))


def main():
    dataframe = read_dataframe("../../../resources/covtype/covtype.data.gz")
    do_logistic_regression(dataframe)


if __name__ == "__main__":
    main()
