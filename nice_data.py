import requests
import pandas
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from sklearn.linear_model import *
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import PolynomialFeatures


def get_intake_data():
    url = "https://www.stichting-nice.nl/covid-19/public/new-intake/"
    data = requests.get(url).json()[0]
    data = pandas.DataFrame.from_dict(data).loc[:, ["date", "value"]]
    data = data.set_index("date")
    data.columns = ["newIntake"]
    return data


def get_ic_data():
    url = "https://www.stichting-nice.nl/covid-19/public/ic-count/"
    data = requests.get(url).json()
    return data


def get_idx_since(date, df):
    v = False
    since_idx = []
    for idx in df.index:
        if idx == date:
            v = True
        since_idx.append(v)
    return since_idx


if __name__ == "__main__":
    df = get_intake_data()
    #print(df)
    peak = "2020-03-30"

    data_since_peak = df.loc[get_idx_since(peak, df), "newIntake"].iloc[:-2]

    regr = Lasso()
    pf = PolynomialFeatures(1)
    y = data_since_peak
    print(y)
    print(np.arange(len(data_since_peak)))
    X = pf.fit_transform(np.arange(len(data_since_peak)).reshape(-1, 1), data_since_peak)
    regr.fit(X, y)
    X = np.arange(len(data_since_peak)).reshape(-1, 1)
    X = pf.transform(X)
    pred = regr.predict(X).tolist()

    df.plot(kind="bar")
    plt.plot(((len(df) - len(data_since_peak)) * [None]) + pred, c="orange")
    plt.xlim(0, len(df) + 5)
    plt.ylim(0, None)
    plt.show()