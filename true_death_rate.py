import pandas
import os
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import operator
import sys
import pprint
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import *
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import pearsonr, spearmanr
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from nice_data import get_intake_data
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

plt.style.use('Solarize_Light2')
plt.rcParams["font.family"] = "CMU Serif"
plt.rcParams["font.size"] = 12
plt.rcParams["font.weight"] = 700

dfs = []
names = []

nice_data = get_intake_data()

DAYS = ["mo", "tu", "we", "th", "fr", "sa", "su"]
HOLIDAYS = [
    "2020-4-12",  # Eerste paasdag
    "2020-4-13",  # Tweede paasdag
    "2020-4-12",  # Koningsdag
    "2020-5-4",  # Nationale Dodenherdenking  -- defacto holiday
    "2020-5-5",  # Bevrijdingsdag
    "2020-5-21",  # Hemelvaartsdag
    "2020-5-31",  # 1e Pinksterdag
    "2020-6-1",  # 2e Pinksterdag
    # TODO: Add more holidays
]

for csv in sorted(os.listdir("text"), key=lambda x: [int(i) for i in x.split(".", 1)[0].split("_")])[2:]:
    csv_key = [int(i) for i in csv.split(".", 1)[0].split("_")]
    csv_key[1] += 1
    names.append("%i-%i-%i" % tuple(csv_key))
    path = os.path.join("text", csv)
    dfs.append(
        pandas.read_csv(path, index_col=0)
    )

start_day = 0  # The index corresponding to the day where the data starts
lookahead = 5  # The number of days to look ahead to find the *true* number of deaths
merged = pandas.concat(dfs, axis=1, sort=True)
merged.columns = names

idx = [str(i).startswith("2020") for i in merged.index]
merged = merged.loc[idx]

n_days = 10  # Number of days to incorporate in the prediction data
n_cols = merged.shape[1]
n_meta_cols = 8  # 6 days indicaters + 2 holiday bools
n_nice_cols = n_days  # Number of days of NICE data
X = np.zeros((n_cols, n_days + n_meta_cols + n_nice_cols))
y = []
y_incomplete = []

dates = []

current_day = start_day
for col in range(n_cols):
    end = -n_cols + col - 1
    dates.append(merged.columns[col - 1])
    start = end - n_days
    X[col, :n_days] = merged.iloc[start:end, col]

    # Add the current day as a one hot encoded feature. Only 6 options are here to prevent colinearity.
    day_i = (current_day % 7)
    if day_i < 6:
        X[col, n_days + day_i] = 1

    # Add historical data from nice
    lookback = 10
    X[col, n_days + n_meta_cols:] = nice_data.loc[merged.iloc[start - lookback:end - lookback, col].index].newIntake

    # Add if yesterday or the day before was a free day (relative to the day being evaluated)
    try:
        if DAYS[(current_day - 2) % 7] in ["sa", "su"] or dates[col - 1] in HOLIDAYS:
            #print("One day before was a free day:", DAYS[(current_day - 1) % 7], dates[col - 1], col, dates[col])
            X[col, n_days + n_meta_cols - 2] = 1
            pass
    except IndexError:
        pass
    try:
        if DAYS[(current_day - 3) % 7] in ["sa", "su"] or dates[col - 2] in HOLIDAYS:
            #print("Two days before was a free day:", DAYS[(current_day - 2) % 7], dates[col - 2], col, dates[col])
            X[col, n_days + n_meta_cols - 1] = 1
            pass
    except IndexError:
        pass
    try:
        print("Using data from date:",merged.columns[col])
        print("y value obtained from date: ", merged.iloc[:, col + lookahead].name)
        print("Predicting for date: ", merged.iloc[end - 1].name)
        y.append(merged.iloc[end - 1, col + lookahead])
    except:
        y_incomplete.append(merged.iloc[end, -1])

    current_day += 1
    print("*" * 10)

X_new = X[len(y):]
X = X[:len(y)]

if len(sys.argv) == 1:
    n_test = 6 
else:
    n_test = int(sys.argv[1])

n_train = len(X) - n_test

print(n_test)

test_start_date = dates[-(n_test + lookahead + 1)]

X_train, y_train = X[:-n_test], y[:-n_test]
X_test, y_test = X[-n_test:], y[-n_test:]

print(X.shape, X_train.shape, X_test.shape)

reg = MLPRegressor((250, 250), alpha=0.0001, max_iter=1000, solver="adam", random_state=12)
reg.fit(X_train, y_train)

feature_names = ["deaths_day_n-%i" % (i + 1) for i in range(0, n_days)] +\
                DAYS[:-1] + \
                ["free_day-1", "free_day-2"] + \
                ["nice_n-%i" % (i + 1) for i in range(0, n_days)]

# pprint.pprint(list(zip(
#     feature_names,
#     ((reg.coef_ / reg.coef_.max()) * 100).astype("int")
# )))

bar_width = 0.4

y_pred = reg.predict(X_train)
print(mean_absolute_error(y_train, y_pred))

fig, ax = plt.subplots()
fig.set_size_inches(14, 8)

# COLORS
pred_train_color = "#B38E54"
pred_test_color = "#E58475"
pred_future_color = "#5EFCFF"
true_train_color = "#FFC66B"
true_test_color = "#FF6952"

plt.bar(
    np.arange(len(y_train)) + bar_width - 0.5,
    y_pred,
    bar_width,
    label="Prediction (train)",
    #color=pred_train_color
)
plt.bar(
    np.arange(len(y_train)) + bar_width * 2 - 0.5,
    y_train,
    bar_width, 
    label="True (train)",
    #color=pred_test_color
)

y_pred = reg.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print(np.array(y_test).astype(int))
print(y_pred.astype(int))

maes = []
error_files = os.listdir("errors")
error_files.sort(key=lambda x: tuple([int(i) for i in x.strip(".tx").split("-")]))

for f in error_files:
    if test_start_date in f:
        break

    with open(os.path.join("errors", f)) as fh:
        maes.append(float(fh.read().strip()))
maes.append(mae)

plt.plot(np.arange(len(X_train) - len(maes), len(X_train)) + 0.5, maes, label="Mean Absolute Error", color="red")
plt.text(len(X_train) - 0.5, mae + 5, s="%i" % mae, color="red")

plt.bar(
    np.arange(len(y_train), len(y_train) + n_test) + bar_width - 0.5,
    y_pred,
    bar_width,
    label="Prediction (test)",
    #color=true_train_color,
    #yerr=mae,
    #ecolor="red",
    #capsize=2
)
plt.bar(
    np.arange(len(y_train), len(y_train) + n_test) + bar_width * 2 - 0.5,
    y_test,
    bar_width,
    label="True (test)",
    #color=true_test_color
)

future_pred = reg.predict(X_new)

plt.bar(
    np.arange(n_test + len(y_train), len(y_train) + n_test + lookahead) + bar_width - 0.5,
    future_pred,
    bar_width,
    label="Prediction (data incomplete)",
    ecolor="red",
    yerr=mae,
    capsize=2
)

plt.bar(
    np.arange(n_test + len(y_train), len(y_train) + n_test + lookahead) + bar_width * 2 - 0.5,
    y_incomplete,
    bar_width,
    label="Prediction (data incomplete)",
    color="#c94c1c",
    alpha=0.5
    #color=pred_future_color,
    #yerr=mae,
    #ecolor="gold",
    #capsize=2
)

future_x = np.arange(n_test + len(y_train), len(y_train) + n_test + lookahead) + bar_width - 0.5
incomplete_x = np.arange(n_test + len(y_train), len(y_train) + n_test + lookahead) + bar_width * 2 - 0.5

for i, pred in enumerate(future_pred):
    plt.text(
        future_x[i] - 0.25, pred + 3 + mae, s="%i" % pred, color="#c94c1c"
    )
    # plt.arrow(
    #     dx=future_x[i] - incomplete_x[i] + bar_width / 2,
    #     dy=future_pred[i] - y_incomplete[i] ,
    #     x=incomplete_x[i],
    #     y=y_incomplete[i],
    #     color="red",
    #     length_includes_head=True,
    #     head_width=0.2,
    #     head_length=2,
    # )

dates = [""] + dates
dates[1] = "2020-3-28"
print(dates)
ax.set_xticklabels(dates)
print(len(dates))
print(len(X) + len(X_new))
ax.xaxis.set_major_locator(MultipleLocator(1))
plt.xticks(rotation=90)
plt.legend()
plt.xlim(-0.5, len(X) + len(X_new))
plt.ylim(0, 200)

plt.suptitle("Predicting true COVID-19 deaths in the Netherlands", y=0.98, fontsize=18)
plt.title("Using training data up to %s inclusively." % test_start_date, fontsize=12, y=1.01)

plt.tight_layout()
plt.subplots_adjust(top=0.9)

plt.savefig("plots/%s.png" % test_start_date)

with open("errors/%s.txt" % test_start_date, "w+") as f:
    f.write("%s" % mae)
