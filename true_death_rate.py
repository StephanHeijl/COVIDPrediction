import pandas
import os
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import operator
import sys
from math import sqrt
import pprint
from sklearn.linear_model import *
from sklearn.ensemble import *
from sklearn.metrics import *
from sklearn.preprocessing import PolynomialFeatures
from sklearn.naive_bayes import *
from scipy.stats import pearsonr, spearmanr
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import *
from sklearn.gaussian_process import *
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

csv_files = [f for f in os.listdir("text") if f.endswith("csv")]
for csv in sorted(csv_files, key=lambda x: [int(i) for i in x.split(".", 1)[0].split("_")])[2:]:
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
        # print("Using data from date:",merged.columns[col])
        # print("y value obtained from date: ", merged.iloc[:, col + lookahead].name)
        # print("Predicting for date: ", merged.iloc[end - 1].name)
        y.append(merged.iloc[end - 1, col + lookahead])
    except:
        y_incomplete.append(merged.iloc[end, -1])

    current_day += 1

X_new = X[len(y):]
X = X[:len(y)]

if len(sys.argv) == 1:
    n_test = 6
else:
    n_test = int(sys.argv[1])

n_train = len(X) - n_test

print(n_test)

test_start_date = dates[-(n_test + lookahead + 1)]

if n_test > 0:
    X_train, y_train = X[:-n_test], y[:-n_test]
    X_test, y_test = X[-n_test:], y[-n_test:]
else:
    X_train, y_train = X, y
    X_test, y_test = None, None

# >>> Experiment with your own regressor here <<<
reg = MLPRegressor((500, 500, 500), alpha=0.0001, max_iter=1000, solver="adam", random_state=12)
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

pred_train_bar = ax.bar(
    np.arange(len(y_train)) + bar_width - 0.5,
    y_pred,
    bar_width,
    label="Prediction (train)",
    #color=pred_train_color
)
true_train_bar = ax.bar(
    np.arange(len(y_train)) + bar_width * 2 - 0.5,
    y_train,
    bar_width,
    label="True (train)",
    #color=pred_test_color
)

if X_test is not None:
    y_pred = reg.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(mae)

    # Find the error for the next day.
    nde = abs((y_pred[0] - y_test[0]) / y_test[0]) * 100

    print(np.array(y_test).astype(int))
    print(y_pred.astype(int))

maes = []
next_day_errs = []

error_files = [f for f in os.listdir("errors") if f.endswith(".txt")]

error_files.sort(key=lambda x: tuple([int(i) for i in x.strip(".tx").split("-")]))

for f in error_files:
    if test_start_date in f:
        break

    with open(os.path.join("errors", f)) as fh:
        err_values = fh.read().strip().split(",")
        maes.append(float(err_values[0]))
        next_day_errs.append(float(err_values[1]))

if X_test is not None:
    maes.append(mae)
    next_day_errs.append(nde)
else:
    mae = maes[-1]
    nde = next_day_errs[-1]

mae_plot = ax.plot(np.arange(len(X_train) - len(maes), len(X_train)) + 0.5, maes, label="Mean Absolute Error", color="red")
ax.text(len(X_train) - 0.5, mae + 5, s="%i" % mae, color="red")

ax_sec = ax.twinx()
ax_sec.grid(False)

percentage_off_plot = ax_sec.plot(
    np.arange(len(X_train) - len(next_day_errs), len(X_train)) + 0.5,
    next_day_errs,
    label="Error for the next day",
    color="blue"
)
ax_sec.text(len(X_train) - 0.5, nde - 5, s="%i%%" % nde, color="blue")

if X_test is not None:
    pred_test_bar = ax.bar(
        np.arange(len(y_train), len(y_train) + n_test) + bar_width - 0.5,
        y_pred,
        bar_width,
        label="Prediction (test)",
        #color=true_train_color,
        #yerr=mae,
        #ecolor="red",
        #capsize=2
    )
    true_test_bar = ax.bar(
        np.arange(len(y_train), len(y_train) + n_test) + bar_width * 2 - 0.5,
        y_test,
        bar_width,
        label="True (test)",
        #color=true_test_color
    )

future_pred = reg.predict(X_new)

incomplete_pred_bar = ax.bar(
    np.arange(n_test + len(y_train), len(y_train) + n_test + lookahead) + bar_width - 0.5,
    future_pred,
    bar_width,
    label="Prediction (data incomplete)",
    ecolor="red",
    yerr=mae,
    capsize=2
)

incomplete_data_bar = ax.bar(
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
    ax.text(
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

if X_test is not None:
    handles = [
        pred_train_bar, true_train_bar, pred_test_bar, true_test_bar,
        incomplete_pred_bar, incomplete_data_bar, mae_plot[0], percentage_off_plot[0],
    ]
else:
    handles = [
        pred_train_bar, true_train_bar,
        incomplete_pred_bar, incomplete_data_bar, mae_plot[0], percentage_off_plot[0],
    ]

labels = [
    h.get_label() for h in handles
]


dates = [""] + dates
dates[1] = "2020-3-28"
print(dates)
ax.set_xticklabels(dates, rotation=90)
print(len(dates))
print(len(X) + len(X_new))
ax.xaxis.set_major_locator(MultipleLocator(1))

ax.legend(handles, labels)
ax.set_xlim(-0.5, len(X) + len(X_new))
ax_sec.set_ylim(0, 100)
ax.set_ylim(0, 200)

ax.set_xlabel("Date")
ax.set_ylabel("Number of deaths")
ax_sec.set_ylabel("Percentage error")

plt.suptitle("Predicting true COVID-19 deaths in the Netherlands", y=0.98, fontsize=18)
plt.title("Using training data up to %s inclusively." % test_start_date, fontsize=12, y=1.01)

plt.tight_layout()
plt.subplots_adjust(top=0.9)

plt.savefig("plots/%s.png" % test_start_date)

if X_test is not None:
    with open("errors/%s.txt" % test_start_date, "w+") as f:
        f.write("%.4f,%.4f" % (mae, nde))
