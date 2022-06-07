import sys

import numpy as np
import pandas as pd
from plotly import express as px
from plotly import figure_factory as ff


def plot_cat_resp_cont_predictors(df, predictor, response, f_name, bin_size=1):
    """
    Function plots conitnuous predictors with categorical response
    """
    # separate predictors based on response
    x_false = df[df[response] == False].loc[:, predictor]
    x_true = df[df[response] == True].loc[:, predictor]
    histogram_data = [x_false, x_true]
    labels = ["False", "True"]
    fig = ff.create_distplot(histogram_data, labels, bin_size=bin_size)

    # Overlay both histograms
    fig.update_layout(
        barmode="overlay",
        title="Revenue by " + predictor,
        xaxis_title=predictor,
        yaxis_title="Distribution",
    )
    # Reduce opacity to see both histograms
    fig.update_traces(opacity=0.75)

    fig.write_html(
        file="./output/" + f_name + ".html",
        include_plotlyjs="cdn",
    )
    # fig.show()


def main():

    # loading our data
    df = pd.read_csv("./dataset/online_shoppers_intention.csv")

    # # shape
    print(df.shape)

    # # looking at our data
    print(df.dtypes)

    # look at descriptive stats
    print(df.describe().T)
    print(df.describe(include=["object", "bool"]).T)

    # spit our variables according to type
    categorical_var = [
        "SpecialDay",
        "Month",
        "OperatingSystems",
        "Browser",
        "Region",
        "TrafficType",
        "VisitorType",
        "Revenue",
        "Weekend",
    ]

    continuous_var = df.loc[:, ~df.columns.isin(categorical_var)].columns

    print(df[continuous_var].skew())

    plot_cat_resp_cont_predictors(
        df, continuous_var[3], "Revenue", continuous_var[3], 30
    )

    df["info_duration_log"] = np.log(df[continuous_var[3]] + 1)

    plot_cat_resp_cont_predictors(
        df, "info_duration_log", "Revenue", "info_duration_log", 0.25
    )

    fig = px.histogram(df["Revenue"], x="Month")
    fig.show()


if __name__ == "__main__":
    sys.exit(main())
