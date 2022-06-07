from locale import normalize

import numpy as np
import pandas as pd
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objects as go


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


# loading our data
df = pd.read_csv("./dataset/online_shoppers_intention.csv")

# shape
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

# view skewness of data
print(df[continuous_var].skew())

# plot info duration
plot_cat_resp_cont_predictors(df, continuous_var[3], "Revenue", continuous_var[3], 30)

# log transform and plot info duration
df["info_duration_log"] = np.log(df[continuous_var[3]] + 1)
plot_cat_resp_cont_predictors(
    df, "info_duration_log", "Revenue", "info_duration_log", 0.25
)

# plot months
months = df.Month.value_counts().index
fig = go.Figure(
    data=[
        go.Bar(
            name="False",
            x=months,
            y=df[df["Revenue"] == False].Month.value_counts(normalize=True).to_list(),
        ),
        go.Bar(
            name="True",
            x=months,
            y=df[df["Revenue"] == True].Month.value_counts(normalize=True).to_list(),
        ),
    ]
)

# Change the bar mode
fig.update_layout(
    barmode="stack",
    title="Revenue by Month",
    xaxis_title="Month",
    yaxis_title="Distribution",
)

fig.write_html(
    file="./output/" + "Revenue_by_month" + ".html",
    include_plotlyjs="cdn",
)
# fig.show()

# plot region
Region = df.Region.value_counts().index
fig = go.Figure(
    data=[
        go.Bar(
            name="False",
            x=Region,
            y=df[df["Revenue"] == False].Region.value_counts(normalize=True).to_list(),
        ),
        go.Bar(
            name="True",
            x=Region,
            y=df[df["Revenue"] == True].Region.value_counts(normalize=True).to_list(),
        ),
    ]
)

# Change the bar mode
fig.update_layout(
    barmode="stack",
    title="Revenue by Region",
    xaxis_title="Region",
    yaxis_title="Count",
)

fig.write_html(
    file="./output/" + "Revenue_by_Region" + ".html",
    include_plotlyjs="cdn",
)
fig.show()
