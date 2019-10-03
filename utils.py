import featuretools as ft
import numpy as np
import pandas as pd
#from featuretools.primitives import (Day, Hour, Max, Mean, Min, Minute, Month, Skew, Std, Week, Weekday, Weekend)
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer


# set global random seed
np.random.seed(40)

pd.__version__ = '0.23.4'

####################
# Case Study Utils #
####################

def preview(df, n=5):
    """return n rows that have fewest number of nulls"""
    order = df.isnull().sum(axis=1).sort_values(kind='mergesort').head(n).index
    return df.loc[order]


def feature_importances(model, feature_names, n=10):
    importances = model.feature_importances_
    zipped = sorted(zip(feature_names, importances), key=lambda x: -x[1])
    for i, f in enumerate(zipped[:n]):
        print("%d: Feature: %s, %.3f" % (i + 1, f[0], f[1]))

"""
def get_train_test_fm(feature_matrix, percentage):
    nrows = feature_matrix.shape[0]
    head = int(nrows * percentage)
    tail = nrows - head
    X_train = feature_matrix.head(head)
    y_train = X_train['trip_duration']
    X_train = X_train.drop(['trip_duration'], axis=1)
    imp = Imputer()
    X_train = imp.fit_transform(X_train)
    X_test = feature_matrix.tail(tail)
    y_test = X_test['trip_duration']
    X_test = X_test.drop(['trip_duration'], axis=1)
    X_test = imp.transform(X_test)

    return (X_train, y_train, X_test, y_test)
"""

def load_uk_retail_data():
    item_purchases = pd.read_csv('Data/item_purchases.csv', encoding = 'unicode_escape')
    invoices = pd.read_csv('Data/invoices.csv', encoding = 'unicode_escape') 
    items = pd.read_csv('Data/items.csv', encoding = 'unicode_escape')
    customers = pd.read_csv('Data/customers.csv', encoding = 'unicode_escape')
    invoices['first_item_purchases_time'] = pd.to_datetime(
        invoices['first_item_purchases_time'], format="%m/%d/%y %H:%M")
    item_purchases['InvoiceDate'] = pd.to_datetime(
        item_purchases['InvoiceDate'], format="%m/%d/%y %H:%M")
    customers['first_invoices_time'] = pd.to_datetime(
        customers['first_invoices_time'], format="%m/%d/%y %H:%M")
    items['first_item_purchases_time'] = pd.to_datetime(
        items['first_item_purchases_time'], format="%m/%d/%y %H:%M")
    return item_purchases, invoices, items, customers


def find_training_examples(item_purchases, invoices, prediction_window,
                           training_window, lead, threshold):
    niter = 2  # hard coded number of cutoffs we will search starting with
    cutoff_time = pd.Timestamp("2011-05-01")  # hard coded start date
    label_times = pd.DataFrame()
    for k in range(1, niter):
        cutoff_time = cutoff_time + pd.Timedelta("45d")
        lt = make_label_times(item_purchases, invoices, cutoff_time,
                              prediction_window,
                              training_window, lead, threshold)
        label_times = label_times.append(lt)

    label_times = label_times.sort_values('cutoff_time')
    return label_times


def make_label_times(item_purchases, invoices, cutoff_time, prediction_window,
                     training_window, lead, threshold):
    data = item_purchases.merge(
        invoices)[["CustomerID", "InvoiceDate", "Quantity", "UnitPrice"]]
    data["amount"] = data["Quantity"] * data["UnitPrice"]

    prediction_window_start = cutoff_time
    prediction_window_end = cutoff_time + prediction_window
    cutoff_time = cutoff_time - lead
    t_start = cutoff_time - training_window

    training_data = data[(data["InvoiceDate"] <= cutoff_time) &
                         (data["InvoiceDate"] > t_start)]
    prediction_data = data[(data["InvoiceDate"] > prediction_window_start) & (
        data["InvoiceDate"] < prediction_window_end)]

    # get customers in training data
    label_times = pd.DataFrame()
    label_times["CustomerID"] = training_data["CustomerID"].dropna().unique()
    label_times["t_start"] = t_start
    label_times["cutoff_time"] = cutoff_time

    labels = prediction_data.groupby("CustomerID")[["amount"]].count()

    label_times = label_times.merge(
        labels, how="left", left_on="CustomerID", right_index=True)

    # if the amount is nan that means the customer made no purchases in
    # prediction window
    label_times["amount"] = label_times["amount"].fillna(0)
    label_times.rename(columns={"amount": "purchases>threshold"}, inplace=True)
    label_times['purchases>threshold'] = label_times['purchases>threshold'] \
        > threshold

    return label_times


def engineer_features_uk_retail(entities, relationships, label_times,
                                training_window):
    trans_primitives = [Minute, Hour, Day, Week, Month, Weekday, Weekend]

    es = ft.EntitySet("entityset",
                      entities=entities,
                      relationships=relationships)

    es.add_last_time_indexes()

    feature_matrix, features = ft.dfs(entityset=es,
                                      target_entity="customers",
                                      trans_primitives=trans_primitives,
                                      agg_primitives=[Mean, Max, Std],
                                      cutoff_time=label_times[[
                                          "CustomerID", "cutoff_time"]],
                                      training_window=training_window)
    feature_matrix.drop("Country", axis=1, inplace=True)
    feature_matrix = feature_matrix.sort_index()
    return feature_matrix
