import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, GridSearchCV
import timeit
import csv
import matplotlib.pyplot as plt

def convert_int_to_float(df):
    for i in range(df.shape[1]):
        col = df.columns[i]
        if df[col].dtypes == 'int64':
                df[col] = df[col].astype(float)


def return_appropriate_model_and_metric(df, col, grd_srch=False):
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import r2_score
    if (is_continuous(df, col)):
        if grd_srch:
            regr = RandomForestRegressor(random_state=0)
            parameters = {
                'n_estimators': [50, 100],
                'max_depth': [2, 16, 64],
                'min_samples_leaf': [1, 2, 4]
            }
            clf = GridSearchCV(regr, parameters)
            return clf, r2_score, True
        else:
            regr = RandomForestRegressor(random_state=0)
            return regr, r2_score, True
    else:
        if grd_srch:
            rfclass = RandomForestClassifier(random_state=0)
            parameters = {
                'n_estimators': [50, 100],
                'max_depth': [2, 16, 64],
                'min_samples_split': [2, 4, 6]
            }
            clf = GridSearchCV(rfclass, parameters)
            return clf, accuracy_score, False
        else:
            rfclass = RandomForestClassifier(random_state=0)

            #return rfclass, accuracy_score, False
            return rfclass, accuracy_score, False


def return_appropriate_model_and_metric_2(df, col, grd_srch=False):
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import r2_score
    mod = "Linear"
    if (is_continuous(df, col)):
        regr = LinearRegression()
        if not grd_srch:
            return regr, r2_score, True, mod
        parameters = {
            'normalize': [True, False]
        }
        clf = GridSearchCV(regr, parameters)
        return clf, r2_score, True, mod
    else:
        mod = "Logistic"
        lgr = LogisticRegression(random_state=0)
        if not grd_srch:
            return lgr, accuracy_score, False, mod
        parameters = {
            'penalty': ['l1', 'l2', 'elasticnet']
        }
        clf = GridSearchCV(lgr, parameters)
        
        return clf, accuracy_score, False, mod


def encode(df): #pass in index
    for i in range(df.shape[1]):
        col = df.columns[i]
        le = LabelEncoder()
        if not is_numerical(df, col):
            data = df[col]
            le.fit(data.dropna().values)
            df.iloc[df.index[df[col].notna()], i] = le.transform(df[col].dropna())


def reverse_label(df, col, output):
    if not is_numerical(df, col):
        le = LabelEncoder()
        data = df[col]
        le.fit(data.dropna().values)
        output = le.inverse_transform(output)
    return output


def add_missing_values(df, missing_col, sample_fraction=0.1):
    #inplace insertion of missing values
    ix = [(row, col) for row in range(df.shape[0])
      for col in missing_col]
    for row, col in random.sample(ix, int(round(sample_fraction * (df.shape[0]*len(missing_col))))):
        df.iat[row, col] = np.nan
    return df


def is_continuous(df, column_name):
    """
    Whether this column has only continuous numerical data.
    """
    return df[column_name].dtypes.kind in (np.typecodes["AllFloat"])


def is_numerical(df, column_name):
    """
    Whether this column has only numerical data, which can be either discrete
    or continuous.
    """
    return df[column_name].dtypes.kind in (
        np.typecodes["AllInteger"] + np.typecodes["AllFloat"])


def x_y_split_m(df, col):
    y_train = df.loc[:, col]
    temp = list(set(df.columns) - set(col))
    columns = [df.columns.get_loc(c) for c in temp]
    #print(columns)
    columns.sort()
    x_train = df.iloc[:, columns]
    return x_train, y_train


def get_train_test_data_m(df, org_df, col):  # col is column name in string
    df_train = df.dropna()
    x_train, y_train = x_y_split_m(df_train, col)

    # missing_indices = df.index[df[:,col].isna()]
    missing_indices, _ = np.where(pd.isnull(df))
    df_test = org_df.iloc[missing_indices]
    x_test, y_test = x_y_split_m(df_test, col)
    return x_train, y_train, x_test, y_test



"""
Input: 
    df: a pandas dataframe with missing values in one or more columns
    missing_is: a list of integer indices where each index is column with missing values

Output:
    df: input dataframe with missing values imputed by best imputation models
    summary: a dictionary of relevant imputation models and their imputation quality
"""


def impute_missing_values(df, missing_is):

    perms = list(itertools.permutations(missing_is))

    temp_sample = random.sample(perms, 1)

    best_order = []
    for i in temp_sample[0]:
        best_order.append(i)

    my_dict = {"best imputation order": best_order,
               "best imputation model": [],
               "mean imputation score": 0.0}

    df_copy = df.copy()
    encode(df_copy)
    df_no_na = df_copy.dropna()
    copy_df = df_no_na.copy()
    org_df = df_no_na.copy()
    convert_int_to_float(copy_df)
    convert_int_to_float(org_df)

    missing_cols = []
    for j in missing_is:
        missing_cols.append(df.columns[j])

    add_missing_values(copy_df, missing_is, 0.1)  # add missing values on multiple columns


    # encode(copy_df)  # encode all categorical/string columns to integer
    # encode(org_df)

    x_train, y_train, x_test, y_test = get_train_test_data_m(copy_df, org_df, missing_cols)


    # iterate over each missing column in the same order as the curent permutation order
    # train models iteratively
    # x_train, x_test will have incrementally more column,
    # and y_train y_test will be different each time
    train_is_set = set(range(df.shape[1])) - set(missing_is)

    x_test = x_test.reset_index(drop=True)

    # print("best order is", best_order)
    # print(len(best_order))

    dict_cols = {}
    for col in missing_cols:
        dict_cols[col] = True

    score_dict = {"Linear": 0, "Logistic": 0, "RandomForest": 0}
    time_dict = {"Linear": 0, "Logistic": 0, "RandomForest": 0}
    mod =''
    score_reg = []
    score_rf = []
    t_types = []
    for i in range(len(best_order)):

        val = best_order[i]

        #print("iteration ", i)
        col = df.columns[val]

        #accuracy scores by models
        

        # print(col)

        # get appropriate model and metric
        # 'model1' is either a random forest regressor or a random forest classifier
        model1, metric, is_regr = return_appropriate_model_and_metric(df, col)
        #print(model1)
        # model2 is either a linear regression or a logisti regression
        model2, metric2, is_regr2, mod = return_appropriate_model_and_metric_2(df, col)
        t_types.append(mod)

        y_train_s = y_train[col]
        #print(y_train_s)
        y_test_s = y_test[col]
        if not (is_numerical(df, col)):
            y_train_s = y_train_s.astype('int')
            y_test_s = y_test_s.astype('int')

        # print(model_with_gridsearch.best_estimator_)
        start = timeit.default_timer()
        model1.fit(x_train, y_train_s)
        y_pred = model1.predict(x_test)
        score = round(metric(y_test_s, y_pred), 4)
        score_rf.append(score)
        stop = timeit.default_timer()
        time_dict['RandomForest'] = stop-start
        
        start = timeit.default_timer()
        model2.fit(x_train, y_train_s)
        y_pred2 = model2.predict(x_test)
        score2 = round(metric(y_test_s, y_pred2), 4)
        score_reg.append(score2)
        stop = timeit.default_timer()
        time_dict[mod] = stop-start
        score_dict[mod] = score2

        
        #### KNN ####
        # knn = KNNImputer(n_neighbors=5)
        # knn.fit(x_train, y_train_s)
        # y_pred_knn = knn.predict(x_test)
        # score = round(metric(y_test_s, y_pred_knn), 4)
        # knn_score = {"KNN": score}
        # score_dict.update(knn_score)

        imputation = None
        unwanted_cols = [key for key in dict_cols]
        temp = list(set(df.columns) - set(unwanted_cols))
        columns = [df.columns.get_loc(c) for c in temp]
        columns.sort()

        df_copy = df.copy()
        encode(df_copy)
        x_impute = df_copy.iloc[df.index[df[col].isna()], columns]
        dict_cols.pop(col, None)

        if score >= score2:
            my_dict["best imputation model"].append(model1)
            my_dict["mean imputation score"] += score
            # print(df.index[df[col].isna()])
            # print(x_impute)

            imputation = model1.predict(x_impute)
            # print("impute length", len(x_impute))
            # print("missing", len(df.index[df[col].isna()]))
            inverse_label_imputation = reverse_label(df, col, imputation)
            df.iloc[df.index[df[col].isna()], val] = inverse_label_imputation
            # do in place imputation here
        else:
            my_dict["best imputation model"].append(model2)
            my_dict["mean imputation score"] += score2
            # print(df.index[df[col].isna()])
            # print(x_impute)
            imputation = model2.predict(x_impute)
            # print("impute length", len(x_impute))
            # print("missing", len(df.index[df[col].isna()]))
            inverse_label_imputation = reverse_label(df, col, imputation)
            df.iloc[df.index[df[col].isna()], val] = inverse_label_imputation
            # do in place imputation here

        x_train = pd.concat([x_train, y_train_s], axis=1)

        new_x_test_column = pd.DataFrame(y_pred, columns=[col])
        data = [x_test, new_x_test_column]
        x_test = pd.concat(data, axis=1)

        new_x_impute_column = pd.DataFrame(imputation, columns=[col])
        data_impute = [x_impute, new_x_impute_column]
        x_impute = pd.concat(data_impute, axis=1)

        train_is_set.add(val)
        indices = list(train_is_set)
        reorder_cols = [df.columns[j] for j in indices]

        x_train = x_train[reorder_cols]
        x_test = x_test[reorder_cols]
        x_impute = x_impute[reorder_cols]

    score_dict["RandomForest"] = sum(score_rf)/len(best_order)

    my_dict["mean imputation score"] = round(my_dict["mean imputation score"] / len(best_order), 4)


    return df, my_dict, score_dict, time_dict


# def linear(df, col, grd_srch=False):
#     from sklearn.metrics import accuracy_score
#     from sklearn.metrics import r2_score

#     regr = LinearRegression()
#     return regr, r2_score, True, mod
#     else:
#         mod = "Logistic"
#         lgr = LogisticRegression(random_state=0)
#         if not grd_srch:
#             return lgr, accuracy_score, False, mod
#         parameters = {
#             'penalty': ['l1', 'l2', 'elasticnet']
#         }
#         clf = GridSearchCV(lgr, parameters)
        
#         return clf, accuracy_score, False, mod

# def logistic(df, col, grd_srch=False):
#     from sklearn.metrics import accuracy_score
#     from sklearn.metrics import r2_score

#     lgr = LogisticRegression(random_state=0)
#     return lrg, accuracy_score, False