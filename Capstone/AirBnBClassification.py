import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn import grid_search

from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

####################################################
# Functions                                        #
####################################################
# Remove outliers
def removeOutliers(df, column, min_val, max_val):
    col_values = df[column].values
    df[column] = np.where(np.logical_or(col_values <= min_val, col_values >= max_val), np.NaN, col_values)
    return df


# Count occurrences of value in a column
def converttocounts(df, id_col, column_to_convert):
    id_list = df[id_col].drop_duplicates()

    df_counts = df.loc[:, [id_col, column_to_convert]]
    df_counts['count'] = 1
    df_counts = df_counts.groupby(by=[id_col, column_to_convert], as_index=False, sort=False).sum()

    new_df = df_counts.pivot(index=id_col, columns=column_to_convert, values='count')
    new_df = new_df.fillna(0)

    # Rename Columns
    categories = list(df[column_to_convert].drop_duplicates())
    for category in categories:
        cat_name = str(category).replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("-",
                                                                                                               "").lower()
        col_name = column_to_convert + '_' + cat_name
        new_df.rename(columns={category: col_name}, inplace=True)

    return new_df


####################################################
# Data Cleansing                                   #
####################################################
def readinputfile():
    # Import data
    print("Reading in data...")
    filepath = "./dataset/train_users_2.csv"
    df_train = pd.read_csv(filepath, header=0, index_col=None)
    print ("Number of features and rows is ", df_train.shape)
    return df_train


def preprocessing(df_train):
    # Change Dates to consistent format

    print("Fixing timestamps...")
    df_train['date_account_created'] = pd.to_datetime(df_train['date_account_created'], format='%Y-%m-%d')
    df_train['timestamp_first_active'] = pd.to_datetime(df_train['timestamp_first_active'], format='%Y%m%d%H%M%S')
    df_train['date_account_created'].fillna(df_train.timestamp_first_active, inplace=True)

    # Remove date_first_booking column
    df_train.drop('date_first_booking', axis=1, inplace=True)

    # Fixing age column
    print("Fixing age column...")
    df_train = removeOutliers(df=df_train, column='age', min_val=15, max_val=85)
    df_train['age'].fillna(df_train['age'].mean(), inplace=True)

    # Fill first_affiliate_tracked column
    print("Filling first_affiliate_tracked column...")
    df_train['first_affiliate_tracked'].fillna(-1, inplace=True)
    print ("Number of features and rows is ", df_train.shape)
    return df_train


####################################################
# Data Normalization                               #
####################################################
# One Hot Encoding


def normalisedata(df_train):
    print("One Hot Encoding categorical data...")
    columns_to_convert = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel',
                          'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type',
                          'first_browser']

    for column in columns_to_convert:
        new_df = pd.get_dummies(df_train[column], prefix=column)
        df_train[new_df.columns] = new_df
        df_train.drop(column, axis=1, inplace=True)

    ####################################################
    # Feature Extraction                               #
    ####################################################
    # Add new date related fields
    print("Adding new fields...")
    df_train['day_account_created'] = df_train['date_account_created'].dt.weekday
    df_train['month_account_created'] = df_train['date_account_created'].dt.month
    df_train['quarter_account_created'] = df_train['date_account_created'].dt.quarter
    df_train['year_account_created'] = df_train['date_account_created'].dt.year
    df_train['hour_first_active'] = df_train['timestamp_first_active'].dt.hour
    df_train['day_first_active'] = df_train['timestamp_first_active'].dt.weekday
    df_train['month_first_active'] = df_train['timestamp_first_active'].dt.month
    df_train['quarter_first_active'] = df_train['timestamp_first_active'].dt.quarter
    df_train['year_first_active'] = df_train['timestamp_first_active'].dt.year
    df_train['created_less_active'] = (df_train['date_account_created'] - df_train['timestamp_first_active']).dt.days

    # Drop unnecessary columns
    columns_to_drop = ['date_account_created', 'timestamp_first_active']
    for column in columns_to_drop:
        if column in df_train.columns:
            df_train.drop(column, axis=1, inplace=True)

    ####################################################
    # Add data from sessions.csv                       #
    ####################################################
    # Import sessions data
    s_filepath = "./dataset/sessions.csv"
    sessions = pd.read_csv(s_filepath, header=0, index_col=False)

    # Determine main device
    print("Determing main device...")
    sessions_device = sessions.loc[:, ['user_id', 'device_type', 'secs_elapsed']]
    aggregated_lvl1 = sessions_device.groupby(['user_id', 'device_type'], as_index=False, sort=False).aggregate(np.sum)
    idx = aggregated_lvl1.groupby(['user_id'], sort=False)['secs_elapsed'].transform(max) == aggregated_lvl1[
        'secs_elapsed']
    df_main = pd.DataFrame(aggregated_lvl1.loc[idx, ['user_id', 'device_type', 'secs_elapsed']])
    df_main.rename(columns={'device_type': 'main_device', 'secs_elapsed': 'main_secs'}, inplace=True)
    new_df = pd.get_dummies(df_main['main_device'], prefix='main_device')
    df_main.drop('main_device', axis=1, inplace=True)
    df_main[new_df.columns] = new_df

    # Determine backup device
    print("Determing backup device...")
    remaining = aggregated_lvl1.drop(aggregated_lvl1.index[idx])
    idx = remaining.groupby(['user_id'], sort=False)['secs_elapsed'].transform(max) == remaining['secs_elapsed']
    df_backup = pd.DataFrame(remaining.loc[idx, ['user_id', 'device_type', 'secs_elapsed']])
    df_backup.rename(columns={'device_type': 'backup_device', 'secs_elapsed': 'backup_secs'}, inplace=True)
    new_df = pd.get_dummies(df_backup['backup_device'], prefix='backup_device')
    df_backup.drop('backup_device', axis=1, inplace=True)
    df_backup[new_df.columns] = new_df

    # Aggregate and combine actions taken columns
    print("Aggregating actions taken...")
    session_actions = sessions.loc[:, ['user_id', 'action', 'action_type', 'action_detail']]
    columns_to_convert = ['action', 'action_type', 'action_detail']
    session_actions = session_actions.fillna('not provided')
    first = True

    for column in columns_to_convert:
        print("Converting " + column + " column...")
        current_data = converttocounts(df=session_actions, id_col='user_id', column_to_convert=column)

        # If first loop, current data becomes existing data, otherwise merge existing and current
        if first:
            first = False
            actions_data = current_data
        else:
            actions_data = pd.concat([actions_data, current_data], axis=1, join='inner')

    # Merge device datasets
    print("Combining results...")
    df_main.set_index('user_id', inplace=True)
    df_backup.set_index('user_id', inplace=True)
    device_data = pd.concat([df_main, df_backup], axis=1, join="outer")

    # Merge device and actions datasets
    combined_results = pd.concat([device_data, actions_data], axis=1, join='outer')
    df_sessions = combined_results.fillna(0)

    # Merge user and session datasets
    df_new = df_train.set_index('id', inplace=False)
    df_train = pd.concat([df_new, df_sessions], axis=1, join='inner')
    return df_train


def trainandtest(df_train):
    labels = df_train['country_destination']
    labels.reset_index()
    labels.rename_axis(None)
    X = df_train.drop('country_destination', axis=1, inplace=False)

    # Split training and testing data in the ratio of 80-20
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.20, random_state=40)

    # Training model
    print("Training model...")

    # Grid Search - Used to find best combination of parameters
    XGB_model = xgb.XGBClassifier(objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0)
    param_grid = {'max_depth': [3, 10], 'learning_rate': [0.1, 0.3], 'n_estimators': [50, 100]}
    model = grid_search.GridSearchCV(estimator=XGB_model, param_grid=param_grid, scoring='accuracy', verbose=10, n_jobs=1, iid=True, refit=True, cv=3)

    model.fit(X_train, y_train)
    print("Best score: %0.3f" % model.best_score_)

    ####################################################
    # Make predictions                                 #
    ####################################################
    print("Making predictions...")
    clf = model.best_estimator_

    y_pred = clf.predict(X_test)
    print (accuracy_score(y_test, y_pred))
    print (f1_score(y_test, y_pred, average='micro'))
    print (classification_report(y_test,y_pred))


    # Use decision tree classifier
    clf = DecisionTreeClassifier()
    print ("Training the decision tree using training data")
    param_grid ={'max_depth': [3, 10]}
    model = grid_search.GridSearchCV(estimator=clf, param_grid=param_grid, scoring='accuracy', cv=10)
    model.fit(X_train, y_train)
    clf = model.best_estimator_

    y_pred = clf.predict(X_test)
    print ("Decision Tree prediction results::")
    print (accuracy_score(y_test, y_pred))
    print (f1_score(y_test, y_pred, average='micro'))
    print (classification_report(y_test,y_pred))

    # Use KNN with neighbour set to 2
    clf = KNeighborsClassifier(n_neighbors=2)
    param_grid = {'n_neighbors': [5, 10]}
    model = grid_search.GridSearchCV(estimator=clf, param_grid=param_grid, scoring='accuracy', cv=10)
    model.fit(X_train, y_train)
    clf = model.best_estimator_
    print ("Predicting the results using KNN")
    y_pred = clf.predict(X_test)
    print (accuracy_score(y_test, y_pred))
    print (classification_report(y_test, y_pred))

    # Use basic versoin of MLP (Neural) classifier
    clf = MLPClassifier()
    print ("Training the MLPClassifier training data")
    param_grid = {'hidden_layer_sizes': [50, 100], 'solver': ['sgd','adam'], 'max_iter':[100,200]}
    model = grid_search.GridSearchCV(estimator=clf, param_grid=param_grid, scoring='accuracy', cv=10)
    model.fit(X_train, y_train)
    clf = model.best_estimator_
    print ("Predicting the results using MLPClassifier")
    y_pred = clf.predict(X_test)
    print (accuracy_score(y_test, y_pred))
    print (classification_report(y_test, y_pred))

    # Use Gaussian classifier
    clf = GaussianNB()
    print ("Training the GaussianNB training data")
    clf.fit(X_train, y_train)
    print ("Predicting the results using GaussianNB")
    y_pred = clf.predict(X_test)
    print (accuracy_score(y_test, y_pred))
    print (classification_report(y_test, y_pred))


if __name__ == '__main__':
    df_train = readinputfile()
    df_train = preprocessing(df_train)
    df_train = normalisedata(df_train)
    df_train = trainandtest(df_train)
