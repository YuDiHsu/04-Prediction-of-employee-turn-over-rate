import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# Common sklearn Model Helpers
from sklearn import metrics
# Libraries for data modelling
from sklearn.ensemble import RandomForestClassifier
# sklearn modules for performance metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split  # import 'train_test_split'


# import mitosheet
# mitosheet.sheet()

# load data 2
def _load_data():
    df = pd.read_excel('df_人員清單(清過).xlsx', index_col=False)
    total_column = df.columns.to_list()
    df_Final = df[total_column]
    row_index = df.index
    return df_Final, row_index


def _choose_features_target(df_Final):
    df_Final_copy = df_Final.copy()
    df_Final_copy = df_Final_copy[['平均工時(不含休息)', '工時中位數(不含休息)', '請假時數', '請假次數']]
    df_Final_copy = df_Final_copy.apply(lambda x: pd.to_numeric(x, errors='coerce'))
    target = df_Final['OnJob'].copy()
    return df_Final_copy, target


def _split_data_train_test(df_Final_copy, target, row_index):
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(df_Final_copy,
                                                                                     target, row_index,
                                                                                     test_size=0.2,
                                                                                     random_state=7,
                                                                                     stratify=target)
    return X_train, X_test, y_train, y_test, indices_train, indices_test


def _model_configuration():
    rf_classifier = RandomForestClassifier(class_weight="balanced",
                                           random_state=7)
    return rf_classifier


def _auo_ml_hyper_parameter_setting(rf_classifier):
    param_grid = {'n_estimators': [50, 75, 100],
                  'min_samples_split': [2, 4, 6],
                  'min_samples_leaf': [1, 2, 3],
                  'max_depth': [5, 10, 15]}

    grid_obj = GridSearchCV(rf_classifier,
                            n_jobs=3,
                            return_train_score=True,
                            param_grid=param_grid,
                            scoring='roc_auc',
                            cv=5)
    return grid_obj


def _model_fitting(grid_obj, X_train, y_train, X_test, y_test):
    grid_fit = grid_obj.fit(X_train, y_train)
    rf_opt = grid_fit.best_estimator_
    print('=' * 20)
    print("best params: " + str(grid_obj.best_estimator_))
    print("best params: " + str(grid_obj.best_params_))
    print('best score:', grid_obj.best_score_)
    print('=' * 20)
    print(
        'Accuracy of RandomForest Regression Classifier on test set: {:.2f}'.format(rf_opt.score(X_test, y_test) * 100))
    print('Accuracy of RandomForest Regression Classifier on train set: {:.2f}'.format(
        rf_opt.score(X_train, y_train) * 100))
    rf_opt.fit(X_train, y_train)
    print(classification_report(y_train, rf_opt.predict(X_train)))
    return rf_opt


def _feature_importance_plot(rf_opt, X_train):
    importances = rf_opt.feature_importances_
    indices = np.argsort(importances)[::-1]  # Sort feature importances in descending order
    names = [X_train.columns[i] for i in
             indices]  # Rearrange feature names so they match the sorted feature importances
    plt.figure(figsize=(15, 7))  # Create plot
    plt.title("Feature Importance")  # Create plot title
    plt.bar(range(X_train.shape[1]), importances[indices])  # Add bars
    plt.xticks(range(X_train.shape[1]), names, rotation=90)  # Add feature names as x-axis labels
    plt.show()  # Show plot


def _feature_importance(rf_opt, X_train, df_Final_copy):
    importances = rf_opt.feature_importances_
    df_param_coeff = pd.DataFrame(columns=['Feature', 'Coefficient'])
    column_length = len(df_Final_copy.columns)  # feature lengths
    for i in range(column_length):
        feat = X_train.columns[i]
        coeff = importances[i]
        df_param_coeff.loc[i] = (feat, coeff)
    df_param_coeff.sort_values(by='Coefficient', ascending=False, inplace=True)
    df_param_coeff = df_param_coeff.reset_index(drop=True)
    print(df_param_coeff.head(10))


def _confuse_matrix(y_train, X_train, rf_opt):
    cnf_matrix = metrics.confusion_matrix(y_train, rf_opt.predict(X_train))
    class_names = [0, 1]  # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    # create heatmap
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()  # Show plot


def _create_final_predict_excel(df_Final, rf_opt, X_train, X_test, indices_train, indices_test):
    df_Final.loc[:, 'preds'] = ''
    df_train = df_Final.iloc[indices_train, :]
    df_train.loc[:, 'preds'] = rf_opt.predict(X_train)
    df_test = df_Final.iloc[indices_test, :]
    df_test.loc[:, 'preds'] = rf_opt.predict(X_test)
    df_Final = pd.concat([df_train, df_test]).sort_index()

    # predicted train and test probability for leave(0) and on(1)
    df_Final.loc[:, 'preds_prob_leave'] = ''
    df_train = df_Final.iloc[indices_train, :]
    df_train.loc[:, 'preds_prob_leave'] = rf_opt.predict_proba(X_train)[:, 0]
    df_test = df_Final.iloc[indices_test, :]
    df_test.loc[:, 'preds_prob_leave'] = rf_opt.predict_proba(X_test)[:, 0]
    df_Final = pd.concat([df_train, df_test]).sort_index()
    df_Final.loc[:, 'preds_prob_On'] = ''
    df_train = df_Final.iloc[indices_train, :]
    df_train.loc[:, 'preds_prob_On'] = rf_opt.predict_proba(X_train)[:, 1]
    df_test = df_Final.iloc[indices_test, :]
    df_test.loc[:, 'preds_prob_On'] = rf_opt.predict_proba(X_test)[:, 1]
    df_Final = pd.concat([df_train, df_test]).sort_index()
    df_Final.to_excel('df.xlsx', index=False)


if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False
    df_Final, row_index = _load_data()

    df_Final_copy, target = _choose_features_target(df_Final)

    X_train, X_test, y_train, y_test, indices_train, indices_test = _split_data_train_test(df_Final_copy, target,
                                                                                           row_index)

    rf_classifier = _model_configuration()

    grid_obj = _auo_ml_hyper_parameter_setting(rf_classifier)

    rf_opt = _model_fitting(grid_obj, X_train, y_train, X_test, y_test)

    _feature_importance(rf_opt, X_train, df_Final_copy)

    _create_final_predict_excel(df_Final, rf_opt, X_train, X_test, indices_train, indices_test)
