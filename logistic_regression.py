import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
# Common sklearn Model Helpers
from sklearn import model_selection
# Libraries for data modelling
from sklearn.linear_model import LogisticRegression
# sklearn modules for performance metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split  # import 'train_test_split'
import datetime


def _serve_days(x):
    today = datetime.datetime.today().date()
    if x['QuitDate']:
        x['ServeDays'] = (x['QuitDate'] - x['TCIJoinDate']).days
    else:
        x['ServeDays'] = (today - x['TCIJoinDate']).days
    return x


def _calculate_threshold(x):
    if float(x['中位總時數']) >= float(x['實際工時']):
        x["超過每人中位數閥值"] = 1
    else:
        x["超過每人中位數閥值"] = 0
    return x


# # 讀上下班打卡的資料
def _loading_PunchInOut_data():
    # df_merge = pd.DataFrame()
    # for i in range(1, 10):
    #     temp = pd.read_excel(os.path.join('.', 'Imported_data', "大江生醫_20210101-20210930 上下班時間.xlsx"),
    #                          sheet_name=f"0{i}", index_col=False, usecols=['英文名', '實際工時', '考勤狀態'])
    #     df_merge = df_merge.append(temp)
    #
    # df_merge = df_merge.reset_index(drop=True)
    #
    # df_merge.to_excel('df_merge.xlsx', index=False)
    df_merge = pd.read_excel(os.path.join('.', 'df_merge.xlsx'), index_col=False)
    work_day = 174  # # day
    df_merge['英文名'] = df_merge['英文名'].astype(str)
    df_merge.loc[df_merge['考勤狀態'] == '出差', '實際工時'] = 8.5
    df_all_work_time = df_merge.groupby('英文名')['實際工時'].sum().reset_index()

    # # 請假次數
    df_vacation_ = df_merge.loc[df_merge['考勤狀態'] == '請假']
    df_vacation = df_vacation_.groupby('英文名')['考勤狀態'].count().reset_index()
    df_vacation = df_vacation.rename(columns={'考勤狀態': '請假次數'})
    name_list = set(df_merge.loc[:, '英文名'].values.tolist())

    df_all_work_time.loc[:, '工時中位數'] = ''
    for n in name_list:
        temp_df = df_merge.loc[df_merge['英文名'] == n].reset_index(drop=True)
        # # 為了去除沒上班的天數
        temp_df.loc[:, '考勤狀態'] = temp_df.loc[:, '考勤狀態'].fillna('')
        tmp_df = temp_df.loc[
            (temp_df['考勤狀態'] != '尚未到職') &
            (temp_df['考勤狀態'] != '已離職') &
            (temp_df['考勤狀態'] != '假日')].reset_index(drop=True)
        df_all_work_time.loc[df_all_work_time['英文名'] == n, '工時中位數'] = tmp_df['實際工時'].median()

    df_all_work_time = df_all_work_time.loc[df_all_work_time['實際工時'] != 0]
    df_all_work_time.loc[:, '中位總時數'] = df_all_work_time.loc[:, '工時中位數'].apply(lambda x: int(x) * work_day)

    df_all_work_time.loc[:, '平均工時'] = df_all_work_time.loc[:, '實際工時'].apply(lambda x: int(x) / work_day)
    df_all = df_all_work_time.merge(df_vacation, how='left', on='英文名')
    df_all.loc[:, '請假次數'] = df_all.loc[:, '請假次數'].fillna(0)

    df_punch = df_all.apply(lambda x: _calculate_threshold(x), axis=1)
    df_punch.to_excel('df_punch.xlsx', index=False)
    return df_punch


# # 讀人資資料表
def _loading_HR_data():
    usecol = {'EmpCName': '中文名', 'EmpEName': '英文名', 'Sex': '性別', 'Marriage': '婚姻狀況',
              'Estatus': '身份別', 'OnJob': '在職狀態', 'TCIJoinDate': '集團到職日', 'QuitDate': '離職日',
              'Attendance': '是否統計考勤'}
    # # OnJob: 0:離職;1:在職;2:留職停薪;3:尚未到職;4:轉調公司
    # # Estatus: 0:臨時人員;1:正職人員;2:約聘人員

    df = pd.read_excel(os.path.join('.', 'imported_data', 'Employee主要身份別.xlsx'), index_col=False, usecols=usecol.keys())
    df.replace({np.nan: ''}, inplace=True)
    EName = df.loc[:, 'EmpEName']

    df = df.loc[(df['Attendance'] == 1) & (df['Estatus'] == 1)]
    # df = df.loc[~df['EmpEName'].duplicated()]
    df = df.drop_duplicates(subset=['EmpEName'])
    # # 清除在職狀態與離職日顯示不合的筆數
    onjob_df = df.loc[(df['OnJob'] == 1) & (df['QuitDate'] == '')]
    quit_df = df.loc[(df['OnJob'] == 0) & (df['QuitDate'] != '')]
    new_df = pd.concat([onjob_df, quit_df]).reset_index(drop=True)

    for date_col in ['TCIJoinDate', 'QuitDate']:
        new_df.loc[:, date_col] = new_df.loc[:, date_col].apply(lambda x: pd.to_datetime(x).date() if x else x)

    _new_df = new_df.apply(lambda x: _serve_days(x), axis=1)
    _new_df = _new_df.loc[_new_df['ServeDays'] > 0]
    df_HR = _new_df.drop(columns=['Estatus', 'Attendance', 'TCIJoinDate', 'QuitDate'])
    df_HR.to_excel(os.path.join('.', 'df_HR.xlsx'), index=False)

    return df_HR


def _logistic_pred(df_Final):
    # # Since we have class imbalance (i.e. more employees with turnover=0 than turnover=1)
    # # let's use stratify=y to maintain the same ratio as in the training dataset when splitting the dataset
    # # Splitting data into training and testing sets


    # X_train, X_test, y_train, y_test = train_test_split(df_Final_copy.values.reshape(-1,1),
    #                                                     target,
    #                                                     test_size=0.25,
    #                                                     random_state=7,
    #                                                     stratify=target)
    target = df_Final['OnJob'].copy()
    df_Final = df_Final[['EmpEName', 'EmpCName', 'Sex', 'Marriage', 'ServeDays', '平均工時', '實際工時',
                         '工時中位數', '中位總時數', '請假次數', '超過每人中位數閥值', 'OnJob']]
    df_Final_copy = df_Final.copy()
    # # use_col = ['Sex', 'Marriage', 'ServeDays', '平均工時', '工時中位數', '請假次數']
    df_Final_copy = df_Final_copy[['Sex', 'Marriage', 'ServeDays', '平均工時', '工時中位數', '請假次數']]
    df_Final_copy = df_Final_copy.apply(lambda x: pd.to_numeric(x, errors='coerce'))
    #
    # print(df_Final_copy.loc[df_Final_copy.isna().values.any(axis=1)])
    try:
        X_train, X_test, y_train, y_test = train_test_split(df_Final_copy,
                                                            target,
                                                            test_size=0.25,
                                                            random_state=7,
                                                            stratify=target)

        # Build machine learning model
        # Logistic Regression
        kfold = model_selection.KFold(n_splits=10, random_state=7, shuffle=True)
        modelCV = LogisticRegression(solver='liblinear',
                                     class_weight="balanced",
                                     random_state=7)
        scoring = 'roc_auc'
        results = model_selection.cross_val_score(
            modelCV, X_train, y_train, cv=kfold, scoring=scoring)
        print("AUC score (STD): %.2f (%.2f)" % (results.mean(), results.std()))

        # fine-tune
        param_grid = {'C': np.arange(1e-03, 2, 0.01)}  # hyper-parameter list to fine-tune
        log_gs = GridSearchCV(LogisticRegression(solver='liblinear',  # setting GridSearchCV
                                                 class_weight="balanced",
                                                 random_state=7),
                              n_jobs=3,
                              return_train_score=True,
                              param_grid=param_grid,
                              scoring='roc_auc',
                              cv=10)

        log_grid = log_gs.fit(X_train, y_train)
        log_opt = log_grid.best_estimator_
        results = log_gs.cv_results_

        print('=' * 20)
        print("best params: " + str(log_gs.best_estimator_))
        print("best params: " + str(log_gs.best_params_))
        print('best score:', log_gs.best_score_)
        print('=' * 20)

        # Evaluation
        # Confusion Matrix
        cnf_matrix = metrics.confusion_matrix(y_test, log_opt.predict(X_test))
        class_names = [0, 1]  # name  of classes


        # Accuracy of Logistic Regression Classifier
        print('Accuracy of Logistic Regression Classifier on test set: {:.2f}'.format(log_opt.score(X_test, y_test) * 100))

        # Classification report for the optimised Log Regression
        log_opt.fit(X_train, y_train)
        print(classification_report(y_test, log_opt.predict(X_test)))

        # fine-tune 後的模型預測
        log_opt.fit(X_train, y_train)  # fit optimised model to the training data

        # predicted train and test
        y_pred_train = log_opt.predict(X_train)
        y_pred_test = log_opt.predict(X_test)

        df_Final.loc[:, 'preds'] = np.hstack([y_pred_train, y_pred_test])

        # predicted train and test probability
        probs = log_opt.predict_proba(X_test)  # predict probabilities
        pros_train_0 = log_opt.predict_proba(X_train)[:, 0]
        pros_train_1 = log_opt.predict_proba(X_train)[:, 1]
        pros_test_0 = log_opt.predict_proba(X_test)[:, 0]
        pros_test_1 = log_opt.predict_proba(X_test)[:, 1]

        df_Final.loc[:, 'preds_prob_leave'] = np.hstack([pros_train_0, pros_test_0])
        df_Final.loc[:, 'preds_prob_On'] = np.hstack([pros_train_1, pros_test_1])
        export_time = datetime.datetime.now().strftime('%Y%m%d_%H')
        df_Final.to_excel(f'EmployeeQuitPred_{export_time}.xlsx', index=False)
        # probs = probs[:, 1] # we will only keep probabilities associated with the employee leaving
        # logit_roc_auc = roc_auc_score(y_test, probs) # calculate AUC score using test dataset
        # print('AUC score: %.3f' % logit_roc_auc)
    except Exception as Ex:
        if not df_Final_copy.loc[df_Final_copy.isna().values.any(axis=1)].empty:
            print(df_Final_copy.loc[df_Final_copy.isna().values.any(axis=1)])
        else:
            print(Ex)


if __name__ == '__main__':
    df_Punch = pd.read_excel(os.path.join('.', 'df_punch.xlsx'), index_col=False)
    df_HR = pd.read_excel(os.path.join('.', 'df_HR.xlsx'), index_col=False)

    df_Punch = df_Punch.rename(columns={'英文名': 'EmpEName'})
    df_tw = df_Punch.merge(df_HR, how='left', on='EmpEName')
    df_tw = df_tw.dropna(subset=['ServeDays', 'Marriage'])

    _logistic_pred(df_tw)
