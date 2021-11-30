import joblib
import numpy as np
import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.model_selection import KFold

import sys
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

trainfile = sys.argv[1] if len(sys.argv) > 1 else "NSL-KDD/KDDTrain+.txt"
testfile = sys.argv[2] if len(sys.argv) > 2 else "NSL-KDD/KDDTest+.txt"

def info(title, test, pred):
    print(title, "=" * 10)
    try:
        print("confusion_matrix, tn, fp, fn, tp = ", multilabel_confusion_matrix(test, pred))
    except: pass
    print("classification_report", classification_report(test, pred))

class DDoSDetector:
    def __init__(self):
        self.__trained = False
        self.__columns = (['duration'
            ,'protocol_type'
            ,'service'
            ,'flag'
            ,'src_bytes'
            ,'dst_bytes'
            ,'land'
            ,'wrong_fragment'
            ,'urgent'
            ,'hot'
            ,'num_failed_logins'
            ,'logged_in'
            ,'num_compromised'
            ,'root_shell'
            ,'su_attempted'
            ,'num_root'
            ,'num_file_creations'
            ,'num_shells'
            ,'num_access_files'
            ,'num_outbound_cmds'
            ,'is_host_login'
            ,'is_guest_login'
            ,'count'
            ,'srv_count'
            ,'serror_rate'
            ,'srv_serror_rate'
            ,'rerror_rate'
            ,'srv_rerror_rate'
            ,'same_srv_rate'
            ,'diff_srv_rate'
            ,'srv_diff_host_rate'
            ,'dst_host_count'
            ,'dst_host_srv_count'
            ,'dst_host_same_srv_rate'
            ,'dst_host_diff_srv_rate'
            ,'dst_host_same_src_port_rate'
            ,'dst_host_srv_diff_host_rate'
            ,'dst_host_serror_rate'
            ,'dst_host_srv_serror_rate'
            ,'dst_host_rerror_rate'
            ,'dst_host_srv_rerror_rate'
            ,'attack'
            ,'level'])

        self.__features__ = ([
            'src_bytes' #vip
            ,'flag' # small
            ,'count' #vip
            ,'dst_host_diff_srv_rate' #vip
            ,'diff_srv_rate' #vvip
            ,'same_srv_rate' #vip
            ,'dst_host_srv_serror_rate' #vvip
            ,'dst_host_serror_rate' #vip

            ,'dst_host_same_src_port_rate' #vip
            ,'dst_bytes' #vip
            ,'dst_host_same_srv_rate' #vip

            ,'service' #vip
            ,'srv_count' #vip
            ,'srv_serror_rate' #vip
            ,'dst_host_count' #vip
            ,'dst_host_rerror_rate' #vip
            ,'protocol_type' #vip
            ,'wrong_fragment'  # vip
            ,'srv_rerror_rate'

            ,'attack'])

        self.__drop_features__ = [''
            ,'dst_host_srv_diff_host_rate' #bad
            ,'dst_host_srv_count' #bad
            ,'rerror_rate' # bad
            ,'dst_host_srv_rerror_rate' #bad
            ,'srv_diff_host_rate' #bad
            ,'serror_rate' #small bad
            ,'logged_in' #bad
            ,'num_compromised' #small bad
            ,'hot' #small bad
            ,'duration' #bad
            ,'is_guest_login' #bad
            ,'num_root' #bad
            ,'num_failed_logins' #small bad
            ,'num_file_creations' #bad
            ,'root_shell' #bad
            ,'land' #bad
            ,'num_access_files' #bad
            ,'num_shells' #small bad
            ,'su_attempted' #small bad
            ,'urgent' #bad
            ,'is_host_login' #bad
            ,'num_outbound_cmds' #bad
            ]

        # normal
        dos_attacks = ['apache2','back','land','neptune','mailbomb','pod',
                'processtable','smurf','teardrop','udpstorm','worm']
        probe_attacks = ['ipsweep','mscan','nmap','portsweep','saint','satan']

        U2R_attacks = ['buffer_overflow','loadmodule','perl','ps',
                'rootkit','sqlattack','xterm']

        R2L_attacks = ['ftp_write', 'guess_passwd', 'http_tunnel', 'imap',
                'httptunnel', 'multihop', 'named', 'phf', 'sendmail',
                'snmpgetattack', 'snmpguess', 'spy', 'warezclient',
                'warezmaster', 'xlock', 'xsnoop']

        attack_labels = ['Normal','DoS','Probe','U2R','R2L']
        self.__attack_maps = {'normal':0}
        for k in dos_attacks:
            self.__attack_maps[k] = 1
        for k in R2L_attacks:
            self.__attack_maps[k] = 2
        for k in U2R_attacks:
            self.__attack_maps[k] = 3
        for k in probe_attacks:
            self.__attack_maps[k] = 4


    def map_attack(self, v):
        if v is None:
            return 5
        return self.__attack_maps[v]

    def trained(self):
        return self.__trained

    def load_train(self, train_file, test_file):
        df = pd.read_csv(train_file)
        df.columns = self.__columns[:]
        # y = df.attack.apply(self.map_attack)
        df_test = pd.read_csv(test_file)
        df_test.columns = self.__columns[:]
        df = df[self.__features__]
        df_test = df_test[self.__features__]

        df_all = df.append(df_test, ignore_index=True)

        tfy = OneHotEncoder(handle_unknown='ignore',sparse=False)
        self.tfy = tfy.fit(df_all.iloc[:, -1:])
        y = self.tfy.transform(df.iloc[:, -1:])

        df = df.drop(columns=['attack'])
        df_all = df_all.drop(columns=['attack'])
        df_test = df_test.drop(columns=['attack'])

        categorical_ix = df.select_dtypes(include=['object', 'bool']).columns
        t = [('cat', OneHotEncoder(handle_unknown='error',sparse=False), categorical_ix)]
        tfx = ColumnTransformer(transformers=t, remainder="passthrough")
        self.tfx = tfx.fit(df_all)
        # print(df.head())

        X = self.tfx.transform(df)
        # print(pd.DataFrame(X).head())

        return X, y

    def load_test(self, test_file):
        df = pd.read_csv(test_file)
        df.columns = self.__columns[:]
        df = df[self.__features__]
        # y = df.attack.apply(self.map_attack)

        attack = df.attack
        y = self.tfy.transform(df.iloc[:, -1:])
        X = self.tfx.transform(df.drop(columns=['attack']))

        return X, y, attack

    def baseRFC(self):
        rf = RandomForestClassifier(n_jobs=-1,
                n_estimators=40,
                random_state=42,
                # oob_score=True,
                max_features="sqrt")
        return rf

    def train(self, X, y):
        regressor = self.baseRFC()

        # scores = cross_validate(regressor, X, y, cv=10, return_estimator=True)
        # print("scores", scores)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                random_state=0, test_size=0.3)
        regressor.fit(X_train, y_train)
        # regressor.fit(X, y)
        y_pred_test = regressor.predict(X_test)
        # info("train", y_test, y_pred_test)
        # importances = list(regressor.feature_importances_)
        # feature_list = self.tfx.get_feature_names_out()
        # plt.bar(height=importances, x=feature_list)
        # plt.show()
        # feature_importances = [(feature, importance) for feature, importance in zip(feature_list, importances)]
        # feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
        # [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
        self.regressor = regressor
        self.__trained = True


    def predict(self, X):
        return self.regressor.predict(X)

    def save_model(self):
        joblib.dump(self.tfx, 'tfx.joblib')
        joblib.dump(self.tfy, 'tfy.joblib')
        joblib.dump(self.regressor, 'model.joblib')

    def load_model(self):
        if all((os.path.exists('tfx.joblib'),
            os.path.exists('model.joblib'),
            os.path.exists('tfy.joblib'))):
            self.tfx = joblib.load('tfx.joblib')
            self.tfy = joblib.load('tfy.joblib')
            self.regressor = joblib.load('model.joblib')
            self.__trained = True

    def GridSearch(self, X_train, y_train):
        estimator = RandomForestClassifier(
                random_state=42,
                # oob_score=True,
                )
        param_grid = {
                "n_estimators"      : [10, 20, 30, 40, 50],
                "max_features"      : ["sqrt"],
                }

        grid = GridSearchCV(estimator, param_grid, n_jobs=-1, cv=10)
        grid.fit(X_train, y_train)
        return grid.best_estimator_, grid.best_score_ , grid.best_params_

    def FeaturesSelect(self, X_train, y_train):
        sel = SelectFromModel(self.baseRFR())
        sel.fit(X_train, y_train)
        supports = sel.get_support()
        print(supports)
        print(len(supports))
        feature_list = self.tfx.get_feature_names_out()
        print(feature_list[supports])
        pd.Series(sel.estimator_.feature_importances_.ravel()).hist()
        plt.show()

detector = DDoSDetector()
detector.load_model()
if not detector.trained():
    X_train, y_train = detector.load_train(trainfile, testfile)
    # detector.FeaturesSelect(X_train, y_train)

    # X_test, y_test = detector.load_test(testfile)
    # best_estimator, best_score, best_params = detector.GridSearch(X_train, y_train)
    # print(best_score)
    # print(best_params)
    # y_pred_test = best_estimator.predict(X_test)
    # info("test", y_test, y_pred_test)

    detector.train(X_train, y_train)
    detector.save_model()

print(detector.tfy.categories_)
X_test, y_test, y_label = detector.load_test(testfile)
y_pred_test = detector.predict(X_test)
y_pred_label = detector.tfy.inverse_transform(y_pred_test)
y_pred_label = y_pred_label[:,0]
print('inverse', y_pred_label)
y_label_v = y_label.apply(detector.map_attack)
y_pred_label_v = pd.Series(y_pred_label).apply(detector.map_attack)
print(y_pred_label_v)
df = pd.DataFrame({
    'aculabel':y_label,
    'acuv': y_label_v,
    'predictlabel':y_pred_label,
    'predict': y_pred_label_v
    })
# .to_csv("pred.csv")
info("test", y_label_v, y_pred_label_v)

# df['predict'].hist()
# plt.show()
