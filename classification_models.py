import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score


def training(X_train, X_test,y_train,y_test,fold_no,model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_pred,y_test)
    f1 = f1_score(y_pred,y_test,average='weighted')
#     auc1 = roc_auc_score(y_pred,y_test)
    print('For Fold {} the accuracy is {} f1 is {}'.format(str(fold_no),acc,f1))
    return acc,f1
def classification_model(X,y):
    model = RandomForestClassifier()
#     model = SVC()
    skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=23)
    fold_no = 1
    accs = []
    f1s = []
    aucs = []
    for train_index,test_index in skf.split(X,y):
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]
        acc,f1 = training(X_train,X_test,y_train,y_test,fold_no,model)
        fold_no += 1
        accs.append(acc)
        f1s.append(f1)
#         aucs.append(auc1)
    return np.mean(accs),np.mean(f1s),np.std(f1s)/np.sqrt(len(f1s))
