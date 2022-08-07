import scipy.io
from scipy.io import arff
import numpy as np
import pandas as pd
from scipy.linalg import svd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from scipy import signal
from sklearn.cluster import KMeans
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
# from kennard_stone import train_test_split
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import KFold
from pls_model import *




def plot_spectra(X,wn):
    for i in range(len(X)):
        plt.plot(wn,X[i])
        plt.ylabel('Feature values')
        plt.xlabel('Features')
    return None
def singular_values(X):
    U,s,VT = svd(X)
    return s
def effective_rank(alpha,s):
    tsum = sum(s)
    r,sum1 = 0,0
    for i in s:
        if sum1/tsum < alpha:
            sum1 +=i
            r += 1
        else:
            break
    return r

def rs_svs_wns(X,wn,alpha,k):
    svs,wns,rs,w_mat = [],[],[],[]
    for i in range(len(X[0])-k+1):
        arr1 = X[:,i:i+k]
        w_mat.append(arr1.T[1])
        S = singular_values(arr1)
        svs.append(S[:k])
        wns.append(wn[i+(k//2)])
        rs.append(effective_rank(alpha,S))
    w_mat = np.array(w_mat)
    return rs,svs,wns,w_mat

def get_sv(rs,svs,wns):
    sv1,sv2 = [],[]
    for i in range(len(rs)):
        if rs[i]==1:
            sv1.append(svs[i][0])
            sv2.append(svs[i][0])
        elif rs[i]==2:
            sv1.append(svs[i][0]/svs[i][1])
            sv2.append(svs[i][0]+svs[i][1])
        else:
            sv1.append(svs[i][0]/(sum(svs[i][1:rs[i]])))
            sv2.append(sum(svs[i][:rs[i]]))
    sv1 = np.array(sv1)
    sv2 = np.array(sv2)
    return sv1,sv2
def get_peaks(sv):
    peak_indices = signal.argrelextrema(sv, np.greater)
    return peak_indices[0]
def plot_peaks(sv1,wns,rs,alpha,k):
    wns = np.array(wns)
    wnt = []
    for i in range(1,k+1):
        wn = []
        s1 = []
        for j in range(len(rs)):
            if rs[j]==i:
                wn.append(wns[j])
                s1.append(sv1[j])
        if len(s1)!=0:
            s1 = np.array(s1)
            wn = np.array(wn)
            peaks = get_peaks(s1)
            wnt += list(wn[peaks])
            plt.figure()
            plt.plot(wn,s1)
            plt.plot(wn[peaks],s1[peaks],'o')
            plt.ylabel('SV'+str(i))
            plt.xlabel('Wavenumber')
            plt.title("alpha: "+str(alpha)+" "+ "Window size: "+str(k))
    return wnt

def get_rs_matrix(X,peaks):
    r_spectra = []
    for p in peaks:
        r_spectra.append(X[:,p+1])
    r_spectra = np.array(r_spectra)
    r_spectra = r_spectra.T
    return r_spectra

def clustering_data(X,wns,w_mat,sv,k):
    kmeans = KMeans(init="random",
    n_clusters=k,
    n_init=10,
    max_iter=300,
    random_state=42)
    kmeans.fit(w_mat)
    l = list(kmeans.labels_)
    ft = [i for i in range(len(wns))]
    wnt,ws,ps = {},{},{}
    for i in range(len(l)):
        wnt[wns[i]]=l[i]
        ws[wns[i]]=sv[i]
        ps[wns[i]]=ft[i]
    maxsv = []
    for i in range(k):
        max1 = 0
        for j in wns:
            if wnt[j]==i:
                if ws[j]>max1:
                    max1 = ws[j]
                    wnp = j
                    pk = ps[j]
        maxsv.append([wnp,max1,pk,i])
    pks,wns = [],[]
    for i in range(k):
        pks.append(maxsv[i][2])
        wns.append(maxsv[i][0])
    dt = get_rs_matrix(X,pks)
    wns.sort()
    return dt,wns,pks

def final_data_algo2(X,wn,alpha):
    s = singular_values(X)
    k = effective_rank(alpha,s)
    k_l = 0
    w = 0
    i = k
    rs,svs,wns,w_mat = rs_svs_wns(X,wn,alpha,k)
    sv1,sv2 = get_sv(rs,svs,wns)
    while k_l!=k and i<len(X[0])-k+1:
        data,wnr,ps = clustering_data(X,wns,w_mat,sv2,i)
        s1 = singular_values(data)
        k_l = effective_rank(alpha,s1)
        w = i
        #print(w)
        i += 1
    data,wnr,pks = clustering_data(X,wns,w_mat,sv2,w)
    return data,wnr,w,pks

mat = scipy.io.loadmat('milk.mat')
X = mat['X']
y = mat['protein']
wn = [i for i in range(4000,10228,4)]
plot_spectra(X,wn)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
data2,wns,w,pks2 = final_data_algo2(X_train,wn,0.98)
X_test2 = get_rs_matrix(X_test,pks2)
rmsep1,comps,std1,new_rmsep1 = pls_model(X_train,X_test,y_train,y_test)
rmsep2,comps,std1,new_rmsep2 = pls_model(data2,X_test2,y_train,y_test)
print("All wavelengths: {} , RMSEP: {}".format(len(X[0]),new_rmsep1))
print("Algorithm 2: {} , RMSEP: {}".format(len(data2[0]),new_rmsep2))
