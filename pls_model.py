
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
def pls_model(X_train,X_test,y_train,y_test):
    x = []
    rmseps = []
    for i in range(1,min(len(X_train[0])+1,51)):
        pls = PLSRegression(n_components=i)
        cv = LeaveOneOut()
        scores = cross_val_score(pls, X_train, y_train, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
        scores = np.absolute(scores)
        rmseps.append(np.sqrt(np.mean(scores)))
        x.append(i)
    p = min(rmseps)
    std = np.std(rmseps)/np.sqrt(len(rmseps))
    rmsep,comps = 0,0
    for i in rmseps:
        if p-std <= i and i <= p+std:
            rmsep,comps = i,rmseps.index(i)+1
            break
    if rmsep==0 and comps==0:
        rmsep = min(rmseps)
        comps = rmseps.index(rmsep)+1
    pls2 = PLSRegression(n_components=comps)
    pls2.fit(X_train,y_train)
    #r2 = r2_score(pls2.predict(X_test),y_test)
    new_rmsep = np.sqrt(mean_squared_error(pls2.predict(X_test),y_test))
    with plt.style.context('ggplot'):
        plt.figure(figsize=(10,6))
        plt.plot(x, np.array(rmseps), '-v', color='black', mfc='black')
        plt.plot(x[comps-1], np.array(rmseps)[comps-1], 'P', ms=10,color='green', mfc='green')
        plt.plot(x[rmseps.index(min(rmseps))],np.array(rmseps)[rmseps.index(min(rmseps))],'P',ms=10,color='red',mfc='red')
        plt.axvline(x=x[comps-1],color='green')
        plt.axvline(x=x[rmseps.index(min(rmseps))],color='red')
        plt.xlabel('Number of PLS components')
        plt.xticks = x
        plt.ylabel('LOOCV Score')
        plt.title('PLS')
    plt.show()
    return rmsep,comps,std,new_rmsep

