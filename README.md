# Variable selection using Unsupervised learning

## List of files:
- Algorithm_1.py -> selects important features using SVD, and prints the performance results for comparison
- Algorithm_2.py -> selects important features using SVD and k-means and prints the performance results for comparison
- pls_model.py -> PLSR model for validation
- classification_model.py -> RFR and SVC models for validation

## Algorithm 1:
- Algorithm 1 uses SVD(Singular value decomposition) for quantifying feature information. Sliding window technique is used to apply SVD on windows and singular value plots are drawn for selection of local maxima points.
## Algorithm 2:
- Algorithm 2 integrates SVD and k-means for selecting important features. It also uses intrinsic dimension equality criterion for choosing the number of features selected. Feature clustering is employed to ensure all the selected features are non-redundant because, in the selected feature subset, the features are taken from different clusters, so the similarity among them is substantially low.

The performance of the proposed algorithms is validated on different end applications: (i) Building multivariate calibration models on spectroscopic data and (ii) General Classification/regression tasks on various benchmark datasets. A total of 49 datasets are used for validation purposes, and the results of feature selection are compared with results obtained using all features.
