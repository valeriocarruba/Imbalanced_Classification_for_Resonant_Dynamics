import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from numpy import mean
from numpy import where
from matplotlib import pyplot
from collections import Counter
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

filename = 'nu6_aligned'
data = pd.read_csv(filename,
                           skiprows=0,
                           header=None,
                           delim_whitespace=True,
                           index_col=None,
                           names = ['id','a','sini','label'],
                           low_memory=False,
                           dtype={'id': np.float64,
                                  'a': np.float64,
                                  'sini': np.float64,
                                  'label': np.integer
                                  }
                           )


X = data.iloc[:, 1:3].values
y = data['label'].values

# summarize class distribution
counter = Counter(y)
print(counter)
# define pipeline
model = ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.7000000000000001, min_samples_leaf=13, min_samples_split=17, n_estimators=100, class_weight='balanced')

steps = [('over', SMOTE()), ('model', model)]
pipeline = Pipeline(steps=steps)
# evaluate pipeline
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

i = 0
values = np.zeros(10)  #cria array com 10 valores
while i < 10:
    scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    values[i] = mean(scores)
    i=i+1

print('Mean ROC AUC: %.3f' % values.mean())
print('Standard deviation: %.3f' % values.std())
