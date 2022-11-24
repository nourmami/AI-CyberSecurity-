from time import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import fetch_kddcup99

# Load dataset and select 10% of it for faster run time
dataset = fetch_kddcup99(subset=None, shuffle=True, percent10=True)
# http://www.kdd.org/kdd-cup/view/kdd-cup-1999/Tasks
X = dataset.data
y = dataset.target

#the features of the dataset
feature_cols = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serrer_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count','dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']
X = pd.DataFrame(X, columns = feature_cols)

#the labels of the dataset (normal or attack) .Series is a one-dimensional labeled array capable of holding any data type
y = pd.Series(y)
X.head()

#convert the columns into floats for efficient processing
for col in X.columns:  
    try:
        X[col] = X[col].astype(float)
    except ValueError:
        pass
# convert the categorical into dummy or indicator variables:
X = pd.get_dummies(X, prefix=['protocol_type_', 'service_', 'flag_'], drop_first=True)
 
X.head()

#count the number of each attack type
y.value_counts()


# fit a classification tree with max_depth=3 on all data
from sklearn.tree import DecisionTreeClassifier

treeModel = DecisionTreeClassifier(max_depth=7)

#cross_val_score returns the mean accuracy on the given test data and labels
scores = cross_val_score(treeModel, X, y, scoring='accuracy', cv=5)
#mean of the scores
print(np.mean(scores)) #0.9955204407492013
treeModel.fit(X, y)


# How about a Random Forest?
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()

scores = cross_val_score(rf, X, y, scoring='accuracy', cv=5)

print(np.mean(scores)) #0.9997307783262454

rf.fit(X, y)



################Isolated Forest####################

# Supervised and Outlier Detection with KDD

# In this example, we will want to use binary data where 1 will represent a "not-normal" attack
from sklearn.model_selection import train_test_split

# convert the labels into boolean true or false
y_binary = y != 'normal.'
# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_binary)
# convert the labels into floats & check our null accuracy
y_test.value_counts(normalize=True)  

#model the data
model = IsolationForest()
model.fit(X_train)  # notice that there is no y in the .fit

y_predicted = model.predict(X_test)


# turn the labels into 0s and 1s instead of -1 and 1
y_predicted = np.where(y_predicted==1, 1, 0)  

#scores   the smaller, the more anomolous the data
#decision_function returns the anomaly score of each sample using the IsolationForest algorithm
scores = model.decision_function(X_test)

from sklearn.metrics import accuracy_score
preds = np.where(scores < 0, 0, 1)  # customize threshold
accuracy_score(preds, y_test)
for t in (-2, -.15, -.1, -.05, 0, .05):
    preds = np.where(scores < t, 0, 1)  # customize threshold
    print(t, accuracy_score(preds, y_test))

## -0.05 0.816988648325 gives us better than the null accuracy, without ever needing the testing set
# This shows how we can can achieve predictive results without labeled data


# This is an interesting use case of novelty detection becuase generally, when given labels
# we do not use such tactics.

