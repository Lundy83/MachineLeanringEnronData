#!/usr/bin/python

import sys
import pickle
from pprint import pprint
sys.path.append("C:/Users/Joe/OneDrive/Desktop/wgu/ud120-projects/tools/")
from feature_format import featureFormat, targetFeatureSplit
import pandas as pd
from matplotlib import pyplot as plt
from time import time
sys.path.append("C:/Users/Joe/OneDrive/Desktop/wgu/ud120-projects/final_project/")
from tester import dump_classifier_and_data, test_classifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','exercised_stock_options', 'total_stock_value', 'bonus',
                      'salary', 'deferred_income', 'long_term_incentive',
                      'restricted_stock', 'total_payments', 'shared_receipt_with_poi',
                      'loan_advances', 'expenses', 'from_poi_to_this_person', 'from_this_person_to_poi']

### Load the dictionary containing the dataset
with open("C:/Users/Joe/OneDrive/Desktop/wgu/ud120-projects/final_project/final_project_dataset.pkl", "rb") as data_file:
#with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)
# Convert to a pandas df
enron_data = pd.DataFrame.from_dict(data_dict, 'index', columns=features_list)

# set the index of df to be the employees series:
#employees = pd.Series(list(data_dict.keys()))
#enron_data.set_index(employees, inplace=True)
    
### Task 2: Remove outliers
enron_data.drop('TOTAL', axis = 0, inplace = True)
enron_data.drop('THE TRAVEL AGENCY IN THE PARK', axis = 0, inplace = True)

# change numeric values into floats or ints; also change NaN to zero:
enron_data['bonus'] = pd.to_numeric(enron_data['bonus'], errors = 'coerce').copy().fillna(0)
enron_data['salary'] = pd.to_numeric(enron_data['salary'], errors = 'coerce').copy().fillna(0)
#print(enron_data.head())

### Task 3: Create new feature(s)
enron_data['bonus-to-salary_ratio'] = enron_data['bonus']/enron_data['salary']


### Store to my_dataset for easy export below.
my_dataset = enron_data

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset.transpose(), features_list)
labels, features = targetFeatureSplit(data)

#####################
# Scale features

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
features = scaler.fit_transform(features)

###################
### Task 4: Try a variety of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

###import features
from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
# Stratified ShuffleSplit cross-validator
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1000, random_state = 42)
skb = SelectKBest(f_classif)
pca = PCA()
t0 = time()
########################
#Classifer 1 GaussianNB
#X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
#Y = np.array([1, 1, 1, 2, 2, 2])
#clf_GNB = GaussianNB()


#clf_GNB.fit(features_train, labels_train)



#prediction = clf_GNB.predict(features_test)

#y_pred = [0, 2, 1, 3]
#y_true = [0, 1, 2, 3]
#accuracy_score(y_true, y_pred)
#accuracy_score(y_true, y_pred, normalize=False)

#print ("training time:", round(time()-t0, 3), "s")
#print ("Accuracy: ",accuracy_score(prediction, labels_test))
#print ("Precision : ",precision_score(prediction, labels_test))
#print ("Recall : ",recall_score(prediction, labels_test))
#print ("F1-Score : ",f1_score(prediction, labels_test))

#Classifer 2 Decision Tree


clf = tree.DecisionTreeClassifier()
###Added for K value error
clf.fit(features_train.values.reshape(-1, 1), labels_train)
pipeline = Pipeline(steps = [("SKB", skb), ("dtree",clf)])
param_grid = {"SKB__k":[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],
              "dtree__criterion": ["gini", "entropy"],
              "dtree__min_samples_split": [2, 4, 8, 10]}


grid = GridSearchCV(pipeline, param_grid, verbose = 0, cv = sss, scoring = 'f1')
grid.fit(features, labels)
# best algorithm
clf = grid.best_estimator_

t0 = time()
clf.fit(features_train, labels_train)
prediction = clf.predict(features_test)
y_pred = [0, 2, 1, 3]
y_true = [0, 1, 2, 3]
accuracy_score(y_true, y_pred)
0.5
accuracy_score(y_true, y_pred, normalize=False)
2

print ("training time:", round(time()-t0, 3), "s")
print ("Accuracy: ",accuracy_score(prediction, labels_test))
print ("Precision : ",precision_score(prediction, labels_test))
print ("Recall : ",recall_score(prediction, labels_test))
print ("F1-Score : ",f1_score(prediction, labels_test))


################### Select K Best feature selection####################


#def Select_K_Best(data_dict, features_list, k):

    #data_array = featureFormat(data_dict, features_list)
    #labels, features = targetFeatureSplit(data_array)

    #k_best = SelectKBest(k=k)
    #k_best.fit(features, labels)
    #scores = k_best.scores_
    #tuples = zip(features_list[1:], scores)
    #k_best_features = sorted(tuples, key=lambda x: x[1], reverse=True)

    #return k_best_features[:k]
    
 # Obtaining the boolean list showing selected features
features_selected_bool = grid.best_estimator_.named_steps['SKB'].get_support()
# Finding the features selected by SelectKBest
features_selected_list = [x for x,y in zip(features_list[1:], features_selected_bool) if y]

print "Total number of features selected by SelectKBest algorithm : ",len(features_selected_list)

# Finding the score of features 
feature_scores =  grid.best_estimator_.named_steps['SKB'].scores_
# Finding the score of features selected by selectKBest
feature_selected_scores = feature_scores[features_selected_bool]

# Creating a pandas dataframe and arranging the features based on their scores and rankimg them 
imp_features_df = pd.DataFrame({'Features_Selected':features_selected_list, 'Features_score':feature_selected_scores})
imp_features_df.sort_values('Features_score', ascending = False,inplace = True)
Rank = pd.Series(list(range(1,len(features_selected_list)+1)))
imp_features_df.set_index(Rank, inplace = True)
print "The following table shows the feature selected along with its corresponding scores"
imp_features_df   
    
    
    
    
    
    
    
#Classifer 3 KNN
#clf = KNeighborsClassifier()
#ss = StratifiedShuffleSplit(n_splits=10, test_size=0.3,random_state = 42)
#pipeline = Pipeline(steps = [("scaling", scaler), ("SKB", skb),  ("knn",clf)])
#param_grid = {"SKB__k":[3,4,5,6,7,8,9,10,11,12,13,14,15, 16, 17, 18], 
#              "knn__n_neighbors": [3,4,5,6,7,8,9,11,12,13,15],
#              }

#grid = GridSearchCV(pipeline, param_grid, verbose = 0, cv = sss, scoring = 'f1')
#t0 = time()
grid.fit(features_train, labels_train)

################

best_clf = grid.best_estimator_
#prediction = clf.predict(features_test)
y_pred = [0, 2, 1, 3]
y_true = [0, 1, 2, 3]
accuracy_score(y_true, y_pred)
0.5
accuracy_score(y_true, y_pred, normalize=False)
2

#print ("training time:", round(time()-t0, 3), "s")
#print ("Accuracy: ",accuracy_score(best_clf, labels_test))
#print ("Precision : ",precision_score(best_clf, labels_test))
#print ("Recall : ",recall_score(best_clf, labels_test))
#print ("F1-Score : ",f1_score(best_clf, labels_test))


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

#clf.fit(features_train, labels_train)

#prediction = clf.predict(features_test)

DT_features_list = ['poi','salary','bonus']

my_dataset = my_dataset.transpose()

test_classifier(clf, my_dataset,DT_features_list, folds=1000)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(best_clf, my_dataset, features_list)