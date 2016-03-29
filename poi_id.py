#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import matplotlib.pyplot as pyplt

def plot_graph(features,labels):
    colors = ['b','r']
    for i in range(len(features)):
        x = features[i][0]
        y = features[i][1]
        pyplt.scatter(x,y,color=colors[int(labels[i])])
        
def get_num_of_features(x_labels):
    l = []
    num = 151
    for feat in x_labels:
        features_list = ['poi']
        features_list.append(feat)
        data = featureFormat(my_dataset, features_list, sort_keys = True)
        labels, features = targetFeatureSplit(data)
        l.append(len(features))
        #pyplt.subplot(num)
        #plot_graph(features, labels)
        #pyplt.xlabel(features_list[1])
        #pyplt.ylabel(features_list[0])
        num +=1
        
    #pyplt.show()
    return l

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','stock_pay_ratio','restricted_stock_deferred_empty','dir_exp_ratio','bonus','director_fees','expenses','to_messages','salary','from_poi_to_this_person',
                 'shared_receipt_with_poi','exercised_stock_options','total_stock_value'] # You will need to use more features
#features_list = ['poi','deferral_payments'] #only 38 valid entries
#features_list = ['poi','total_payments'] #124 entries
#features_list = ['poi','loan_advances'] #Not enough info
#features_list = ['poi','bonus']
#features_list = ['poi','restricted_stock_deferred'] #Bad predictive power
#features_list = ['poi','deferred_income'] #only 48 valid points
#features_list = ['poi','total_stock_value']
#features_list = ['poi','expenses']
#features_list = ['poi','exercised_stock_options'] #somewhat good predictor for determining who is poi
#features_list = ['poi','other']
#features_list = ['poi','long_term_incentive']
#features_list = ['poi','restricted_stock']
#features_list = ['poi','director_fees'] # bad predictive power



x_labels = ['salary', 'deferral_payments', 'total_payments','loan_advances','bonus'
            ,'restricted_stock_deferred','deferred_income','total_stock_value','expenses',
            'exercised_stock_options','other','long_term_incentive','restricted_stock','director_fees']
x_labels_e = ['to_messages','from_poi_to_this_person','from_messages','from_this_person_to_poi',
              'shared_receipt_with_poi']

#features_list.append(x_labels[7])
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop("TOTAL",0)

num_pois = 0

nan_count = [0] * len(x_labels)
for key in data_dict.keys():
    for i in range(len(x_labels)):
        if data_dict[key][x_labels[i]] == 'NaN':
            nan_count[i] += 1
            
print "Num of NaNs:", zip(nan_count,x_labels)

### Task 3: Create new feature(s)
for key in data_dict.keys():
    if data_dict[key]['poi'] == 1:
        num_pois += 1
        
    if data_dict[key]['director_fees'] == 'NaN':
        data_dict[key]['director_fees'] = 1
    if data_dict[key]['expenses'] == 'NaN':
        data_dict[key]['expenses'] = 1
    data_dict[key]['dir_exp_ratio'] = (float(data_dict[key]['director_fees']) / float(data_dict[key]['expenses'])) + 1
    
    if data_dict[key]['restricted_stock_deferred'] == 'NaN':
        data_dict[key]['restricted_stock_deferred_empty'] = 0.
    else:
        data_dict[key]['restricted_stock_deferred_empty'] = 1.
    if data_dict[key]['total_stock_value'] != 'NaN' and data_dict[key]['total_payments'] != 'NaN':
        data_dict[key]['stock_pay_ratio'] = float(data_dict[key]['total_stock_value']) / float(data_dict[key]['total_payments'] )
    else:
        data_dict[key]['stock_pay_ratio'] = -1.

print "num of pois:", num_pois
print "num of features:", len(x_labels) + len(x_labels_e)


        
### Store to my_dataset for easy export below.
my_dataset = data_dict


l = []

#l = get_num_of_features(x_labels)

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


from sklearn.feature_selection import f_classif,SelectPercentile




plot_graph(features, labels)

pyplt.xlabel(features_list[1])
pyplt.ylabel(features_list[2])
#pyplt.show()    
    
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
    
sel = SelectPercentile(f_classif,percentile=50)

sel.fit(features_train, labels_train)

features_sel = sel.transform(features_train)
features_test_sel = sel.transform(features_test)

support = sel.get_support()

sup_list = []
for i in range(len(support)):
    if support[i]:
        sup_list.append((features_list[i+1], sel.scores_[i]))
        


#print "Feature scores ", sel.scores_
print "Features selected by SelectPercentile function: ", sup_list



### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#from sklearn.naive_bayes import GaussianNB
#from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
#clf = DecisionTreeClassifier()
#clf = GaussianNB()
#rfc = RandomForestClassifier()
dtc = DecisionTreeClassifier()
parameters = {'criterion' : ('gini','entropy'),'n_estimators' : [10,30,50,15,25],'min_samples_split':[1,2,3,4,5,10,15],'max_features':('auto','sqrt','log2')}
parameters = {'criterion' : ('gini','entropy'),'min_samples_split':[1,5,10,15],'max_features':('auto','sqrt','log2')}
from sklearn.grid_search import GridSearchCV
clf = GridSearchCV(dtc,parameters)
'''
clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=10,
            min_weight_fraction_leaf=0.0, n_estimators=30, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)

clf = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features='log2', max_leaf_nodes=None, min_samples_leaf=1,
            min_samples_split=10, min_weight_fraction_leaf=0.0,
            random_state=None, splitter='best')
'''
#clf = DecisionTreeClassifier()

clf = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features='log2', max_leaf_nodes=None, min_samples_leaf=1,
            min_samples_split=1, min_weight_fraction_leaf=0.0,
            random_state=None, splitter='best')

            
#print clf
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!



clf.fit(features_sel,labels_train)
'''
import pprint as pp

pp.pprint(clf.grid_scores_)

print clf.best_estimator_
'''

pred = clf.predict(features_test_sel)

from sklearn.metrics import precision_score, recall_score, accuracy_score

print "accuracy score: ", accuracy_score(labels_test,pred,normalize=True)

print "recall score: ", recall_score(labels_test, pred)
print "precision score: ", precision_score(labels_test, pred)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)