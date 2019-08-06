## Multiclass Prediction
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

train_csvin = pd.read_csv( 'train.csv' )
test_csvin = pd.read_csv( 'test.csv' )

y = train_csvin['label']
X = train_csvin.drop(['label'],axis=1)
X_test = test_csvin

X = X/255.0
X_test = X_test/255.0

nr_samples = 32000

X_train = X[ : nr_samples ]
X_test = X[ nr_samples : 42000 ]
y_train = y[ : nr_samples ]
y_test = y[ nr_samples : 42000 ]

def print_classification_rpt( y_actual, y_pred ):
    print('Classification Report:')
    print( classification_report( y_actual, y_pred ))
    acc = accuracy_score( y_actual, y_pred )
    print( 'Accuracy : ' + str( acc ))

## KNN
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier( n_neighbors = 10 )
knn_clf.fit( X_train, y_train )
y_pred = knn_clf.predict( X_test )
print_classification_rpt( y_test, y_pred )


from sklearn.model_selection import cross_val_score
cross_val_score( sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy')

from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier()
forest_clf.fit( X_train, y_train )
forest_clf.predict( [some_digit] )
forest_clf.classes_
forest_clf.predict_proba( [some_digit] ) #shows list of probablities wrt predictable outcomes

from sklearn.model_selection import cross_val_score
cross_val_score( forest_clf, X_train, y_train, cv=3, scoring='accuracy')


## Training different models on the training set, running it against the test set
from sklearn.model_selection import GridSearchCV

## KNN
from sklearn.neighbors import KNeighborsClassifier
param_grid = {'n_neighbors' : [2,4,6] }
knn_clf = KNeighborsClassifier()
gridsearch = GridSearchCV( knn_clf, param_grid, cv=3 )
gridsearch.fit( X_train, y_train )
gridsearch.best_estimator_
gridsearch.cv_results_
gridsearch.cv_results_

clf = KNeighborsClassifier(n_neighbors=5)
clf.fit( X_train, y_train )
y_pred = clf.predict( X_test )
print( "accuracy: {}".format(accuracy_score(y_test,y_pred)))