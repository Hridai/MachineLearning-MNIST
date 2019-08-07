## Multiclass Prediction
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
import seaborn as sns

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

def plot_confusion_matrix( y_actual, y_pred ):
    print('Confusion Matrix:')
    c_mtx = confusion_matrix( y_actual, y_pred )
    fig, ax = plt.subplots( figsize=(6,6) )
    sns.heatmap( c_mtx, annot=True, fmt='d', linewidths=.5, cbar = False, ax=ax )
    plt.ylabel('True Label')
    plt.xlabel('Pred Label')

def get_best_score( gridsearchcvObj ):
    print(gridsearchcvObj.best_score_)
    print(gridsearchcvObj.best_params_)
    print(gridsearchcvObj.best_estimator_)
    return gridsearchcvObj.best_score_

## Logistic Regression
from sklearn.linear_model import LogisticRegression
lin_clf = LogisticRegression(random_state=0)
param_grid = { 'C' : [0.040, 0.050, 0.060],
              'multi_class' : ['multinomial'],
              'penalty' : ['l1'],
              'solver' : ['saga'],
              'tol' : [0.1]}
lin_gs = GridSearchCV( lin_clf, param_grid, verbose = 1, cv = 5 )
lin_gs.fit( X_train, y_train )
lin_score_grid = get_best_score( lin_gs )
y_pred = lin_gs.predict( X_test )
print_classification_rpt( y_test, y_pred )
plot_confusion_matrix( y_test, y_pred )

## KNN
# Takes a while so have removed GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier( n_neighbors = 10 )
knn_clf.fit( X_train, y_train )
y_pred = knn_clf.predict( X_test )
print_classification_rpt( y_test, y_pred )
plot_confusion_matrix( y_test, y_pred )

## Random Forest
from sklearn.ensemble import RandomForestClassifier
rfo_clf = RandomForestClassifier( random_state=0 )
param_grid = { 'max_depth' : [ 15 ],
              'max_features' : [100],
              'min_samples_split' : [5],
              'n_estimators' : [50]}
rfo_gs = GridSearchCV( rfo_clf, param_grid, verbose=1, cv=5 )
rfo_gs.fit( X_train, y_train )
get_best_score( rfo_gs )
y_pred = rfo_gs.predict( X_test )
print_classification_rpt( y_test, y_pred )
plot_confusion_matrix( y_test, y_pred )

## Support Vector Machine
# Takes a while so have removed GridSearchCV
from sklearn.svm import SVC
svc_clf = SVC( C=5, gamma=0.05, kernel='rbf', random_state=0 )
svc_clf.fit( X_train, y_train )
y_pred = svc_clf.predict( X_test )
print_classification_rpt( y_test, y_pred )
plot_confusion_matrix( y_test, y_pred )




### Prev work
from sklearn.model_selection import cross_val_score
cross_val_score( sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy')

from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier()
forest_clf.fit( X_train, y_train )
forest_clf.predict( [some_digit] )
forest_clf.classes_
forest_clf.predict_proba( [some_digit] ) #shows list of probablities wrt predictable outcomes

