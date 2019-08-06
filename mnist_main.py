## "The Hello World" of classification tasks: MNIST.

from sklearn.datasets import fetch_openml
import matplotlib
import matplotlib.pyplot as plt, numpy as np

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc="upper left", fontsize=16)
    plt.ylim([0, 1])

def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)

mnist = fetch_openml('MNIST_784')
X = mnist['data']
y = mnist['target']

## visualise digit(s)
target_index = 11
some_digit = X[target_index]
plt.imshow(some_digit.reshape(28,28),cmap=matplotlib.cm.binary )
plt.axis('off')
plt.title( 'target = ' + y[target_index] )
plt.show


## training/test split
X_train, X_test, y_train, y_test = X[:60000], X[:10000], y[:60000], y[:10000]
randomise_set = np.random.permutation(60000)
X_train, y_train = X_train[randomise_set], y_train[y_train]
y_train = y_train.astype(np.int8)


## binary classifier
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42,max_iter=5,tol=-np.infty)
sgd_clf.fit(X_train,y_train_5)
sgd_clf.predict([some_digit])


## cross-validation
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
skfolds = StratifiedKFold(random_state = 42, n_splits=3)
for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]
    
    clone_clf.fit( X_train_folds, y_train_folds )
    y_pred = clone_clf.predict( X_test_fold )
    n_correct = sum( y_pred == y_test_fold )
    print( n_correct / len( y_pred ))
# You can do the above loop in one line of code per the below.
from sklearn.model_selection import cross_val_score
cross_val_score( sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy')


## prediction matrix, plotting precision/recall graph
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve
confusion_matrix(y_train_5,y_train_pred) 
precision = precision_score(y_train_5, y_train_pred) # True Positives / (True Positives + False Positives)
recall = recall_score(y_train_5, y_train_pred) # True Positives / (True Positives + False Negatives)
f1score = f1_score(y_train_5,y_train_pred)
print( 'Precision: ', precision )
print( 'Recall: ', recall )
print( 'F1score = ', f1score)

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method='decision_function')
precisions, recalls, thresholds = precision_recall_curve(y_train_5,y_scores)
plot_precision_recall_vs_threshold(precisions,recalls,thresholds)

plot_precision_vs_recall(precisions,recalls)
plt.show()

fpr, tpr, thresholds = roc_curve( y_train_5, y_scores )
plot_roc_curve(fpr,tpr)
plt.show()


## Multiclass Prediction
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier()
forest_clf.fit( X_train, y_train )
forest_clf.predict( [some_digit] )
forest_clf.classes_
forest_clf.predict_proba( [some_digit] ) #shows list of probablities wrt predictable outcomes

from sklearn.model_selection import cross_val_score
cross_val_score( forest_clf, X_train, y_train, cv=3, scoring='accuracy')


## Training different models on the training set, running it against the test set
