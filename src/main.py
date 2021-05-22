import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import plot_confusion_matrix, classification_report, f1_score



from sklearn import svm
from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors, datasets

from sklearn.model_selection import StratifiedShuffleSplit

def dfs(beans, X, y, X_train, X_test, y_train, y_test):
    print("DFS ====")
    clf = tree.DecisionTreeClassifier(max_depth=10, min_samples_leaf=5)
    
    clf = clf.fit(X_test, y_test)
    
    plt.figure(figsize=(30, 30), dpi=500)
    tree.plot_tree(clf, feature_names=beans.columns[:-1].tolist(),class_names=pd.unique(y_test)) 
   
    predictions = clf.predict(X_test)  
    print("\nClassification Report ========= ")
    print(classification_report(y_test, predictions))  
    # print(beans.columns[:-1])
    # r = tree.export_text(clf, feature_names=beans.columns[:-1].tolist())
    # print(r)
    print("accuracy: ", clf.score(X_test, y_test))
    

def neural(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    
    clf = MLPClassifier(solver='adam', alpha=0.001, hidden_layer_sizes=11, max_iter=400)
    clf.fit(X_train, y_train)
   
    # generate predictions
    predictions = clf.predict(X_test)
    print(classification_report(y_test, predictions))
    
    score = f1_score(y_true=y_test, y_pred=predictions, average='weighted')
    print("F1-Score: ",  f1_score(y_true=y_test, y_pred=predictions, average='weighted'))
    return score
    
    # show confusion matrix
    # plot_confusion_matrix(clf, X_test, y_test)
    # plt.show()
 
def neural_params_search(X, y):
    scaler = StandardScaler()
    scaler.fit(X)

    X_train = scaler.transform(X)
    # Evaluates and searches for the best mlp parameters
    params = {
                'solver': ['adam', 'sgd'], 
                'alpha': [0.0001, 0.05], 
                'hidden_layer_sizes': [8, 9, 10, 11, 12], 
                'max_iter': [300, 500]
            }
    grid = GridSearchCV(MLPClassifier(), params, scoring='f1_weighted', refit=True, verbose=0, n_jobs=2)
    grid.fit(X_train, y)
    grid.best_estimator_
    grid.best_params_

     
# Support vector classification
def svc(beans, X, y, X_train, X_test, y_train, y_test, kernel="linear"):
    print("SVC ====")
    print("kernel used: ", kernel)
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    clf = svm.SVC(kernel=kernel, cache_size=500, C=10, gamma=0.001)
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)
    print(classification_report(y_test, predictions))

    # plot_confusion_matrix(clf, X_test, y_test)
    # plt.show()
    
def svc_params_search(X, y):
    # Evaluates and searches for the best svc parameters
    params = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}
    grid = GridSearchCV(svm.SVC(),params, refit=True, verbose=2)
    grid.fit(X, y)
    print(grid.best_estimator_)



#Nearest Neighbors Classification
def knn(X, y, X_train, X_test, y_train, y_test):
    print("KNN ====")
    neigh = KNeighborsClassifier(n_neighbors=6, leaf_size=50)
    neigh.fit(X_train, y_train)

    predictions = neigh.predict(X_test)
    print(classification_report(y_test, predictions))

    # plot_confusion_matrix(neigh, X_test, y_test)
    # plt.show()


def main():
    beans = pd.read_csv('./resources/processed.csv', na_values=['NA'])
    
    print(beans.head())
    
    n = beans.isnull().sum().sum()
    if n > 0:
        print(f'[!] {n} values are missing.')

    # extract values and class labels
    X, y = np.array(beans.iloc[:, :-1]), np.array(beans.iloc[:, -1])
    # split values into testing and training datasets
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=0)
    
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, train_size=0.8, random_state=0)
    
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
   

    #dfs(beans, X, y, X_train, X_test, y_train, y_test)
     
    # neural(X_train, X_test, y_train, y_test)

    #svc(beans, X, y, X_train, X_test, y_train, y_test, "linear")
    #svc(beans, X, y, X_train, X_test, y_train, y_test, "rbf")
    #svc_params_search(X, y, X_train, X_test, y_train, y_test)

    neural_params_search(X_train, y_train)
    # knn(X,y, X_train, X_test, y_train, y_test)

    
    
if __name__ == "__main__":
    main()