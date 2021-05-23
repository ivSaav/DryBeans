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

def dtc(X_train, X_test, y_train, y_test):
    clf = tree.DecisionTreeClassifier(max_depth=10, min_samples_leaf=5)
    clf = clf.fit(X_test, y_test)
       
    # generate predictions
    predictions = clf.predict(X_test)  
    print(classification_report(y_test, predictions))  
    
    # save classification report
    report = classification_report(y_test, predictions, output_dict=True)
    print("F1 Score: ", report['weighted avg']['f1-score'])
    
    # show confusion matrix
    fig, ax = plt.subplots(figsize=(14, 10))
    plot_confusion_matrix(clf, X_test, y_test, ax=ax).ax_.set_title('DTC Confusion Matrix')
    plt.show()
    
    return report
    

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
    
    # save classification report
    report = classification_report(y_test, predictions, output_dict=True)
    print("F1 Score: ", report['weighted avg']['f1-score'])
    
    # show confusion matrix
    # fig, ax = plt.subplots(figsize=(14, 10))
    # plot_confusion_matrix(clf, X_test, y_test, ax=ax).ax_.set_title('MLP Confusion Matrix')
    # plt.show()
    
    return report
 
def neural_params_search(X, y):
    scaler = StandardScaler()
    scaler.fit(X)

    X_train = scaler.transform(X)
    # Evaluates and searches for the best mlp parameters
    params = {
                'solver': ['adam', 'sgd'], 
                'alpha': [0.0001, 0.05], 
                'hidden_layer_sizes': [10, 11, 12], 
                'max_iter': [400, 500]
            }
    grid = GridSearchCV(MLPClassifier(), params, scoring='f1_weighted', refit=True, verbose=0, n_jobs=2)
    grid.fit(X_train, y)
    print(grid.best_estimator_)
    print(grid.best_params_)

     
# Support vector classification
def svc(X_train, X_test, y_train, y_test, kernel="linear"):
    print("SVC ====")
    print("kernel used: ", kernel)
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    clf = svm.SVC(kernel=kernel, cache_size=500, C=10, gamma=0.001)
    clf.fit(X_train, y_train)

    # generate predictions
    predictions = clf.predict(X_test)
    print(classification_report(y_test, predictions))
    
    # save classification report
    report = classification_report(y_test, predictions, output_dict=True)
    print("F1 Score: ", report['weighted avg']['f1-score'])
    
    # # show confusion matrix
    # fig, ax = plt.subplots(figsize=(14, 10))
    # plot_confusion_matrix(clf, X_test, y_test, ax=ax).ax_.set_title('SVC Confusion Matrix')
    # plt.show()
    
    return report
    
def svc_params_search(X, y):
    
    scaler = StandardScaler()
    scaler.fit(X)
    
    X_train = scaler.transform(X)
    
    # Evaluates and searches for the best svc parameters
    params = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}
    grid = GridSearchCV(svm.SVC(),params, scoring='f1_weighted', refit=True, verbose=0, n_jobs=2)
    grid.fit(X, y)
    print(grid.best_estimator_)
    print(grid.best_params_)


#Nearest Neighbors Classification
def knn(X, y, X_train, X_test, y_train, y_test):
    print("KNN ====")
    neigh = KNeighborsClassifier(n_neighbors=6, leaf_size=50)
    neigh.fit(X_train, y_train)

    predictions = neigh.predict(X_test)
    print(classification_report(y_test, predictions))

    # plot_confusion_matrix(neigh, X_test, y_test)
    # plt.show()


def compare_class_results(reports, model_names):
    
    dic = { 'Model' : [], 'Class' : [], 'Score':[]}
    
    # extracting relevant data from each model report
    for modelName, rep in zip(model_names, reports): 
        for i, (key, values) in enumerate(rep.items()):
            if (i > 6):
                break;
            dic['Model'].append(modelName)
            dic['Class'].append(key)
            dic['Score'].append(values["f1-score"])
      
    df = pd.DataFrame (dic, columns = ['Model','Class', 'Score'])
    
    plt.figure(figsize=(14,10))
    sb.barplot(x='Class', y='Score', hue='Model', data=df)
    plt.legend(title="Classifier", bbox_to_anchor=(1.05, 0.6), loc=2, borderaxespad=0.)
    plt.ylabel("F1-Score", size=14)
    plt.xlabel("Bean Class", size=14)
    plt.title("Score Analysis By Class", size=18)
    plt.show()
    
def compare_overall_results(reports, model_names):
    
    dic = { 'Model' : [], 'Score':[]}
    
    # extracting relevant data from each model report
    for modelName, rep in zip(model_names, reports): 
        dic['Model'].append(modelName)
        dic['Score'].append(rep['weighted avg']["f1-score"])
      
    df = pd.DataFrame (dic, columns = ['Model','Score'])
    
    plt.figure(figsize=(14,10))
    sb.barplot(x='Model', y='Score', data=df)
    # plt.legend(title="Classifier", bbox_to_anchor=(1.05, 0.6), loc=2, borderaxespad=0.)
    plt.ylabel("Weighted F1-Score", size=14)
    plt.xlabel("Classifier", size=14)
    plt.title("Classifier Weighted F1-Score Analysis", size=18)
    plt.show()
        
        
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
   

    dtc_report = dtc(X_train, X_test, y_train, y_test)
     
    neural_report = neural(X_train, X_test, y_train, y_test)

    #svc(beans, X, y, X_train, X_test, y_train, y_test, "linear")
    svc_report = svc(X_train, X_test, y_train, y_test, "rbf")
    # svc_params_search(X, y)

    # compare_class_results([neural_report, svc_report, dtc_report], ['MLP', 'SVC', 'DTC'] )
    # neural_params_search(X_train, y_train)
    # knn(X,y, X_train, X_test, y_train, y_test)

    compare_overall_results([neural_report, svc_report, dtc_report], ['MLP', 'SVC', 'DTC'])
    
if __name__ == "__main__":
    main()