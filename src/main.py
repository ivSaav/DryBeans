import math
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

def dfs(beans, X, y, X_train, X_test, y_train, y_test):
    clf = tree.DecisionTreeClassifier()
    
    clf = clf.fit(X_test, y_test)
    
    plt.figure(figsize=(30, 30), dpi=500)
    tree.plot_tree(clf, feature_names=beans.columns[:-1].tolist(),class_names=pd.unique(y_test)) 
   
    plt.savefig('tree.png') 
    
    # print(beans.columns[:-1])
    # r = tree.export_text(clf, feature_names=beans.columns[:-1].tolist())
    # print(r)
    print("accuracy: ", clf.score(X, y))
    

def neural(beans, X,y, X_train, X_test, y_train, y_test):
    clf = MLPClassifier().fit(X_train, y_train)
    clf.score(X_test, y_test)  
    print("accuracy: ", clf.score(X, y))  
    
    

def main():
    beans = pd.read_csv('./resources/processed.csv', na_values=['NA'])
    
    print(beans.head())
    
    n = beans.isnull().sum().sum()
    if n > 0:
        print(f'[!] {n} values are missing.')

    # extract values and class labels
    X, y = beans.iloc[:, :-1], beans.iloc[:, -1]
    # split values into testing and training datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8)
    
    # dfs(beans, X, y, X_train, X_test, y_train, y_test)
     
    neural(beans, X,y, X_train, X_test, y_train, y_test)
   
    
    
if __name__ == "__main__":
    main()