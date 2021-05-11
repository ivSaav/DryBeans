import math
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split

def dfs(beans):
    X, Y = beans.iloc[:, :-1], beans.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, train_size=0.8)
    
    clf = tree.DecisionTreeClassifier()
    
    clf = clf.fit(X_test, y_test)
    
    plt.figure(figsize=(20, 20), dpi=500)
    tree.plot_tree(clf, feature_names=beans.columns[:-1].tolist(),class_names=pd.unique(y_test)) 
   
    plt.savefig('tree.png') 
    
    # print(beans.columns[:-1])
    # r = tree.export_text(clf, feature_names=beans.columns[:-1].tolist())
    # print(r)
    print("accuracy: ", clf.score(X, Y))
    

def main():
    beans = pd.read_csv('./resources/processed.csv', na_values=['NA'])
    
    print(beans.head())
    
    n = beans.isnull().sum().sum()
    if n > 0:
        print(f'[!] {n} values are missing.')
    
    dfs(beans)
    
    
   
    
    
if __name__ == "__main__":
    main()