import math
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import plot_confusion_matrix, classification_report

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
    

def neural(beans, X, y, X_train, X_test, y_train, y_test):
    
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    clf = MLPClassifier(solver='adam', max_iter=500)
    clf.fit(X_train, y_train)
    
    # generate predictions
    predictions = clf.predict(X_test)
    print("\nClassification Report ========= ")
    print(classification_report(y_test, predictions))
    
    # score model
    print("Score: ", clf.score(X_test, y_test))  
    
    # show confusion matrix
    plot_confusion_matrix(clf, X_test, y_test)
    plt.show()
     
    
    

def main():
    beans = pd.read_csv('./resources/processed.csv', na_values=['NA'])
    
    print(beans.head())
    
    n = beans.isnull().sum().sum()
    if n > 0:
        print(f'[!] {n} values are missing.')

    # extract values and class labels
    X, y = beans.iloc[:, :-1], beans.iloc[:, -1]
    # split values into testing and training datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=5)
    
    # dfs(beans, X, y, X_train, X_test, y_train, y_test)
     
    neural(beans, X, y, X_train, X_test, y_train, y_test)
   
    
    
if __name__ == "__main__":
    main()