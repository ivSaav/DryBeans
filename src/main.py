import matplotlib.pyplot as plt
import seaborn as sb

import pandas as pd


def main():
    #print("Dry Beans\n\n", beans_data)

    #GIVES STATISTICS 
    #print(beans_data.describe())
    
    beans_data = pd.read_excel('./resources/dataset.xlsx', na_values=['NA'])
    
    print(beans_data.head())
    
    # select SEKER class beans
    seker_data = beans_data.loc[beans_data['Class'] == 'SEKER']

    sb.pairplot(seker_data.dropna(), hue='Class')
    plt.savefig('out.png')

if __name__ == "__main__":
    main()