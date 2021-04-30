import matplotlib.pyplot as plt
import seaborn as sb

import pandas as pd

beans_data = pd.read_excel('../resources/dataset.xlsx', na_values=['NA'])

def main():
    #print("Dry Beans\n\n", beans_data)

    #GIVES STATISTICS 
    #print(beans_data.describe())
    #print(beans_data.head())

    sb.pairplot(beans_data.dropna(), hue='Class')
    plt.show()

if __name__ == "__main__":
    main()