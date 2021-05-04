import matplotlib.pyplot as plt
import seaborn as sb

import pandas as pd


def plot_class_data(ds, class_name, out_file, out_dir='./output/'):
    data = ds.loc[ds['Class'] == f'{class_name}']
    print(f'[!] Saving solution for: {class_name}')
    sb.pairplot(data.dropna(), hue='Class')
    plt.savefig(f'{out_dir}{out_file}.png')
    
def main():
    #print("Dry Beans\n\n", beans_data)

    #GIVES STATISTICS 
    #print(beans_data.describe())
        
    beans_data = pd.read_excel('./resources/dataset.xlsx', na_values=['NA'])
    
    # null values verification
    n = beans_data.isnull().sum().sum()
    if n > 0:
        print(f'[!] {n} values are missing.')
    
    # renaming 'roundness' collumn to be consistent with the other columns
    beans_data.rename(columns = {'roundness' : 'Roundness'}, inplace=True)
    
    print("=== DESCRIBE ===")
    print(beans_data.describe())
    
    # plot_class_data(beans_data, 'SEKER', 'barbunya_orig')
    # plot_class_data(beans_data, 'BARBUNYA', 'barbunya_orig')
    # plot_class_data(beans_data, 'BOMBAY', 'bombay_orig')
    # plot_class_data(beans_data, 'CALI', 'cali_orig')
    # plot_class_data(beans_data, 'HOROZ', 'horoz_orig')
    # plot_class_data(beans_data, 'SIRA', 'sira_orig')
    # plot_class_data(beans_data, 'DERMASON', 'dermason_orig')
    
    # open out file in append mode
    outfile = './resources/processed.csvs'
    f = open(outfile, 'w')
    f.seek(0)
    f.truncate() # delete previous contents
    f.close()
    
    
    # droping any seker rows with a Solidity value less than 0.96 (alll solidity values except one are clustered above 0.96)
    # droping values bellow 0.68 as Seker bean has a round shape and most values are clustered above this value
    # droping ShapeFactor4 values bellow 0.98 as all other values are clustered above this one
    beans_data = beans_data.loc[ (beans_data['Class'] == 'SEKER') &  (beans_data['Roundness'] >= 0.68) & (beans_data['Solidity'] >= 0.96) & (beans_data['ShapeFactor4'] >= 0.98) ]
    
    beans_data.to_csv(outfile)
   
    
    
if __name__ == "__main__":
    main()