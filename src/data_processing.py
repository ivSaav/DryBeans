
import math
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np

def plot_class_data(ds, class_name, out_file, out_dir='./output/'):
    data = ds.loc[ds['Class'] == f'{class_name}']
    print(f'[!] Saving solution for: {class_name}')
    sb.pairplot(data.dropna(), hue='Class')
    plt.savefig(f'{out_dir}{out_file}.png')
    
def violin_plot(data, columns, out_file, out_dir='./output/'):
    plt.figure(figsize=(14, 10))

    if len(columns) == 0: 
        columns = data.columns
    
    plot_cols = math.ceil(math.sqrt(len(columns)))
    plot_rows = max(1, math.ceil(len(columns)/plot_cols))

    for column_index, column in enumerate(columns):
        if column=='Class':
            continue
        plt.subplot(plot_rows, plot_cols, column_index + 1)
        sb.violinplot(x='Class', y=column, data=data)
    plt.savefig(f'{out_dir}{out_file}.png')



#print("Dry Beans\n\n", beans_data)
#GIVES STATISTICS 
#print(beans_data.describe())

#beans_data = pd.read_excel('./resources/dataset.xlsx', na_values=['NA'])  
beans_data = pd.read_excel('./resources/dataset.xlsx', na_values=['NA'], engine='openpyxl')

# null values verification
n = beans_data.isnull().sum().sum()
if n > 0:
    print(f'[!] {n} values are missing.')

# renaming 'roundness' collumn to be consistent with the other columns
beans_data.rename(columns = {'roundness' : 'Roundness'}, inplace=True)

print("=== DESCRIBE ===")
print(beans_data.describe())

# violin_plot(beans_data, [], 'violin_all')
#violin_plot(beans_data, ['Area', 'Eccentricity', 'Roundness', 'Solidity'], 'violin')
# violin_plot(beans_data, ['ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4'], 'violin_shapefactors')

# plot_class_data(beans_data, 'SEKER', 'seker_orig')
# plot_class_data(beans_data, 'BARBUNYA', 'barbunya_orig')
# plot_class_data(beans_data, 'BOMBAY', 'bombay_orig')
# plot_class_data(beans_data, 'CALI', 'cali_orig')
# plot_class_data(beans_data, 'HOROZ', 'horoz_orig')
# plot_class_data(beans_data, 'SIRA', 'sira_orig')
# plot_class_data(beans_data, 'DERMASON', 'dermason_orig')

# date frame t obe written into csv
print(beans_data.columns.insert(0, 'id'))
final_data = pd.DataFrame(columns=beans_data.columns)
cp = beans_data.copy()

data = []
# droping any seker rows with a Solidity value less than 0.96 (alll solidity values except one are clustered above 0.96)
# droping values bellow 0.68 as Seker bean has a round shape and most values are clustered above this value
# droping ShapeFactor4 values bellow 0.98 as all other values are clustered above this one
seker_data = beans_data.loc[ (beans_data['Class'] == 'SEKER') &  (beans_data['Roundness'] >= 0.68) & (beans_data['Solidity'] >= 0.96) & (beans_data['ShapeFactor4'] >= 0.98) ]
data.append(len(seker_data))
final_data = final_data.append(seker_data)

barb_data = beans_data.loc[ (beans_data['Class'] == 'BARBUNYA') &  (beans_data['MinorAxisLength'] < 325)]
data.append(len(barb_data))
final_data = final_data.append(barb_data)


bombay_data = beans_data.loc[ (beans_data['Class'] == 'BOMBAY') &  (beans_data['ShapeFactor2'] < 0.0014)]
data.append(len(bombay_data))
final_data = final_data.append(bombay_data)
# plot_class_data(bombay_data, 'BOMBAY', 'bombay')




cali_data = beans_data.loc[ (beans_data['Class'] == 'CALI') &  (beans_data['Area'] < 110000) & (beans_data['Eccentricity'] > 0.70) & (beans_data['MinorAxisLength'] > 185) ]
data.append(len(cali_data))
final_data = final_data.append(cali_data)

horoz_data = beans_data.loc[ (beans_data['Class'] == 'HOROZ') &  (beans_data['Area'] < 80000)  & (beans_data['ConvexArea'] < 80000) & (beans_data['EquivDiameter'] < 320)]
data.append(len(horoz_data))
final_data = final_data.append(horoz_data)
# plot_class_data(horoz_data, 'HOROZ', 'horoz')

sira_data = beans_data.loc[ (beans_data['Class'] == 'SIRA') &  (beans_data['Area'] < 63000) & (beans_data['Roundness'] > 0.59)] 
data.append(len(sira_data))
final_data = final_data.append(sira_data)
# # fina_data = final_data.append(horoz_data)

derma_data = beans_data.loc[ (beans_data['Class'] == 'DERMASON') &  (beans_data['Perimeter'] < 850) & (beans_data['Roundness'] > 0.59)] 
data.append(len(derma_data))
final_data = final_data.append(derma_data)



def barChart(beans_data, data):
    plt.rcdefaults()
    fig, ax = plt.subplots()

    # Example data
    classes = pd.unique(beans_data.iloc[:, -1])

    print(classes)

    labels = classes
    plt.xticks(range(len(data)), labels)
    plt.xlabel('Class')
    plt.ylabel('Amounts')
    plt.title('Number of beans')
    plt.bar(range(len(data)), data) 
    plt.show()


barChart(beans_data, data)

"""
print("EQUALS ", cp.equals(beans_data))
# open outup file in append mode
outfile = './resources/processed.csv'
f = open(outfile, 'w')
f.seek(0)
f.truncate() # delete previous contents
f.close()

print("FINAL", final_data.head())

final_data.to_csv(outfile, index=False)"""