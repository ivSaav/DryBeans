import pandas as pd

import os


beans_data = pd.read_csv('../resources/Dry_Bean_Dataset.xlsx')

def main():
    print("Dry Beans\n", beans_data)
    
if __name__ == "__main__":
    main()