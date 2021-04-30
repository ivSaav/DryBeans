import pandas as pd

beans_data = pd.read_csv('../resources/dataset.xlsx')

def main():
    print("Dry Beans\n", beans_data)
    
if __name__ == "__main__":
    main()