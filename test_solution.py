import pandas as pd
from solution import PredictionModel

def main():
    print("Initializing PredictionModel...")
    model = PredictionModel()
    
    print("Loading test data...")
    # Load a few rows from train.csv to test the predict function
    df = pd.read_csv('data/raw/mediascope/train.csv').head(10)
    
    print("Predicting...")
    results = model.predict(df[['QueryText']])
    
    print("\nResults:")
    print(results.to_string())

if __name__ == "__main__":
    main()
