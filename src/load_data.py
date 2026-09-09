import pandas as pd

def load_data(csv_path):
    df = pd.read_csv(csv_path, sep=";")
    return df

if __name__ == "__main__":
    df = load_data("data/raw/student-por.csv")
    print(df.shape)