import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

DATA_PATH = ".data/dataset.csv"

def open_data():
    df = pd.read_csv(DATA_PATH)
    train, test = train_test_split(df, test_size=0.2)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    train["sentence"] = train["sentence"].str.strip().str.lower()
    test["sentence"] = test["sentence"].str.strip().str.lower()
    return train, test

def preprocessing():
    train, test = open_data()
    # OneHotEncoding of target variables
    train["category"] = train["category"].str.split(",")
    test["category"] = test["category"].str.split(",")
    mlb = MultiLabelBinarizer()
    one_hot_encoded_train = mlb.fit_transform(train['category'])
    one_hot_train_df = pd.DataFrame(one_hot_encoded_train, columns=mlb.classes_)
    train = pd.concat([train, one_hot_train_df], axis=1).drop('category', axis=1)
    one_hot_encoded_test = mlb.transform(test['category'])
    one_hot_test_df = pd.DataFrame(one_hot_encoded_test, columns=mlb.classes_)
    test = pd.concat([test, one_hot_test_df], axis=1).drop('category', axis=1)
    # Format sentences
    train["sentence"] = train["sentence"].str.strip().str.lower()
    test["sentence"] = test["sentence"].str.strip().str.lower()