import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, hamming_loss,
                             jaccard_score, precision_score, recall_score)
from sklearn.model_selection import KFold, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.tree import DecisionTreeClassifier
import re
import spacy
import dill
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

DATA_PATH = ".data/dataset.csv"

def open_data():
    df = pd.read_csv(DATA_PATH)
    train, test = train_test_split(df, test_size=0.2)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    train["sentence"] = train["sentence"].str.strip().str.lower()
    test["sentence"] = test["sentence"].str.strip().str.lower()
    return train, test

nlp = spacy.load("pt_core_news_sm")
def process_string(string):
    string = string.strip().lower()
    string = re.sub(r'[^\w\s]', '', string)
    string = " ".join([token.lemma_ for token in nlp(string)])
    return re.sub(' +', ' ', string)

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
    # Preprocessing string
    train["sentence"] = train["sentence"].apply(process_string)
    test["sentence"] = test["sentence"].apply(process_string)
    labels = ['educação', 'finanças', 'indústrias',
              'orgão público', 'varejo']
    X_train = train.sentence
    X_test = test.sentence
    y_train = train[labels]
    y_test = test[labels]
    stop_words_pt = stopwords.words('portuguese')
    vectorizer = TfidfVectorizer(stop_words=stop_words_pt)
    X_train_matrix = vectorizer.fit_transform(X_train)
    X_test_matrix = vectorizer.transform(X_test)
    with open('./models/vectorizer.pkl', 'wb') as file:
        dill.dump(vectorizer, file)
    return X_train_matrix, y_train, X_test_matrix, y_test

def training():
    X_train_matrix, y_train, X_test_matrix, y_test = preprocessing()
    kf = KFold(n_splits=5)
    accuracy_list = []
    hamming_list = []
    for i, (train_index, test_index) in enumerate(kf.split(X_train_matrix)):
        X_tr, X_te = X_train_matrix[train_index], X_train_matrix[test_index]
        y_tr, y_te = y_train.loc[train_index], y_train.loc[test_index]
        print(f"Training in Fold: {i}")
        clf = MultiOutputClassifier(DecisionTreeClassifier()).fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        accuracy = round(accuracy_score(y_te, y_pred), 2)
        hamming = round(hamming_loss(y_te, y_pred), 2)
        accuracy_list.append(accuracy)
        hamming_list.append(hamming)
        final_model = clf
        if hamming < min(hamming_list):
            final_model = clf
        print('Accuracy Score: ', accuracy)
        print('Hamming Loss: ', hamming)
        print("*" * 25)
    predictions = final_model.predict(X_test_matrix)
    accuracy = round(accuracy_score(y_test, predictions), 2)
    hamming = round(hamming_loss(y_test, predictions), 2)
    print('Accuracy Score on the validation set: ', accuracy)
    print('Hamming Loss on the validation set: ', hamming)
    accuracy = np.round(accuracy_score(y_test, predictions), 4)
    hamming = np.round(hamming_loss(y_test, predictions), 4)
    precision = np.round(precision_score(y_test, predictions, average='macro'), 4)
    recall = np.round(recall_score(y_test, predictions, average='macro'), 4)
    f1 = np.round(f1_score(y_test, predictions, average='macro'), 4)
    jaccard = np.round(jaccard_score(y_test, predictions, average='macro'), 4)

    print(f"Accuracy: {accuracy}\nHamming Loss: {hamming}\nPrecision: {precision}\n"
          f"Recall: {recall}\nF1 Score: {f1}\nJaccard Score: {jaccard}")
    with open('./models/text_classification_model.pkl', 'wb') as file:
        dill.dump(final_model, file)
