import dill
import numpy as np
import re
import spacy
from train import process_string

with open('./models/text_classification_model.pkl', 'rb') as file:
    model = dill.load(file)
with open('./models/vectorizer.pkl', 'rb') as file:
    vectorizer = dill.load(file)
nlp = spacy.load("pt_core_news_sm")

def predict(string):
    labels = ['educação', 'finanças', 'indústrias',
              'orgão público', 'varejo']
    string = process_string(string)
    string_matrix = vectorizer.transform([string])
    predictions = model.predict(string_matrix)
    predicted_categories = np.array(labels)[predictions.astype(bool)[0]]
    return ('Segundo nosso modelo o texto digitado possui maior probabilidade de '
            f'ser da(s) categoria(s): {predicted_categories}')

if __name__ == "__main__":
    sample = ("Programa é voltado para alunos de 14 a 24 anos, regularmente matriculados no ensino médio "
              "da rede pública. Governo vai pagar R$ 2 mil por ano, além de bônus, a quem seguir os critérios "
              "do benefício.")
    prediction_string = predict(sample)
    print(prediction_string)