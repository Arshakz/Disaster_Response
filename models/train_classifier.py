import sys

import numpy as np
import pandas as pd
import sqlite3
from sqlalchemy import create_engine

import re
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from statistics import mean
import pickle

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier

import nltk
nltk.download(['punkt', 'wordnet'])



def load_data(database_filepath):
    """
    Extractes data from SQLite database
    
    INPUT:
    database_filepath: DataBase path and name
       
    OUTPUT:
    X: Message column from the table.
    Y: all columns besides Message from the table
    category_names: column names
    
    """
        
    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('InsertTableName', engine)
    df.drop(['original', 'genre'], axis=1, inplace=True)
    X = df['message']
    Y = df.loc[:, df.columns != 'message']
    category_names=Y.columns
    return X, Y, category_names


def tokenize(text):
     """
    Tokenizes data
    
    INPUT:
    text; will take message column from database
       
    OUTPUT:
    clean_tokens: tokenized, stripped, converted into lowercase.... text
    """
    
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    
    """
    Build Model / pipeline

    OUTPUT:
    cv: GridSearchCV model with parapeters including in pipeline
    """
        
        
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                          ('tfidf', TfidfTransformer()),
                          ('clf', MultiOutputClassifier(AdaBoostClassifier()))])
    parameters = {
    'vect__ngram_range': ((1, 1), (1, 2)),
    'clf__estimator__n_estimators':[10,20]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def myScores(y_test, y_pred, category_names):
        
    """
    Evaluates built model and returnes scores
    
    INPUT:
    X_test: test part of Message column
    Y_test: all coulms from tabel besides Message column
    category_name: column name of tables stored in db
       
    OUTPUT:
    results: outputs Precission, Recall and f1 scores
    """
        
    results = {}
    numb=0
    for col in category_names:
        myWeights=precision_recall_fscore_support(y_test[col], y_pred[:,numb], average='weighted')
        results[col]={}
        results[col]['precision']=myWeights[0]
        results[col]['recall']=myWeights[1]
        results[col]['f1_score']=myWeights[2]
        numb+=1
    return results



def evaluate_model(model, X_test, Y_test, category_names):
    
    """
    Evaluates buil model and returnes scores
    
    INPUT:
    model: is the output of the build_model() function
    X_test: test part of Message column
    Y_test: all coulms from tabel besides Message column
    category_name: column name of tables stored in db
       
    OUTPUT:
    totalWeights(totalResults): outputs Precission, Recall and f1 scores
    """
        
        
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_pred = model.predict(X_test)
    totalResults=myScores(y_test, y_pred, category_names)
    return totalWeights(totalResults)




def save_model(model, model_filepath):
        
    """
    Saves model in pickle format
    
    INPUT:
    model: is the output of the build_model() function
    model_filepath: path and name where to save db in pickle format
    
    OUTPUT:
    Saves the model
    """
    # Pickle best model
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()