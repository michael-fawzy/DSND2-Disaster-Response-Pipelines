# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re # To remove punctuation with a regular expression
import pickle # To export model as a pickle file

from sklearn.feature_extraction.text import CountVectorizer # Using Bag of Words to vectorize text
from sklearn.feature_extraction.text import TfidfTransformer # Transform text
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

# download necessary NLTK data
import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

def load_data(database_filepath):
    '''
    Function to load data from database
    Parameters: 
    database_filepath: filepath of the database to loadv

    Returns: 
    x: Series of messages in the dataframe message column
    y: Dataframe of other columns starting from "related" column
    category_names: Names of message categories
    '''
    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('CleanedData', engine)
    X = df['message']
    y = df[df.columns[4:]] #Produces a dataframe containing columns and their values
    category_names = y.columns

    return X, y, category_names


def tokenize(text):
    '''Function to tokenize data'''

    
    # Remove punctuation characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) 

    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove stop words
    tokens = [token for token in tokens if token not in stopwords.words("english")] 

    # Initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():
   '''Function to Build a text processing and machine learning pipeline'''
   pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

   parameters = {'tfidf__use_idf':[True, False],
              'clf__estimator__n_estimators':[10, 20]}
    
   model = GridSearchCV(pipeline, param_grid=parameters)
    

   return model
    
def evaluate_model(model, X_test, y_test, category_names):
    '''Function to train and tune a model using GridSearchCV'''

    y_pred = model.predict(X_test)
    # Convert y-pred to dataframe to get labels and prevent label error 
    y_pred_df= pd.DataFrame(y_pred, columns = category_names)
    for column in y_test.columns:
        print(column)
        print(classification_report(y_test[column], y_pred_df[column], target_names = category_names))
        print('-'*80)
        
        

def save_model(model, model_filepath):
    '''Function to export the final model as a pickle file'''

    pickle.dump(model, open(model_filepath, 'wb'))



def main():
    '''The main function the script runs'''

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

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