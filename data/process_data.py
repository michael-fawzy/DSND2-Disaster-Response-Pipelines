import sys
import pandas as pd
from sqlalchemy import create_engine
 

def load_data(messages_filepath, categories_filepath):
    
    '''
    Function to load data from csv files into separate dataframes, merge the dataframes, and return a single dataframe

    Parameters: 
    messages_filepath: path of disaster_messages.csv
    categories_filepath: path of disaster_categories.csv

    Returns: 
    df: Merged dataframe 
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id', how='left')
    return df 



def clean_data(df):
    ''' 
    Function to clean data from previous dataframe
    
    Parameters: 
    df: The merged dataframe resulting from load_data function

    Returns: 
    df: cleaned dataframe 
    '''
    categories = df['categories'].str.split(';', expand=True) # Expand the splitted strings into separate columns.
    row = categories.loc[0]
    # use this row to extract a list of new column names for categories.
    category_colnames = row.tolist()
    # rename the columns of `categories`
    categories.columns = category_colnames
    categories.columns = categories.columns.str[:-2]
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    df = df.drop('categories', axis=1)
    df= pd.concat([df, categories], axis=1)
    df = df.drop_duplicates()
    df = df[df.related != 2] # Drop rows containing 2 in 'related' column after merging the dataframes to keep only binary (0,1) values
    return df 

        
def save_data(df, database_filename):
    
    '''
    Function to save the cleaned dataframe into a SQL database

    Parameters: 
    df: Cleaned dataframe from clean_data function
    database_filename: filepath of the database to save the cleaned data to
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('CleanedData', engine, index=False,  if_exists='replace') # if_exists='replace' Prevents ValueError: Table 'CleanedData' already exists. 


def main():
    '''The main function the script runs'''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()