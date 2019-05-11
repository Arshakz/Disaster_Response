import sys

import pandas as pd
import numpy as np
from sqlalchemy import create_engine




def load_data(messages_filepath, categories_filepath):
    
    """
    Loads data from 2 different csv files
    
    INPUT:
    messages_filepath: filepath and filename of csv file of messages.
    categories_filepath: filepath and filename of csv file of categories.
       
    OUTPUT:
    df: returnes 1 dataset with is a combination of messages and categories datasets merged on `id` column
    """
        
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')   
    return df


def clean_data(df):
    
    """
    Generally converts categories part of df into numbers
    
    INPUT:
    df: merged dataframe of messages and categoreies (load_data())
       
    OUTPUT:
    df: takes from categories dataframe lest digits and get rid of other parts
    """
    
    # create a dataframe of the 36 individual category columns
    dr=categories['categories'].str.split(';', expand=True).rename(columns = lambda x: "categories_"+str(x+1))
    categories = pd.concat([categories, dr], axis=1, join='outer')
    categories.drop('categories', axis=1, inplace=True)
    
    row = categories.iloc[0]
    
    category_colnames = ['id']

    for k in row.iloc[1:]:
        category_colnames.append(k[:-2])
        
    categories.columns = category_colnames
    
    #categories.drop('id', axis=1, inplace=True)
    for column in categories:
        # set each value to be the last character of the string
        if column!='id':
            categories[column]=categories[column].str[-1]

            # convert column from string to numeric
            categories[column] = categories[column].astype(np.int)
            
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    
    
    df.drop_duplicates(inplace=True)
    
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.merge(df, categories, on='id')
    df.drop('id', axis=1, inplace=True)
    
    return df


def save_data(df, database_filename):
    
    """
    Saves dataframe into SQLite database
    
    INPUT:
    df: merged dataframe of messages and categoreies (clean_data())
    database_filename: database filename where to save dtafrmae
       
    OUTPUT:
    Saves dataframe in the mentioned database filename
    """
        
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql(df, engine, index=False) 


def main():
    
    """
    Loads data from 2 different csv files
    
    INPUT:
    df: merged dataframe of messages and categoreies (load_data())
       
    OUTPUT:
    df: takes from categories dataframe lest digits and get rid of other parts
    """
        
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