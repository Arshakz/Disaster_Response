1. Installations


Analytics of the dataset was done with Python 3.6.4

Modules which were used:
  .Pandas
  .Numpy
  .Sickit-Learn
  .nltk
  .re
  .statisticks
  .pickle
  .sqlalchemy
  
It is better to have Anaconda installed in your computer, from which you will open Jupyter IDE and run syntac there.

-------------------------------------------------------
2. Project Motivation


Dataset was provided by Figure8. Dataset includes messages and message categories connected to disasters, during which pople send tousands of messages, emails, posts asking for help, asking for differnett stuff. I prepared a machine learning process which clean data and understand what people needs. It will help companies, organizations understand what kind of support to deliver to which place and of courese they will be able quickly reponse and prevent further disasters. 


-------------------------------------------------------
3. File Descriptions
-------------------------------------------------------

There are 3 files
- app
 - templates
   - go.html
   - master.html
 - run.py

-data
 -disaster_categories.csv
 -disaster_messages.csv
 -process_data.py

-models
 -train_classifier.py

app folder contains files to show results of analyzes
data folder conatins files with datasets and the cleaing and joining part of it
models folder containf file which analyze dataset and output data


-------------------------------------------------------
4. How to Interact with your project

4.1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

4.2. Run the following command in the app's directory to run your web app.
    `python run.py`

4.3. Go to http://0.0.0.0:3001/



5. Licensing, Authors, Acknowledgements, etc.
-------------------------------------------------------


Analyzes was done by Arshak Zakaryan for the project of Udacity.com.
You can use this research for your own use.
