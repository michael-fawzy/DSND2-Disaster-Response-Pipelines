# DSND2-Disaster-Response-Pipelines
2nd project in the Data Scientist Nanodegree

### Summary
A flask web app that depends on an ETL pipeline and a machine learning pipeline to categorize emergency messages based on the needs communicated by the sender. An emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the data.

Disaster data from Figure Eight was used to build a model for an API that classifies disaster messages.

### Project Components
There are three components in this project:

1. ETL Pipeline

In a Python script, process_data.py, a data cleaning pipeline is written to:

- Load the messages and categories datasets
- Merge the two datasets
- Clean the data
- Store it in a SQLite database

2. ML Pipeline

In a Python script, train_classifier.py, a machine learning pipeline was written to:

- Load data from the SQLite database
- Split the dataset into training and test sets
- Build a text processing and machine learning pipeline
- Train and tunes a model using GridSearchCV
- Output results on the test set
- Export the final model as a pickle file

3. Flask Web App

Displays data visualizations using Plotly and accepts user input of messages to be categorized using the Python script run.py.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
