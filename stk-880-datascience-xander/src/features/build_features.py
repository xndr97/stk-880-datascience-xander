import pandas as pd
from sklearn.model_selection import train_test_split
import logging

def main(logging):

    # load the dataset
    logging.info("#### Loading the raw data")
    dataset = pd.read_csv('data/raw/pima-indians-diabetes.csv')

    # split the data into X and y
    X = dataset.iloc[:, 0:8]
    y = dataset.iloc[:, 8]

    # split into train and test using the tuple
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2020)

    # saving the datasets as processed datasets
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, filename="stk-cookiecutter-project.log")
    main(logging)