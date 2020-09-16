import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import sys
import logging

sys.path.append('src')
sys.path.append('src/visualization')

from visualization.visualize import *


def compile_model(n_features):
    model=Sequential() # This is the type of model
    model.add(Dense(12, input_dim=n_features, activation='relu')) # This is the first layer. 12 nodes, with features = the number of variables
    model.add(Dropout(0.2)) # This is to account for overfitting
    model.add(Dense(8, activation='relu')) # This is the 2nd layer
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid')) # Final output layer. Has to have a 1, with sigmoid since 0 and 1 output

    # compile the model after defining it
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # you can Google loss functions
    return model

def fit_model(model, features, labels, n_epochs=10, n_batch=10, val_split=0.1):
    # Fit the actual model
    # Storing it into history for plotting purposes. It will give you the loss, accuracy and the validation accuracy
    history = model.fit(features, labels, epochs=n_epochs, batch_size=n_batch, validation_split=val_split)
    return history


def main(logging):

    # Create the untrained model
    # Model function has been defined, now run the model and store it
    logging.info("#### Compiling the model")
    model = compile_model(8)  # 8 since we are working with 8 features

    # load the data
    logging.info("#### Loading the data")
    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv')

    # Train your model on the data
    logging.info("#### Model fitting")
    history = fit_model(model, X_train, y_train, n_epochs=50, n_batch=30,val_split=0.2)
    loss_plot(history)

    # Storing it as a pickle
    model_path = 'models/stk_model_v1.h5' # h5 = tensorflow model
    history.model.save(model_path)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, filename="stk-cookiecutter-project.log")
    main(logging)









