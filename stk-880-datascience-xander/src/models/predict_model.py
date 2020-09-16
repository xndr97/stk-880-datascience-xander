import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import sys
import logging

# The below lines of code are to invoke the functions from visualization.py script
sys.path.append('src')
sys.path.append('src/visualization')

from visualization.visualize import *

def main(logging):

    # load data
    logging.info("#### Loading Data")
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv')

    # loadding the trained model
    logging.info("#### Loading Model")
    model = load_model('models/stk_model_v1.h5')

    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    logging.info("#### Evaluating the model")
    logging.info("#### Model accuracy {}".format(accuracy))

    y_pred = model.predict_classes(X_test)

    # confusion matrix: Google to see what it is
    conf_matrix = confusion_matrix(y_test, y_pred)
    logging.info("#### Confusion matrix: {}".format(conf_matrix))

    # Now to plot it. The function comes from the visualize.py script
    logging.info("#### Plotting the confusion matrix")
    plot_confusion_matrix(cm=conf_matrix, normalize=True, target_names=['0', '1'], filepath='reports/figures/confusion_matrix.png')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, filename="stk-cookiecutter-project.log")
    main(logging)


