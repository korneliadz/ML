from colorama import Fore, Back, Style

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def compute_and_plot_confusion_matrix(y_test, y_pred):
    # Create a confusion matrix that compares `y_test` and predicted values
    conf_mat = confusion_matrix(y_test, y_pred)

    # Create a dataframe for a array-formatted Confusion matrix
    # Assign corresponding names to labels
    confusion_mat_df = pd.DataFrame(conf_mat,
                         index = ['Alive', 'Kicked the bucket'], 
                         columns = ['Alive', 'Kicked the bucket'])

    #Plot the confusion matrix
    plt.figure(figsize = (5,4))
    sns.heatmap(confusion_mat_df, annot = True)
    plt.title('Alive/Dead Confusion Matrix')
    plt.ylabel('True Values')
    plt.xlabel('Predicted Values')
    plt.show()
    
    return conf_mat


def classification_report_print(classifier):
    print(f"Average Accuracy of SVM: {classifier.accuracy:.3f}")
    print(f"Average F1 Score of SVM: {classifier.f1:.3f}")
    print(Back.BLACK + Fore.YELLOW +
        "{:^{width}}".format("Classification Report of SVM:", width=70) 
        + Back.RESET + Fore.WHITE)
    print(classifier.report)