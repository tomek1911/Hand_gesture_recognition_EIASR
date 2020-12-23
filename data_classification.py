import csv
import matplotlib.pyplot as plt

class DataClassification:
    """Class provides tools to classify images using features."""
    pass

    def getXyfromCSV(path_features_csv):
        X = []
        y = []

        with open(path_features_csv) as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader, None)
            for row in csv_reader:
                temp_row = row
                X_row = temp_row[:len(temp_row)-1]
                y_row = temp_row[len(temp_row)-1:]
                y_row = y_row[0]
                X.append(X_row)
                y.append(y_row)
        
        return X, y
    
    @staticmethod
    def generate_conf_matrix(matrix, title='Confusion Matrix Hand gesture Recognition' safe=False, matrix_name='confusion_matrix.jpg'):
        fig = plt.figure()
        plt.matshow(confusion_matrices)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicated Label')
        if safe:
            plt.savefig(matrix_name)