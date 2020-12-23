import csv


class DataClassification:
    """Class provides tools to classify images using features."""
    pass

    def getXyfromCSV(path_features_csv):
        with open(path_features_csv) as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader, None)
            for row in csv_reader:
                temp_row = row
                X = temp_row[:len(temp_row)-1]
                y = temp_row[len(temp_row)-1:]
        return X, y