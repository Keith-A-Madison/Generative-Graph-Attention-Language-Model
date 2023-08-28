import random
import os

def get_training_data(training_data_dir):
        filenames = []

        for filename in os.listdir(training_data_dir):
                filenames.append(os.path.join(training_data_dir, filename))

        random.shuffle(filenames)

        lines = []

        for filename in filenames:
                with open(filename, "r") as file:
                        for line in file:
                                lines.append(line.strip())

        return lines
