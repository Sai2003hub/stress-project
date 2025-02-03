import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def preprocess_image_data(csv_path):
    data = pd.read_csv(csv_path)
    
    # Convert the 'pixels' column from strings to image arrays
    data['pixels'] = data['pixels'].apply(lambda x: np.array(x.split(), dtype='float32').reshape(48, 48, 1))

    # Normalize pixel values to the range [0, 1]
    data['pixels'] = data['pixels'] / 255.0

    # Split data into training, validation, and test sets based on the 'Usage' column
    train_data = data[data['Usage'] == 'Training']
    val_data = data[data['Usage'] == 'PublicTest']
    test_data = data[data['Usage'] == 'PrivateTest']

    # Save preprocessed data
    np.save('C:/Users/Adithya/Downloads/idk1/data/train_data.npy', train_data)
    np.save('C:/Users/Adithya/Downloads/idk1/data/val_data.npy', val_data)
    np.save('C:/Users/Adithya/Downloads/idk1/data/test_data.npy', test_data)
    
    return train_data, val_data, test_data

# Example usage
csv_path = 'C:/Users/Adithya/Downloads/idk1/data/fer2013.csv'
train_data, val_data, test_data = preprocess_image_data(csv_path)

print("Data preprocessing completed.")
