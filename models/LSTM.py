import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.utils import ACTIVITIES, labels

def transform_data(df, window_size):
    seqs = []    
    for i in range(0, df.shape[0] - window_size, window_size // 3 ):
        seqs.append(np.array(df.iloc[i:i+window_size]))

    return seqs
    

def load_data(suite, window_size=128):
    X = []
    y = []
    print(f"Loading data...")
    for sample in suite.samples:
        # .drop(columns=['time'])
        acc = sample.accelerometer
        acc.rename(columns={"x": "acc_x", "y": "acc_y", "z": "acc_z"}, inplace=True)

        gyro = sample.gyro.drop(columns=['time'])
        gyro.rename(columns={"x": "gyro_x", "y": "gyro_y", "z": "gyro_z"}, inplace=True)

        orient = sample.orient.drop(columns=['time'])

        gravity = sample.gravity.drop(columns=['time'])
        gravity.rename(columns={"x": "gravity_x", "y": "gravity_y", "z": "gravity_z"}, inplace=True)

        V = pd.concat([acc, gyro, orient, gravity], axis=1, join='inner')
        seq_list = transform_data(V, window_size)
        
        X += seq_list
        y += [ACTIVITIES[sample.activity]] * len(seq_list)

    y = np.array(y)
    print("Done\n")
    return np.array(X), pd.get_dummies(y).astype(np.int8)

def train_test_info(X_train, y_train, X_test, y_test):
    data_split = pd.Series({'train': X_train.shape[0], 'test': X_test.shape[0]})

    print('------------------------------')
    print('| Train and test data shapes |')
    print('------------------------------')    
    print('X_train shape: ', X_train.shape)
    print('y_train shape: ', y_train.shape)
    print('X_test shape: ', X_test.shape)
    print('y_test shape: ', y_test.shape)
    print('\n')

    print('----------------------------')
    print('| Train / test split [%] |')
    print('----------------------------')  
    plot = data_split.plot.pie(y='train', figsize=(5, 5))
    plt.show()
    print('\n')
    
    print('----------------------------')
    print('| Class distribution train |')
    print('----------------------------')  
    class_distribution(y_train, labels)
    print('\n')

    print('---------------------------')
    print('| Class distribution test |')
    print('---------------------------')  
    class_distribution(y_test, labels)


def class_distribution(y, class_labels: list):
    classes = pd.from_dummies(y).value_counts()
    classes.rename(index={i: class_labels[i] for i in range(len(class_labels))}, inplace=True)
    print(classes)
    plot = classes.plot.pie(y='chodz', figsize=(5, 5))
    plt.show()