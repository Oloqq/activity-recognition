import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.utils import ACTIVITIES, labels, plot_confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score


def transform_data(df, window_size=128):
    result_df = pd.DataFrame()
    columns_to_process = ['acc_z', 'acc_y', 'acc_x', 'gyro_z', 'gyro_y', 'gyro_x', 'qz', 'qy', 'qx',
                      'qw', 'roll', 'pitch', 'yaw', 'gravity_x', 'gravity_y', 'gravity_z']

    for column in columns_to_process:
        result_df[f'mean_{column}'] = df[column].rolling(window=window_size).mean()
        result_df[f'max_{column}'] = df[column].rolling(window=window_size).max()
        result_df[f'min_{column}'] = df[column].rolling(window=window_size).min()
        result_df[f'std_{column}'] = df[column].rolling(window=window_size).std()

    result_df = result_df.dropna()
    return result_df

def shuffle_data(X, y):
    print('Shuffling data...')
    indexes = np.arange(len(y))
    np.random.shuffle(indexes)
    x_shuffle, y_shuffle = X.iloc[indexes].reset_index(drop=True), y[indexes]
    
    print('Done\n')
    return x_shuffle, y_shuffle

def load_data(suite, window_size=128, shuffle=True):
    X = pd.DataFrame()
    y = []
    print(f"Loading data...")
    for sample in suite.samples:
        acc = sample.accelerometer.drop(columns=['time'])
        acc.rename(columns={"x": "acc_x", "y": "acc_y", "z": "acc_z"}, inplace=True)

        gyro = sample.gyro.drop(columns=['time'])
        gyro.rename(columns={"x": "gyro_x", "y": "gyro_y", "z": "gyro_z"}, inplace=True)

        orient = sample.orient.drop(columns=['time'])

        gravity = sample.gravity.drop(columns=['time'])
        gravity.rename(columns={"x": "gravity_x", "y": "gravity_y", "z": "gravity_z"}, inplace=True)

        V = pd.concat([acc, gyro, orient, gravity], axis=1, join='inner')
        result = transform_data(V, window_size)

        X = pd.concat([X, result])
        y += [ACTIVITIES.get(sample.activity)] * result.shape[0]
    X.set_index(np.arange(X.shape[0]), inplace=True)
    y = np.array(y)

    if shuffle:
        X, y = shuffle_data(X, y)


    print("Done\n")
    return X, np.array(y)



def perform_model(model, X_train, y_train, X_test, y_test, class_labels, cm_normalize='true', print_cm=True, cm_cmap=plt.cm.Greens):

    results = dict()

    # train the model
    print('training the model..')
    model.fit(X_train, y_train)
    print('Done \n')

    # predict test data
    print('Predicting test data')
    y_pred = model.predict(X_test)
    print('Done \n')
    results['predicted'] = y_pred


    # calculate overall accuracty of the model
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    results['accuracy'] = accuracy
    print("------------")
    print('| Accuracy |')
    print("------------")
    print(f'{round(accuracy * 100, 2)}\n')


    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    results['confusion_matrix'] = cm
    if print_cm:
        print('--------------------')
        print('| Confusion Matrix |')
        print('--------------------')

    plot_confusion_matrix(model, X_test, y_test, class_labels, cm_normalize, cm_cmap)

    # get classification report
    print('-------------------------')
    print('| Classifiction Report |')
    print('-------------------------')
    classification__report = classification_report(y_test, y_pred)
    results['classification_report'] = classification__report
    print(classification__report)

    results['model'] = model

    return results

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
    classes = [class_labels[i] for i in y]
    classes = pd.Series(classes).value_counts()
    print(classes)
    plot = classes.plot.pie(y='chodz', figsize=(5, 5))
    plt.show()
