import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

def getRedWineQualityDataset():
    names=[
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
        "quality",
    ]
    df = pd.read_csv('Data/winequality-red.csv', header=None, delimiter=',', names = names)
    print('rows, columns:', df.shape)
    df = df.convert_objects(convert_numeric=True)
    df = df.loc[1:, names]
    df.hist()
    plt.show()
    X = df.loc[1:, list(set(names) - set(['quality']))].values
    y = df.loc[1:, 'quality'].astype(np.int64).values
    return (df, X, y)


def getWhiteWineQualityDataset():
    names=[
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
        "quality",
    ]

    # df = pd.read_csv('winequality-white.csv', header=None, delimiter=';', names = names)
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv', header=None, delimiter=';', names = names)

    print('rows, columns:', df.shape)
    df = df.convert_objects(convert_numeric=True)
    df = df.loc[1:, names]
    df.hist()
    # plt.show()
    X = df.loc[1:, list(set(names) - set(['quality']))].values
    y = df.loc[1:, 'quality'].astype(np.int64).values
    return (df, X, y)


def getAbaloneDataset():
    names = [
        'sex', 
        'length', 
        'diameter', 
        'height', 
        'whole weight', 
        'suched weight', 
        'viscera weight', 
        'shell weight', 
        'rings',
    ]
    df = pd.read_csv(
        'http://mlr.cs.umass.edu/ml/machine-learning-databases/abalone/abalone.data', 
        header=None, 
        names=names,
    )
    df.hist()
    # plt.show()
    df.loc[:, 'sex'] = LabelEncoder().fit_transform(df.loc[:, 'sex'])
    X = df.loc[:, list(set(names) - set(['rings']))].values
    y = df.loc[:, 'rings'].values
    return (df, X, y)

def plot_curve(train_scores, test_scores, sizes, plt_range=[0.8, 1.0]):
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(sizes, train_mean,
             color='blue', marker='o',
             markersize=5, label='training accuracy')

    plt.fill_between(sizes,
                     train_mean + train_std,
                     train_mean - train_std,
                     alpha=0.15, color='blue')

    plt.plot(sizes, test_mean,
             color='green', linestyle='--',
             marker='s', markersize=5,
             label='validation accuracy')

    plt.fill_between(sizes,
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, color='green')

    plt.grid()
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim(plt_range)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df, X, y = getWhiteWineQualityDataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
    print X_train.shape, X_test.shape, y_train.shape, y_test.shape