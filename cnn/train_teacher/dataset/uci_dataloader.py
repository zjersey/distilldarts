import json
import platform
import numpy as np
import pandas as pd
from numpy import concatenate as npconcat
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB

import torch
from torch.utils.data import Dataset, DataLoader

BASE_DIR = "data/DataSet_Example/"
if platform.system() == "Windows":
    BASE_DIR = "C:/" + BASE_DIR
PIMA_PATH = {"diabetes": BASE_DIR + "PIMA/diabetes.csv"}
WINEQ_PATH = {"name": BASE_DIR + "WINEQ/winequality.names",
              "red": BASE_DIR + "WINEQ/winequality-red.csv",
              "white": BASE_DIR + "WINEQ/winequality-white.csv"}
ISOLET_PATH = {"info": BASE_DIR + "ISOLET/isolet.info",
               "name": BASE_DIR + "ISOLET/isolet.names",
               "1234": BASE_DIR + "ISOLET/isolet1+2+3+4.data",
               "5": BASE_DIR + "ISOLET/isolet5.data"}
HIGGSBOSON_PATH = {"training": BASE_DIR + "HIGGSBOSON/training.csv",
                   "test": BASE_DIR + 'HIGGSBOSON/test.csv'}
HOMULTIN_PATH = {"features": BASE_DIR + "HOMULTIN/features.npy",
                 "labels": BASE_DIR + "HOMULTIN/labels.npy"}

class DataLoader(Dataset):
    """Provide many kinds of data loader for each kind of dataset"""
    def __init__(self, transform=None, dataset_name=None, mode='train', seed=1234, val_rate=0.1, test_rate=0.1):
        self.transform = transform
        self.val_rate = val_rate
        self.test_rate = test_rate
        self.seed = seed
        self.rng = np.random.RandomState(seed=seed)
        self.cache = None
        self.mode = mode
        self.dataset_name = dataset_name
        self.loader = {"Pima": self.pima,
                       "Wineq": self.wineq,
                       "Isolet": self.isolet,
                       "Higgsboson": self.higgsboson,
                       "Homultin": self.homultin}
        if dataset_name in self.loader.keys():
            self.loader[dataset_name]()
        else:
            raise(ValueError("no dataset named %s"%dataset_name))

    def set_seed(self, seed):
        self.seed = seed
        self.rng = np.random.RandomState(seed=seed)
        if self.dataset_name in self.loader.keys():
            self.loader[self.dataset_name]()
        else:
            raise(ValueError("no dataset named %s"%self.dataset_name))


    def split_dataset(self, features, labels):
        pool_index = range(features.shape[0])
        num_val = int(features.shape[0] * self.val_rate)
        num_test = int(features.shape[0] * self.test_rate)

        val_index = self.rng.choice(range(features.shape[0]), num_val, replace=False)
        pool_index = list(set(pool_index) - set(val_index))
        test_index = self.rng.choice(pool_index, num_test, replace=False)
        train_index = list(set(pool_index) - set(test_index))
        assert( len(set(train_index).intersection(set(val_index)))==0 )
        assert( len(set(train_index).intersection(set(test_index)))==0 )
        assert( len(set(val_index).intersection(set(test_index)))==0 )
        self.train_index = train_index
        self.val_index = val_index
        self.test_index = test_index
        if self.mode == 'train':
            features = features[train_index,:]
            labels = labels[train_index]
        elif self.mode == 'val':
            features = features[val_index,:]
            labels = labels[val_index]
        elif self.mode == 'test':
            features = features[test_index,:]
            labels = labels[test_index]
        else:
            raise(ValueError("no mode named %s"%self.mode))
        return features, labels

    def pima(self, path=None):
        """Load pima data. All the data will be stored in a dict which
        will be set as `self.cache`, instead of being returned outside.
        The dict will be composed of two parts including features and
        labels.

        :param path: if you want to use your path instead of the default
          path, you can set this parameter.

        """
        self.num_classes = 2
        if path is None:
            path = PIMA_PATH
        scaler = MinMaxScaler()

        pima_ = np.array(pd.read_csv(path["diabetes"]))
        features, labels = pima_[:, :-1], pima_[:, -1]
        features = scaler.fit_transform(features)

        features, labels = self.split_dataset(features, labels)
        self.input_dim = features.shape[1]

        self.cache = {"features": features, "labels": labels}

    def wineq(self, path=None):
        """Load wine quality data. All the data will be stored in a dict
        which will be set as `self.cache`, instead of being returned
        outside. The dict will be composed of two parts including
        features and labels.

        :param path: if you want to use your path instead of the default
          path, you can set this parameter.

        """
        if path is None:
            path = WINEQ_PATH
        scaler = MinMaxScaler()

        wineq_red = pd.read_csv(path["red"], sep=";")
        wineq_white = pd.read_csv(path["white"], sep=";")
        wineq_ = npconcat((np.array(wineq_red), np.array(wineq_white)))
        features, labels = wineq_[:, :-1], wineq_[:, -1]
        features = scaler.fit_transform(features)
        label_min = labels.min()
        label_max = labels.max()
        labels = labels - label_min
        self.num_classes = int(label_max - label_min + 1)

        features, labels = self.split_dataset(features, labels)
        self.input_dim = features.shape[1]

        self.cache = {"features": features, "labels": labels}

    def isolet(self, path=None):
        """Load isolet data. All the data will be stored in a dict which
        will be set as `self.cache`, instead of being returned outside.
        The dict will be composed of two parts including features and
        labels.

        :param path: if you want to use your path instead of the default
          path, you can set this parameter.

        """
        self.num_classes = 26
        if path is None:
            path = ISOLET_PATH
        scaler = MinMaxScaler()

        # random split the dataset
        isolet_names = ["f" + str(i) for i in range(1, 618)] + ["label"]
        isolet1234 = pd.read_csv(path["1234"], sep=",", header=None,
                                 names=isolet_names)
        isolet5 = pd.read_csv(path["5"], sep=",", header=None,
                              names=isolet_names)
        isolet_ = npconcat((np.array(isolet1234), np.array(isolet5)))
        features, labels = isolet_[:, :-1], isolet_[:, -1]
        features = scaler.fit_transform(features)
        labels = labels - 1

        features, labels = self.split_dataset(features, labels)
        self.input_dim = features.shape[1]

        self.cache = {"features": features, "labels": labels}

    def higgsboson(self, path=None):
        """Load higgsboson data. All the data will be stored in a dict
        which will be set as `self.cache`, instead of being returned
        outside. The dict will be composed of two parts including
        features and labels.

        :param path: if you want to use your path instead of the default
          path, you can set this parameter.

        """
        self.num_classes = 2
        if path is None:
            path = HIGGSBOSON_PATH
        scaler = MinMaxScaler()

        if self.mode in ['train', 'val', 'test']:
            higgsboson_ = pd.read_csv(path["training"]).drop(columns="EventId")
            labels = higgsboson_["Label"]
            higgsboson_["Label"] = labels.apply(lambda x: 0 if x == "s" else 1)
            higgsboson_ = np.array(higgsboson_)
            higgsboson_[higgsboson_ == -999] = 0
            features, labels = higgsboson_[:, :-1], higgsboson_[:, -1]
            features = scaler.fit_transform(features)
    
            features, labels = self.split_dataset(features, labels)
        else:
            higgsboson_ = pd.read_csv(path["test"]).drop(columns="EventId")
            labels = higgsboson_["Label"]
            higgsboson_["Label"] = labels.apply(lambda x: 0 if x == "s" else 1)
            higgsboson_ = np.array(higgsboson_)
            higgsboson_[higgsboson_ == -999] = 0
            features, labels = higgsboson_[:, :-1], higgsboson_[:, -1]
            features = scaler.fit_transform(features)

        self.input_dim = features.shape[1]

        self.cache = {"features": features, "labels": labels}

    def homultin(self, path=None):
        """Load homultin data. All the data will be stored in a dict
        which will be set as `self.cache`, instead of being returned
        outside. The dict will be composed of two parts including
        features and labels.

        :param path: if you want to use your path instead of the default
          path, you can set this parameter.
        """
        if path is None:
            path = HOMULTIN_PATH

        features = np.load(path["features"])
        labels = np.load(path["labels"])
        labels_mean = np.mean(labels)
        labels[labels > labels_mean] = 1
        labels[labels < labels_mean] = 0

        features, labels = self.split_dataset(features, labels)
        self.input_dim = features.shape[1]

        self.cache = {"features": features, "labels": labels}

    def __len__(self):
        return self.cache['labels'].size

    def __getitem__(self, item):
        feature = self.cache["features"][item,:]
        label = self.cache["labels"][item]

        feature = torch.Tensor(feature)
        label = torch.Tensor([label]).long().squeeze()
        if self.transform is not None:
            try:
                feature = self.transform(feature)
            except:
                print("Cannot transform feature: {}".format(img_path))
        return feature, label
