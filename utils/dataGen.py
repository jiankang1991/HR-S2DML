
import glob
from collections import defaultdict
import os
import numpy as np
import random


import torchvision.transforms as transforms

from PIL import Image
from skimage import io

def default_loader(path):
    return Image.open(path).convert('RGB')


def eurosat_loader(path):
    return io.imread(path)


def one_hot_encode(x, n_classes):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    return np.eye(n_classes)[x]



class DataGeneratorSM:
    """
    generate train and val dataset based on the following data structure:
    Data structure:
    └── SeaLake
        ├── SeaLake_1000.jpg
        ├── SeaLake_1001.jpg
        ├── SeaLake_1002.jpg
        ├── SeaLake_1003.jpg
        ├── SeaLake_1004.jpg
        ├── SeaLake_1005.jpg
        ├── SeaLake_1006.jpg
        ├── SeaLake_1007.jpg
        ├── SeaLake_1008.jpg
        ├── SeaLake_1009.jpg
        ├── SeaLake_100.jpg
        ├── SeaLake_1010.jpg
        ├── SeaLake_1011.jpg
    """

    def __init__(self, data, dataset, train_per, imgExt='jpg', imgTransform=None, phase='train'):

        self.dataset = dataset
        self.datadir = os.path.join(data, dataset)
        self.sceneList = [os.path.join(self.datadir, x) for x in sorted(os.listdir(self.datadir)) if os.path.isdir(os.path.join(self.datadir, x))]
        self.sceneFilesNum = defaultdict()
        
        self.train_per = train_per

        self.train_idx2fileDict = defaultdict()
        self.test_idx2fileDict = defaultdict()
        self.val_idx2fileDict = defaultdict()

        self.scene2Label = defaultdict()
        self.imgTransform = imgTransform
        self.imgExt = imgExt
        self.phase = phase
        self.CreateIdx2fileDict()


    def CreateIdx2fileDict(self):

        self.train_numImgs = 0
        self.test_numImgs = 0

        train_count = 0
        test_count = 0

        for label, scenePth in enumerate(self.sceneList):
            self.scene2Label[os.path.basename(scenePth)] = label

            subdirImgPth = sorted(glob.glob(os.path.join(scenePth, '*.'+self.imgExt)))
            random.seed(42)
            random.shuffle(subdirImgPth)

            train_subdirImgPth = subdirImgPth[:int(self.train_per*len(subdirImgPth))]
            test_subdirImgPth = subdirImgPth[int(self.train_per*len(subdirImgPth)):]

            # self.sceneFilesNum[os.path.basename(scenePth)] = len(subdirImgPth)
            self.train_numImgs += len(train_subdirImgPth)
            self.test_numImgs += len(test_subdirImgPth)

            for imgPth in train_subdirImgPth:
                self.train_idx2fileDict[train_count] = (imgPth, label)
                train_count += 1
            
            for imgPth in test_subdirImgPth:
                self.test_idx2fileDict[test_count] = (imgPth, label)
                test_count += 1
        
        print("total number of classes: {}".format(len(self.sceneList)))
        print("total number of train images: {}".format(self.train_numImgs))
        print("total number of test images: {}".format(self.test_numImgs))

        # # self.totalDataIndex = list(range(self.numImgs))
        self.trainDataIndex = list(range(self.train_numImgs))
        self.testDataIndex = list(range(self.test_numImgs))


    def __getitem__(self, index):

        if self.phase == 'train':
            idx = self.trainDataIndex[index]
        else:
            idx = self.testDataIndex[index]
        
        return self.__data_generation(idx)

            
    def __data_generation(self, idx):
        
        # imgPth, imgLb = self.idx2fileDict[idx]

        if self.phase == 'train':
            imgPth, imgLb = self.train_idx2fileDict[idx]
        else:
            imgPth, imgLb = self.test_idx2fileDict[idx]

        if self.dataset == 'eurosat':
            img = eurosat_loader(imgPth).astype(np.float32)
        else:
            img = default_loader(imgPth)

        if self.imgTransform is not None:
            img = self.imgTransform(img)
        
        # print(img.shape)
        oneHotVec = one_hot_encode(imgLb, len(self.sceneList))

        return {'img': img, 'label': imgLb, 'idx':idx, 'onehot':oneHotVec.astype(np.float32)}
        # one hot encoding
        # oneHotVec = one_hot_encode(imgLb, len(self.sceneList))

        # return {'img': img, 'label': imgLb}

    def __len__(self):
        
        if self.phase == 'train':
            return len(self.trainDataIndex)
        else:
            return len(self.testDataIndex)




class DataGenerator:
    """
    generate train and val dataset based on the following data structure:
    Data structure:
    └── SeaLake
        ├── SeaLake_1000.jpg
        ├── SeaLake_1001.jpg
        ├── SeaLake_1002.jpg
        ├── SeaLake_1003.jpg
        ├── SeaLake_1004.jpg
        ├── SeaLake_1005.jpg
        ├── SeaLake_1006.jpg
        ├── SeaLake_1007.jpg
        ├── SeaLake_1008.jpg
        ├── SeaLake_1009.jpg
        ├── SeaLake_100.jpg
        ├── SeaLake_1010.jpg
        ├── SeaLake_1011.jpg
    """

    def __init__(self, data, dataset, imgExt='jpg', imgTransform=None):

        self.dataset = dataset
        self.datadir = os.path.join(data, dataset)
        self.sceneList = [os.path.join(self.datadir, x) for x in sorted(os.listdir(self.datadir)) if os.path.isdir(os.path.join(self.datadir, x))]
        self.sceneFilesNum = defaultdict()
        
        # self.train_per = train_per

        self.train_idx2fileDict = defaultdict()
        # self.test_idx2fileDict = defaultdict()
        # self.val_idx2fileDict = defaultdict()

        self.scene2Label = defaultdict()
        self.imgTransform = imgTransform
        self.imgExt = imgExt
        self.CreateIdx2fileDict()


    def CreateIdx2fileDict(self):

        self.train_numImgs = 0
        # self.test_numImgs = 0

        train_count = 0
        # test_count = 0

        for label, scenePth in enumerate(self.sceneList):
            self.scene2Label[os.path.basename(scenePth)] = label

            subdirImgPth = sorted(glob.glob(os.path.join(scenePth, '*.'+self.imgExt)))
            random.seed(42)
            random.shuffle(subdirImgPth)

            train_subdirImgPth = subdirImgPth
            # test_subdirImgPth = subdirImgPth[int(self.train_per*len(subdirImgPth)):]

            # self.sceneFilesNum[os.path.basename(scenePth)] = len(subdirImgPth)
            self.train_numImgs += len(train_subdirImgPth)
            # self.test_numImgs += len(test_subdirImgPth)

            for imgPth in train_subdirImgPth:
                self.train_idx2fileDict[train_count] = (imgPth, label)
                train_count += 1
            
        
        print("total number of classes: {}".format(len(self.sceneList)))
        print("total number of train images: {}".format(self.train_numImgs))
        # print("total number of test images: {}".format(self.test_numImgs))

        # # self.totalDataIndex = list(range(self.numImgs))
        self.trainDataIndex = list(range(self.train_numImgs))
        # self.testDataIndex = list(range(self.test_numImgs))


    def __getitem__(self, index):

        idx = self.trainDataIndex[index]
                
        return self.__data_generation(idx)

            
    def __data_generation(self, idx):
        
        # imgPth, imgLb = self.idx2fileDict[idx]

        imgPth, imgLb = self.train_idx2fileDict[idx]

        if self.dataset == 'eurosat':
            img = eurosat_loader(imgPth).astype(np.float32)
        else:
            img = default_loader(imgPth)

        if self.imgTransform is not None:
            img = self.imgTransform(img)
        
        # print(img.shape)
        oneHotVec = one_hot_encode(imgLb, len(self.sceneList))

        return {'img': img, 'label': imgLb, 'idx':idx, 'onehot':oneHotVec.astype(np.float32)}
        # one hot encoding
        # oneHotVec = one_hot_encode(imgLb, len(self.sceneList))

        # return {'img': img, 'label': imgLb}

    def __len__(self):
        
        return len(self.trainDataIndex)



class DataGeneratorSM_trip:

    def __init__(self, data, dataset, train_per, imgExt='jpg', imgTransform=None, phase='train'):

        self.datadir = os.path.join(data, dataset)
        self.sceneList = [os.path.join(self.datadir, x) for x in sorted(os.listdir(self.datadir)) if os.path.isdir(os.path.join(self.datadir, x))]
        self.sceneFilesNum = defaultdict()

        self.train_per = train_per

        self.train_idx2fileDict = defaultdict()
        self.test_idx2fileDict = defaultdict()
        # self.val_idx2fileDict = defaultdict()

        self.train_label2idx = defaultdict()
        self.scene2Label = defaultdict()
        self.imgTransform = imgTransform
        self.imgExt = imgExt
        self.phase = phase
        self.labels_list = None

        self.CreateIdx2fileDict()

    def CreateIdx2fileDict(self):
        # import random
        # random.seed(42)

        self.train_numImgs = 0
        self.test_numImgs = 0
        # self.val_numImgs = 0

        train_count = 0
        test_count = 0
        # val_count = 0

        for label, scenePth in enumerate(self.sceneList):
            self.scene2Label[os.path.basename(scenePth)] = label

            subdirImgPth = sorted(glob.glob(os.path.join(scenePth, '*.'+self.imgExt)))
            random.seed(42)
            random.shuffle(subdirImgPth)
            
            train_subdirImgPth = subdirImgPth[:int(self.train_per*len(subdirImgPth))]
            test_subdirImgPth = subdirImgPth[int(self.train_per*len(subdirImgPth)):]

            # self.sceneFilesNum[os.path.basename(scenePth)] = len(subdirImgPth)
            self.train_numImgs += len(train_subdirImgPth)
            self.test_numImgs += len(test_subdirImgPth)
            # self.val_numImgs += len(val_subdirImgPth)

            for imgPth in train_subdirImgPth:
                self.train_idx2fileDict[train_count] = (imgPth, label)

                if label in self.train_label2idx:
                    self.train_label2idx[label].append(train_count)
                else:
                    self.train_label2idx[label] = [train_count]
                train_count += 1
                
            for imgPth in test_subdirImgPth:
                self.test_idx2fileDict[test_count] = (imgPth, label)
                test_count += 1
            
        
        self.labels_list = list(range(len(self.sceneList)))

        print("total number of classes: {}".format(len(self.sceneList)))
        print("total number of train images: {}".format(self.train_numImgs))
        print("total number of test images: {}".format(self.test_numImgs))

        # self.totalDataIndex = list(range(self.numImgs))
        self.trainDataIndex = list(range(self.train_numImgs))
        self.testDataIndex = list(range(self.test_numImgs))

    def __getitem__(self, index):

        if self.phase == 'train':
            idx = self.trainDataIndex[index]
            _, imgLb = self.train_idx2fileDict[idx]
            positive_index = idx
            while positive_index == idx:
                positive_index = np.random.choice(self.train_label2idx[imgLb])
            
            negative_label = np.random.choice(list(set(self.labels_list) - set([imgLb])))
            negative_index = np.random.choice(self.train_label2idx[negative_label])
            return self.__data_generation_triplet(idx, positive_index, negative_index)
        else:
            idx = self.testDataIndex[index]
            return self.__data_generation(idx)

    def __data_generation(self, idx):
        
        # imgPth, imgLb = self.idx2fileDict[idx]
        if self.phase == 'test':
            imgPth, imgLb = self.test_idx2fileDict[idx]

        img = default_loader(imgPth)

        if self.imgTransform is not None:
            img = self.imgTransform(img)
        
        # print(img.shape)

        # return {'img': img, 'label': imgLb, 'idx':idx}
        # one hot encoding
        # oneHotVec = one_hot_encode(imgLb, len(self.sceneList))

        return {'img': img, 'label': imgLb}

    def __data_generation_triplet(self, idx, pos_idx, neg_idx):

        anc_imgPth, anc_label = self.train_idx2fileDict[idx]
        pos_imgPth, _ = self.train_idx2fileDict[pos_idx]
        neg_imgPth, _ = self.train_idx2fileDict[neg_idx]

        anc_img = default_loader(anc_imgPth)
        pos_img = default_loader(pos_imgPth)
        neg_img = default_loader(neg_imgPth)

        if self.imgTransform is not None:
            anc_img = self.imgTransform(anc_img)
            pos_img = self.imgTransform(pos_img)
            neg_img = self.imgTransform(neg_img)

        return {'anc':anc_img, 'pos':pos_img, 'neg':neg_img, 'anc_label':anc_label}

    def __len__(self):
        
        if self.phase == 'train':
            return len(self.trainDataIndex)
        else:
            return len(self.testDataIndex)

