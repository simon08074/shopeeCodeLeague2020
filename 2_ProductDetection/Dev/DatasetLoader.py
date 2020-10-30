import os # generate filepath, check file exist 
from tqdm import tqdm # show for-loop progress

import cv2
import pandas as pd
import numpy as np

class DatasetLoader:
    
    def __init__(self, dataset_root='../dataset/', img_resize=(224,224)):
        self.dataset_root = dataset_root # relative path of dataset root
        self.img_resize = img_resize
    
    
    def getIndexListForLoadTrain(self, dataset_root=None, csvname='train.csv'):
        # check if dataset_root be assigned
        if dataset_root == None :
            dataset_root = self.dataset_root
        # load tables
        df = pd.read_csv(os.path.join(dataset_root, csvname))
        
        # create index_list
        index_list = np.arange(len(df))
        np.random.shuffle(index_list)
        return index_list
    
        
    def loadTrainByIndexList(self, dataset_root=None, csvname='train.csv', toRGB=True, normalization=True, verbose=True, index_list=[]):
        # check if dataset_root be assigned
        if dataset_root == None :
            dataset_root = self.dataset_root
        # load tables
        df = pd.read_csv(os.path.join(dataset_root, csvname))
        # sample by index_list
        df = df[df.index.isin(index_list)]
        
        # init result 
        images = []
        labels = []
        
        # load images and labels
        if verbose:
            for filename, category in tqdm(df.values):

                # load image
                img = self._loadImage(isTrain=True, category=category, filename=filename)
                # append
                images.append(img)
                labels.append(category)
        else:
            for filename, category in df.values:

                # load image
                img = self._loadImage(isTrain=True, category=category, filename=filename)
                # append
                images.append(img)
                labels.append(category)
            
        # trans to numpy type
        images = np.array(images)
        # color channel to RGB or not
        images = images[:,:, ::-1] if toRGB else images
        # normalize to [0,1] or not
        images = images/255
        
        labels = np.array(labels)
        
        return images, labels
    
    
    def loadTrain(self, dataset_root=None, csvname='train.csv', toRGB=True, normalization=True, n=5000, verbose=True, random_state=666):
        # check if dataset_root be assigned
        if dataset_root == None :
            dataset_root = self.dataset_root
        # load tables
        df = pd.read_csv(os.path.join(dataset_root, csvname))
        # init result 
        images = []
        labels = []
        
        # check how many images should be loaded
        if n >= len(df):
            # remain df as original 
            pass 
        elif n >= 1: 
            # actual number
            df = df.sample(n=n, random_state=random_state, axis=0)
        elif 1 > n > 0:
            # percentage
            n = int(len(df) * n)
            df = df.sample(n=n, random_state=random_state, axis=0)
        else :
            # just keep df as original
            pass 
        
        # load images and labels
        if verbose:
            for filename, category in tqdm(df.values):
                # load image
                img = self._loadImage(isTrain=True, category=category, filename=filename)
                # append
                images.append(img)
                labels.append(category)
        else:
            for filename, category in df.values:
                # load image
                img = self._loadImage(isTrain=True, category=category, filename=filename)
                # append
                images.append(img)
                labels.append(category)
            
        # trans to numpy type
        images = np.array(images)
        # color channel to RGB or not
        images = images[:,:, ::-1] if toRGB else images
        # normalize to [0,1] or not
        images = images/255
        
        labels = np.array(labels)
        
        return images, labels
        
        
    def loadTest(self, dataset_root=None, csvname='test.csv', toRGB=True, normalization=True, verbose=True):
        # check if dataset_root be assigned
        if dataset_root == None :
            dataset_root = self.dataset_root
        # load tables
        df = pd.read_csv(os.path.join(dataset_root, csvname))
        # init result
        filenames = []
        images = []
        
        # load images
        if verbose:
            for filename, category in tqdm(df.values):
                # load image
                img = self._loadImage(isTrain=False, category=category, filename=filename)
                # append
                filenames.append(filename)
                images.append(img)
        else:
            for filename, category in df.values:
                # load image
                img = self._loadImage(isTrain=False, category=category, filename=filename)
                # append
                filenames.append(filename)
                images.append(img)
        
        # trans to numpy type
        images = np.array(images)
        # color channel to RGB or not
        images = images[:,:, ::-1] if toRGB else images
        # normalize to [0,1] or not
        images = images/255
        
        return filenames, images
    
    #======================= private functions =======================#
    
    def _loadImage(self, isTrain= True, category= 0, filename="", normalization=True):
        
        #=======================
        # create filepath
        #=======================
        
        # Training set
        if isTrain:
            # check category
            if category < 0 or category > 41:
                raise ValueError("[ERROR] Unexpected category value from training set, it should be in range [0,41], but get:", category)

            isTrain = "train/train"
            category = str(category) if category >= 10 else ("0" + str(category))

        # Testing set
        else:
            # check cetegory
            if category != 43:
                raise ValueError("[ERROR] Unexpected category value from testing set, it should be 43, but get:", category)

            isTrain = "test/test"
            category = ""

        # concatenate as filepath
        filepath = os.path.join(self.dataset_root, isTrain, category, filename)
        
        
        #=======================
        # load image and proceed
        #=======================
        
        # load image
        if not os.path.exists(filepath):
            raise FileNotFoundError("[ERROR] File doesn't exist!, filepath:", filepath)
        else:    
            # load 
            img = cv2.imread(filepath)
            # resize
            img = cv2.resize(img, self.img_resize, interpolation = cv2.INTER_AREA)
            
            return img
            
#-------------------------- Main func ---------------------------#

def main():
    loader = DatasetLoader()
    
    # test load Train by index_list
    index_list = loader.getIndexListForLoadTrain()
    print(index_list[:10], len(index_list))
    images, labels = loader.loadTrainByIndexList(verbose=True, index_list=index_list[:10])
    
    # test loadTrain
    images, labels = loader.loadTrain(n=100, verbose=True)
    print(images.shape, labels.shape)
    
    # test loadTest
    filenames, images = loader.loadTest(verbose=True)
    print(images.shape)
    
if __name__ == '__main__':
    main()
