import os
import cv2

# dataset/image relative
def loadImage(datasetRoot= "./", isTrain= True, category= 0, filename="", needRGB=True):
    
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
    filepath = os.path.join(datasetRoot, isTrain, category, filename)
    # load image
    if os.path.exists(filepath):
        if needRGB: # RGB
            return cv2.imread(filepath, cv2.COLOR_BGR2RGB)
            # return cv2.imread(filepath)[:,:,:, ::-1] # BGR to RGB
        else : # BGR
            return cv2.imread(filepath)
    else:
        raise FileNotFoundError("[ERROR] File doesn't exist!, filepath:", filepath)
        
        
#-------------------------- Main func ---------------------------#

# main for testing
def main():
    loadImage(datasetRoot="../dataset", isTrain=True, category=3, filename="45e2d0c97f7bdf8cbf3594beb6fdcda0.jpg")
    pass

if __name__ == '__main__':
    main()