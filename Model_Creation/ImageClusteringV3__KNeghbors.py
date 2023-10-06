# for loading/processing the images  
from tensorflow.keras.utils import load_img 
from tensorflow.keras.utils import img_to_array 
from keras.applications.vgg16 import preprocess_input 
import tensorflow as tf

# models 
from keras.applications.vgg16 import VGG16 
from keras.models import Model
from sklearn.neighbors import KDTree

# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from PIL import ImageFile

# for everything else
import os
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import pickle
            
"""    
PATH is were the image folders are saved
PAth1 is were the data for those images are saved.
"""

PATH = r'C:\Users\lukez\PycharmProjects\CMSC499\Models\Images'
PATH1 = r'C:\Users\lukez\PycharmProjects\CMSC499\Models'
            
#model = VGG16()
#model = Model(inputs = model.inputs, outputs = model.layers[-2].output)

def extract_features(file, model):
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    # load the image as a 224x224 array
    img = load_img(file, target_size=(224,224))

    #img = tf.image.resize(img, [224,224])
    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img) 
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1,224,224,3) 
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True)
    return features


def loadImages(product, maxProducts):
    path = PATH
    path = path + '\\' + product
    productImages = []
    number_products = 0

    os.chdir(path)


    folders = os.listdir(path)
    if((maxProducts > len(folders)) or maxProducts == -1):
        maxProducts = len(folders)
    for folder in range(maxProducts):
        number_products = number_products + 1
        """
        print()
        print(folders[folder])
        print()
        """
        with os.scandir(path + '\\' + folders[folder]) as files:
            for image in files:
                #print(image)
                productImages.append({'image': image.name, 'productName':folders[folder]})
        #print()
        
    return productImages, number_products





# function that lets you view a cluster (based on identifier)        
def view_cluster(groups, cluster, product):
    plt.figure(figsize = (25,25));
    path = PATH
    path = path + '\\' + product
    # gets the list of filenames for a cluster
    files = groups[cluster]
    
    if len(files) > 90:
        print(f"Clipping cluster size from {len(files)} to 30")
        files = files[:89]
    # plot each image in the cluster
    for index, file in enumerate(files):
        iamgeFile = file.split('__')
        os.chdir(path + '\\' + iamgeFile[0])
        plt.subplot(10,10,index+1);
        img = load_img(iamgeFile[1])
        img = np.array(img)
        plt.imshow(img)
        plt.axis('off')

# function that lets you view a cluster (based on identifier)        
def get_cluster_value(groups, cluster, percentage = .7):
    # gets the list of filenames for a cluster
    files = groups[cluster]

    imagetypes = {}
    typesList = []
    for index, file in enumerate(files):
        iamgeFile = file.split('__')
        if (not(iamgeFile[0] in imagetypes)):
            imagetypes[iamgeFile[0]] = 1
            typesList.append(iamgeFile[0])
        else:
            imagetypes[iamgeFile[0]] = imagetypes[iamgeFile[0]] + 1
    
    for types in typesList:
        print(imagetypes[types]/len(files))
        if((imagetypes[types]/len(files)) > percentage):
            return types
    return 'No Match'

def create_Model(productImages, number_products, model, product):
    data = {}
    path = PATH
    path = path + '\\' + product
    # lop through each image in the dataset
    for image in productImages:
        # try to extract the features and update the dictionary
        os.chdir(path + '\\' + image['productName'])
        feat = extract_features(image['image'],model)
        data[image['productName'] + '__' + image['image']] = feat
        # if something fails, save the extracted features as a pickle file (optional)
        
            
    
    # get a list of the filenames
    filenames = np.array(list(data.keys()))

    # get a list of just the features
    feat1 = np.array(list(data.values()))

    # reshape so that there are 210 samples of 4096 vectors
    feat1 = feat1.reshape(-1,4096)

    # reduce the amount of dimensions in the feature vector
    pca = PCA(n_components=200)
    pca.fit(feat1)
    x = pca.transform(feat1)

    kdt = KDTree(x, leaf_size=30, metric='euclidean')
    """
    view_cluster(groups, 1, product)
    print()
    print()
    print()
    view_cluster(groups, 2, product)
    print()
    print()
    print()
    view_cluster(groups, 3, product)
    """
    path1 = PATH1
    print(productImages)
    os.chdir(path1)

    with open('kNeghborsList1_' + product + '.pkl','wb') as f:
        pickle.dump(productImages,f)

    with open('kNeghborsModel1_' + product + '.pkl','wb') as f:
        pickle.dump(kdt,f)

    with open('pcaModel1_' + product + '.pkl','wb') as f:
        pickle.dump(pca,f)

    

def CREATE_Model_Upper(product):
    # load the model first and pass as an argument
    #site for a bunch of models we could try
    #https://keras.io/api/applications/
    model = VGG16()
    model = Model(inputs = model.inputs, outputs = model.layers[-2].output)

    images, number_products = loadImages(product, -1)
    create_Model(images, number_products, model, product)



def main():
    product = 'mouse'
    CREATE_Model_Upper(product)

if __name__ == "__main__":
    main()