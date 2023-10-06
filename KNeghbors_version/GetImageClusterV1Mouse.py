# for loading/processing the images  
from tensorflow.keras.utils import load_img 
from tensorflow.keras.utils import img_to_array 
from keras.applications.vgg16 import preprocess_input 
import tensorflow as tf
# models 
from keras.applications.vgg16 import VGG16 
from keras.models import Model

# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# for everything else
import os
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import pickle
import json

PATH = r'C:\Users\lukez\PycharmProjects\CMSC499Project\Models\\'


def extract_features(file):
    # load the model first and pass as an argument
    model = VGG16()
    model = Model(inputs = model.inputs, outputs = model.layers[-2].output)
    # load the image as a 224x224 array
    img = tf.image.resize(file, [224,224])
    #img = load_img(file, target_size=(224,224))
    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img) 
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1,224,224,3) 
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True)
    return features

def getProdcutName(imag, pca_model, kmeans_model, kmeansDictionary):
    data2 = {}
    feat = extract_features(imag)
    data2['test'] = feat
    # if something fails, save the extracted features as a pickle file (optional)

            
    
    # get a list of the filenames
    filenames = np.array(list(data2.keys()))

    # get a list of just the features
    feat2 = np.array(list(data2.values()))

    # reshape so that there are 210 samples of 4096 vectors
    feat2 = feat2.reshape(-1,4096)

    # reduce the amount of dimensions in the feature vector
    x = pca_model.transform(feat2)
    #print(x.shape)
    cluster = kmeans_model.predict(x)

    Dict = kmeansDictionary

    return (Dict[cluster[0]])

def getProdcutName2(imag, pca_model, kNeghborsModel, kNeghborsList):
    data2 = {}
    feat = extract_features(imag)
    data2['test'] = feat
    # if something fails, save the extracted features as a pickle file (optional)

            
    
    # get a list of the filenames
    filenames = np.array(list(data2.keys()))

    # get a list of just the features
    feat2 = np.array(list(data2.values()))

    # reshape so that there are 210 samples of 4096 vectors
    feat2 = feat2.reshape(-1,4096)

    # reduce the amount of dimensions in the feature vector
    x = pca_model.transform(feat2)
    #print(x.shape)
    distance, position = kNeghborsModel.query(x, k=20, return_distance=True)

    return (kNeghborsList[position[0][0]])
    
def loadProductData(file):
    finalDict = {}
    with open(file, "r") as ProductDataFull:
        lines = ProductDataFull.readlines()
        for line in lines:
            if(len(line) > 1):
                #print(line)
                line = line.replace('\'', '\"')
                lineDict = json.loads(str(line))
                finalDict[lineDict['Name']] = lineDict
    
    return finalDict
                
def getImageInfo(imag, prodcut):
    #path1 = 
    #os.chdir(path1)
    with open (PATH + 'pcaModel1_' + prodcut + '.pkl', 'rb') as fp:
            pca_model = pickle.load(fp)
    with open (PATH + 'kNeghborsModel1_' + prodcut + '.pkl', 'rb') as fp:
            kmeans_model = pickle.load(fp)
    with open (PATH + 'kNeghborsList1_' + prodcut + '.pkl', 'rb') as fp:
            kmeansDictionary = pickle.load(fp)

    ProductsDict = (loadProductData(PATH + 'ProductData_' + prodcut + '.txt'))
    #print(kmeansDictionary)
    #print()
    #print('stuff')

    prodcutName = (getProdcutName2(imag, pca_model, kmeans_model, kmeansDictionary))
    print(prodcutName['productName'])
    
    """
    ProductsDict = (loadProductData('ProductData_' + prodcut + '.txt'))
    #print(kmeansDictionary)
    #print()
    #print('stuff')

    prodcutName = (getProdcutName(imag, pca_model, kmeans_model, kmeansDictionary))
    """

    if(prodcutName['productName'] == 'No Match'):
        return 'No Match'
    else:
        return (ProductsDict[prodcutName['productName']])
    

def main():
    #path = r"C:\Users\lukez\PycharmProjects\CMSC499\Image Scrape\Images\chair\Amazon Basics Ergonomic Adjustable HighBack Mesh Chair with FlipUp Arms and Headrest Contoured Mesh Seat  Black"
    imageName = 'image3.png'
    #os.chdir(path)
    img = load_img(imageName)

    print(getImageInfo(img, 'chair'))
    
    #path1 = r"C:\Users\lukez\PycharmProjects\CMSC499\Image Scrape"
    #os.chdir(path1)
    #with open ('pcaModel_chair.pkl', 'rb') as fp:
    #        pca_model = pickle.load(fp)
    #with open ('kmeansModel_chair.pkl', 'rb') as fp:
    #        kmeans_model = pickle.load(fp)
    #with open ('kmeansDictionary_chair.pkl', 'rb') as fp:
    #        kmeansDictionary = pickle.load(fp)

    #ProductsDict = (loadProductData('ProductData.txt'))

    #path = r"C:\Users\lukez\PycharmProjects\CMSC499\Image Scrape\Images\chair\Amazon Basics Ergonomic Adjustable HighBack Mesh Chair with FlipUp Arms and Headrest Contoured Mesh Seat  Black"
    #imageName = 'image3.png'
    #prodcutName = (getProdcutName(path, imageName, pca_model, kmeans_model, kmeansDictionary))
    

    #print(ProductsDict[prodcutName])

    

if __name__ == "__main__":
    main()