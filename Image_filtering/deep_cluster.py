from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from os import listdir
from os.path import isfile, join
import glob
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, SpectralClustering, KMeans

from functools import reduce
from operator import add

def cluster_images_from_dir(img_dir="./",algo=DBSCAN,kw_args={'metric':"cosine",'eps':0.3, 'min_samples':10,'algorithm':'brute'}):

    model = ResNet50(weights='imagenet', include_top=False)

    files = reduce(add,[glob.glob(img_dir + e) for e in ['*.jpg', '*.png']])
    X_train=np.zeros((len(files),224,224,3))
    X_raw=np.zeros((len(files),224,224,3),dtype=np.uint8)
    for i,f in enumerate(files):
        img = image.load_img(f, target_size=(224, 224))
        X_raw[i]=np.array(img,dtype=np.uint8)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        X_train[i]=x

    features = model.predict(X_train)
    features = features[:,0,0,:]
    labels = algo(**kw_args).fit(features).labels_
    return labels

if __name__ == "__main__":
    # DBSCAN
    dbscan_labels = cluster_images_from_dir(img_dir='bing/')
    # KMeans
    kmeans_labels = cluster_images_from_dir(img_dir='bing/',algo=KMeans,kw_args={'n_clusters':2+1})
    from collections import Counter
    import ipdb; ipdb.set_trace()