from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from os import listdir
from os.path import isfile, join
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


model = ResNet50(weights='imagenet', include_top=False)

img_dir="bing"
files = [f for f in listdir(img_dir) if isfile(join(img_dir, f))]
X_train=np.zeros((len(files),224,224,3))
for i,f in enumerate(files):
    img = image.load_img(img_dir+"/"+f, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    X_train[i]=x

features = model.predict(X_train)

features = features[:,0,0,:]

linfit = PCA(n_components=2)
linfeatures=linfit.fit_transform(features)

# In[]
db = DBSCAN(eps=0.01, min_samples=10).fit(linfeatures).labels_
plt.plot(linfeatures[db!=-1,0], linfeatures[db!=-1,1],'ro')
plt.plot(linfeatures[db==-1,0], linfeatures[db==-1,1],'ko')
plt.plot(linfeatures[0], linfeatures[1],'ko')
plt.show()