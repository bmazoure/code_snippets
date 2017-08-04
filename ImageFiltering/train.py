from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from os import listdir
from os.path import isfile, join
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from getImages import getImages
# In[]
query='dog'
max_its=200
print("Downloading images")
#getImages(query=query,max_its=max_its)
print("done")

# In[]
model = ResNet50(weights='imagenet', include_top=False)

img_dir="bing"
files = [f for f in listdir(img_dir) if isfile(join(img_dir, f))]
X_train=np.zeros((len(files),224,224,3))
X_raw=np.zeros((len(files),224,224,3),dtype=np.uint8)
for i,f in enumerate(files):
    img = image.load_img(img_dir+"/"+f, target_size=(224, 224))
    X_raw[i]=np.array(img,dtype=np.uint8)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    X_train[i]=x

features = model.predict(X_train)

features = features[:,0,0,:]

#linfit = PCA(n_components=2)
#linfeatures=linfit.fit_transform(features)

# In[]
#db = DBSCAN(eps=0.01, min_samples=10).fit(linfeatures).labels_
#plt.plot(linfeatures[db!=-1,0], linfeatures[db!=-1,1],'ro')
#plt.plot(linfeatures[db==-1,0], linfeatures[db==-1,1],'ko')
#plt.plot(linfeatures[0], linfeatures[1],'ko')
#plt.show()

# In[]
metric='cosine'
labels = DBSCAN(eps=0.47, min_samples=10,algorithm='brute',metric=metric).fit(features).labels_
X_new=X_raw[labels==0]
X_bad=X_raw[labels!=0]
def gallery(array, ncols=13):
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape((nrows, ncols, height, width, intensity))
              .swapaxes(1,2)
              .reshape((height*nrows, width*ncols, intensity)))
    return result
result = gallery(X_new[:48],ncols=12)
fig=plt.imshow(result)
plt.imsave("../media/ImageFiltering_"+query+"_good"+metric+".jpg",result)
plt.show()
result = gallery(X_bad[:18],ncols=9)
plt.imshow(result)
plt.imsave("../media/ImageFiltering_"+query+"_bad"+metric+".jpg",result)
plt.show()
