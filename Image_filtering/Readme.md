## [Search Image Filtering] - Using clustering + transfer learning, filter noisy images out from Bing Image search.

After a PCA transformation to 2 dimensions from 2048:

<img src="https://raw.githubusercontent.com/ArtificialBreeze/MachineLearning/master/media/ImageFiltering_1.png"></img>


We now apply *dbscan* with &#949;=0.01 and *min_samples=5*, obtaining a reduced gaussian-like shape:

<img src="https://raw.githubusercontent.com/ArtificialBreeze/MachineLearning/master/media/ImageFiltering_2.png"></img>

The images labeled as 0 are considered the true positives and should be added to the real training set. The remaining images (labeled with -1, which stands for *no cluster*) can be dismissed.

### Examples

| Query | Metrics | Conserved | Discarded |
|-------|---------|-----------|-----------|
|  dog  |Euclidean|  <img src="https://raw.githubusercontent.com/ArtificialBreeze/MachineLearning/master/media/ImageFiltering_dog_good_euclidean.jpg"></img>         |    <img src="https://raw.githubusercontent.com/ArtificialBreeze/MachineLearning/master/media/ImageFiltering_dog_bad_euclidean.jpg"></img>       |
|  dog  |Cosine|  <img src="https://raw.githubusercontent.com/ArtificialBreeze/MachineLearning/master/media/ImageFiltering_dog_good_cosine.jpg"></img>         |    <img src="https://raw.githubusercontent.com/ArtificialBreeze/MachineLearning/master/media/ImageFiltering_dog_bad_cosine.jpg"></img>       |
| metro |Cosine|  <img src="https://raw.githubusercontent.com/ArtificialBreeze/MachineLearning/master/media/ImageFiltering_metro_good_cosine.jpg"></img>         |    <img src="https://raw.githubusercontent.com/ArtificialBreeze/MachineLearning/master/media/ImageFiltering_metro_bad_cosine.jpg"></img>       |