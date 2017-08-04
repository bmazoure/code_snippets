# Machine Learning developments


List of useful projects currently being worked on:

* [GloVe2word] - Flask app (deployable on Heroku) which allows to evaluate mathematical expressions with words using their respective GloVe encodings (Pennington *et al.* 2014)

| nearest neighbors of <br/> <em>frog</em> | Litoria             |  Leptodactylidae | Rana | Eleutherodactylus |
| --- | ------------------------------- | ------------------- | ---------------- | ------------------- |
| Pictures | <img src="http://nlp.stanford.edu/projects/glove/images/litoria.jpg"></img> | <img src="http://nlp.stanford.edu/projects/glove/images/leptodactylidae.jpg"></img> | <img src="http://nlp.stanford.edu/projects/glove/images/rana.jpg"></img> | <img src="http://nlp.stanford.edu/projects/glove/images/eleutherodactylus.jpg"></img> |

| Comparisons | man -> woman             |  city -> zip | comparative -> superlative |
| --- | ------------------------|-------------------------|-------------------------|
| GloVe Geometry | <img src="http://nlp.stanford.edu/projects/glove/images/man_woman_small.jpg"></img>  | <img src="http://nlp.stanford.edu/projects/glove/images/city_zip_small.jpg"></img> | <img src="http://nlp.stanford.edu/projects/glove/images/comparative_superlative_small.jpg"></img> |

1. Download the embedding vectors```wget http://nlp.stanford.edu/data/wordvecs/glove.6B.zip & unzip glove.6B.zip``` and put them in the ```./embeddings``` directory.
2. Try it out locally by running ```python app.py```, then go on *localhost:5000*.
It should give something like this:
<img src="https://raw.githubusercontent.com/ArtificialBreeze/MachineLearning/master/media/GloVe2word.PNG"></img>
3. [*Optional*] Deploy to Heroku: ```heroku create```, ```git add .```, ```git commit -m "init"``` and ```git push heroku master```.
* [Google Image Filtering] - Using clustering techniques and transfer learning, filter noisy images from Google Image search. This technique would allow researchers to automatically construct large datasets with minimal false positive counts.



