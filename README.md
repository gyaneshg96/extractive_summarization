# Extractive Summarization
## We use 2 particular methods
1. genisim summarizer (Text rank Okapi BM25 metric)<br/>
2. pytextrank summarizer (Text rank with Jaccard metric)<br/>
3. clustering summarizer (uses K Means with skipthoughts vectors)<br/>

## Download Dependencies
pip install -r requirements.txt

## For skipthoughts
Download the pickle files from the sources, as

wget http://www.cs.toronto.edu/~rkiros/models/dictionary.txt <br/>
wget http://www.cs.toronto.edu/~rkiros/models/utable.npy <br/>
wget http://www.cs.toronto.edu/~rkiros/models/btable.npy <br/>
wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz <br/>
wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz.pkl <br/>
wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz <br/>
wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz.pkl <br/>

And set the paths in 'skipthoughts.py' accordingly

## Get the story from the files

`from preprocessing import process_text`
`story, summary = process_text(filename)`

## Use libraries based on the summarizer

`from models.gensim_model import gensim_method`
`summary = gensim_method(story, ratio)`
