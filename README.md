# extractive_summarization
## Some ways to perform extractive summarization
## We use 2 particular methods
i) genisim summarizer (Text rank O-25 metric)
ii) pytextrank summarizer (Text rank with jaccard metric)
iii) clustering summarizer (uses K Means with skipthoughts vectors)

## For skipthoughts
Download the pickle files from the sources, as 

wget http://www.cs.toronto.edu/~rkiros/models/dictionary.txt
wget http://www.cs.toronto.edu/~rkiros/models/utable.npy
wget http://www.cs.toronto.edu/~rkiros/models/btable.npy
wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz
wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz.pkl
wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz
wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz.pkl

And set the path in skipthoughts accordingly

## Get the story from the files

from preprocessing import process_text
story, summary = process_text(filename)

## Use libraries based on the summarizer

from models.gensim_model import gensim_method
summary = gensim_method(story, ratio)
