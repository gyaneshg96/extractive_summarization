import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from preprocessing import process_text
import skipthoughts

''' load the skipthoughts sentence embeddings'''

model = skipthoughts.load_model()
encoder = skipthoughts.Encoder(model)

stories = []
summaries = []
sent2keys = []
i = 0
for line in open("names.txt").readlines():
	filename = "dataset/stories_text_summarization_dataset_train/"+line[:-1]
	print(filename,i)
	X,Y = process_text(filename)
	inversemap = {}
	inversemap[sent] = i 
	story.append(X)
	summary.append(Y)
	sent2keys.append(inversemap)
