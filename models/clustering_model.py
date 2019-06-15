import skipthoughts
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import KMeans
import nltk
import numpy as np

''' ******** CLUSTERING SENTENCE VECTORS ********'''
#uses clustering to find the sentence closest to cluster sentences
#we use skipthought vectors for most accurate embeddings


# load models
# this will take time and is heavy
model = skipthoughts.load_model()
encoder = skipthoughts.Encoder(model)

## finds sentences closes to cluster centers

def findSentences(vectors,kmeans,n_clusters, story_list):
	avg = []
	for j in range(n_clusters):
		idx = np.where(kmeans.labels_ == j)[0]
		avg.append(np.mean(idx))
	closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, vectors)
	ordering = sorted(range(n_clusters), key=lambda k: avg[k])
	summary = ' '.join([story_list[closest[idx]] for idx in ordering])
	return summary

# calling function
def clustering_method(story,ratio):
	story_list = nltk.sent_tokenize(story)
	
	# find vectors
	vectors = encoder.encode(story_list,verbose=False)
	
	# reduce space
	svd = TruncatedSVD(500)
	vectors = svd.fit_transform(vectors)
	n_clusters = int(ratio*len(story_list))
	
	#clustering
	kmeans = KMeans(n_clusters=n_clusters)
	kmeans = kmeans.fit(vectors)
	return findSentences(vectors, kmeans,n_clusters, story_list)