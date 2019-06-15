from gensim.summarization import summarize


''' ******** GENSIM ********'''
#uses textrank, with O-25 function

def gensim_method(story,ratio):
	return summarize(story,ratio=ratio)

def gensim_method_many(stories,ratio):
	predictions = []
	for story in stories:
		predictions.append(gensim_method(story))