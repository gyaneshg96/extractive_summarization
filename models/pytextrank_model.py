import sys,os
import pytextrank
import json
from nltk import sent_tokenize


''' ******** PYTEXTRANK ********'''
#uses textrank, with Jaccard Distance
#also considers POS tags and word importances


''' to comply with the pipeline the authors gave
we are creating functions as reqd.'''


## convert sentence into json
def make_json(story):
	json_cont = {}
	json_cont["id"] = 100
	json_cont["text"] = story
	with open('temp1.json', 'w') as json_file:
		json.dump(json_cont, json_file)

## POS tags of sentences
def parseSentence():
	with open('temp2.json', 'w') as f:
		for graf in pytextrank.parse_doc(pytextrank.json_iter('temp1.json')):
			f.write("%s\n" % pytextrank.pretty_print(graf._asdict()))

## list of keywords and phrases
def keyPhrases():
	graph, ranks = pytextrank.text_rank('temp2.json')
	pytextrank.render_ranks(graph, ranks)
	with open('temp3.json','w') as f:
		for rl in pytextrank.normalize_key_phrases('temp2.json', ranks):
			f.write("%s\n" % pytextrank.pretty_print(rl._asdict()))

## find top sentences using TextRank
## and arrange them in order of passage

def topSentences(strlen):
	kernel = pytextrank.rank_kernel('temp3.json')
	i = 0
	summary = []
	for s in pytextrank.top_sentences(kernel, 'temp2.json'):
		summary.append(s.text)
		i = i+1
		if i > strlen:
			break
	return summary

## remove all files created in process

def cleanUp():
	os.remove('temp1.json')
	os.remove('temp2.json')
	os.remove('temp3.json')
	os.remove('graph.dot')

## method which calls all of above
def pytextrank_method(story,ratio):
	story_list = sent_tokenize(story)
	num_sent = len(story_list)
	make_json(story)
	parseSentence()
	keyPhrases()
	summary = topSentences(num_sent*ratio)
	cleanUp()	
	return ' '.join(summary)