import numpy as np
import os,re

apostrophe = {"It's":"It is","\'re":" are","\'ll":" will","\'m":" am","\'d":" would","She's":"She is","He's":"He is"
			  ,"didn't":"did not","won't":"will not","can't":"cannot","shouldn't":"should not","wouldn't":"would not"
			  ,"couldn't":"could not"}


def replaceapostrophe(string):
  for k,v in apostrophe.items():
    string = re.sub(k,v,string)
  return string

def removepunctuation(sent):
    sent = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", sent)

## take filename and return the story and summary
def process_text(filename):
	f = open(filename,"r")
	story = ''
	summary = ''
	bit = 0
	counter = 0
	for l in f.readlines():
		#print(l)
		if l == '\n':
			continue
		if counter == 0:
			l = re.sub(r'^[^--]*-- ','',l)
		#l = removepunctuation(l)
		if l == '@highlight\n':
			bit = 1
			continue
		if bit == 0:
			story += l[:-1]+' '
		if bit == 1:
			summary += l[:-1]+'. '
		bit = 0
		counter = 1
	return story,summary

#use yield




