import sys
from preprocessing import process_text
import nltk
from nltk import sent_tokenize
from models import gensim_model,pytextrank_model

try:
	nltk.data.find('tokenizers/punkt')
except LookupError:
	nltk.download('punkt')

if len(sys.argv) < 4:
	print("Usage : python run.py <filename> <method> <ratio>")
	sys.exit(1)

filename = sys.argv[1]
default_ratio = 0.15

story, summary = process_text(filename)


prediction = []
if len(sys.argv) == 4:
	ratio = default_ratio
if len(sys.argv) == 5:
	ratio = float(sys.argv[3])

method = sys.argv[2]
if method == "gensim":
	prediction = gensim_model.gensim_method(story,ratio)
elif method == "pytextrank":
	prediction = pytextrank_model.pytextrank_method(story,ratio)
elif method == "clustering":
	from models import clustering_model
	prediction = clustering_model.clustering_method(story,ratio)
else:
	print("Invalid argument")
	sys.exit(2)
print(prediction)

#if summary != []:
