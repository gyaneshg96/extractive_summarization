import nltk
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
import rouge

try:
	nltk.data.find('tokenizers/punkt')
except LookupError:
	nltk.download('punkt')

def bleu_dataset(predictions,summaries):
	score = []
	for i in range(len(predictions)):
		sm = len(predictions[i])
		summ = 0
		for sent1 in predictions[i]:
			summ += corpus_bleu(sent1,summaries[i])
		score.append(summ/sm)
	return sum(score)/len(score)
def single_bleu

def rouge_dataset(predictions,summaries):
	evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'])
	scores = evaluator.get_scores(predictions,summaries)




