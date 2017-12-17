import os
import store_word_dict as swd
import csv
# Load the word-vector dictionary
import sys
from word_vec_dict import WordVecDict
import argparse
import eval_help as ev

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--google", required=False, type=bool, default=False, help="Use this flag to include results from google\'s word2vec")
parser.add_argument("-d", "--display_fn", required=False, type=str, default="empty.txt", help="Specify filename with words you'd for which like to display closeness")
parser.add_argument("-n", "--num_evals", required=False, default=25, type=int, help="Specify number of evaluations")
parser.add_argument("-t", "--topn", required=False, default=3, type=int, help="Specify topn for evaluation")
parser.add_argument("-v", "--verbose", required=False, default=False, help="Verbose output")
parser.add_argument("--filename", required=True, help="File full of vectors")

args = parser.parse_args()

print('BEGIN')
eval1 = False # RELSIM dataset
eval2 = True # GOOGLE Analogy dataset
eval3 = True # MEN word sim dataset
eval4 = False # TODO SAT Question dataset

##############################################
##############################################
#				LOAD						 #
##############################################
##############################################

gdic = WordVecDict()
# dict_file_name = 'GoogleNews-vectors-negative300.bin'
# gdic.make_dict(dict_file_name, True)
# dict_file_name = 'GoogleNews-test.txt'
# dict_file_name = '../van8_vecs2.txt'
dict_file_name = '../vectors/vanilla_vectors.txt'

if args.google:
	print('Loading baseline w2v...')
	gdic.make_dict(dict_file_name, False)

jdic = WordVecDict()
dict_file_name = args.filename
print('Loading ' + dict_file_name + ' w2v...')
additional_words = []
jdic.make_dict(dict_file_name, False)
print('DICTS LOADED')
print('Our Vocab length:', len(jdic.get_dict().vocab))
if args.google:
	print('Baseline Vocab length:', len(gdic.get_dict().vocab))

google = "baseline"
us = args.filename.split('/')[1]

##############################################
##############################################
#				EVAL 1						 #
##############################################
##############################################
if eval1:

### load analogies ###
	ratings_file_name = 'relsim/relsim_mean_ratings.csv'
	comp_ratings = []
	with open(ratings_file_name, newline='') as ratings:
	    spamreader = csv.reader(ratings, delimiter=',')
	    
	    for row in list(spamreader)[1:]:
	        comp_ratings.append(row)


	###Attempt to guess word D in an analogy from the dataset.
	print("Evaluating on RELSIM dataset")
	print("Total number of relations", len(comp_ratings))
	print("Evaluating for first", args.num_evals)
	ev.evaluate_relsim(min(len(comp_ratings), args.num_evals), args.topn, comp_ratings, jdic, us)
	if args.google:
		ev.evaluate_relsim(min(len(comp_ratings), args.num_evals), args.topn, comp_ratings, gdic, google)


##############################################
##############################################
#				EVAL 2						 #
##############################################
##############################################
if eval2:
	all_categories = []
	cat_names = []

	questions_fn = 'googlesim/questions-words.txt'
	current_questions = []
	cat_names.append("dummy")

	with open(questions_fn, newline='') as questions:
	    spamreader = csv.reader(questions, delimiter=' ')
	    for row in list(spamreader):
	    	if row[0] != ':':
	    		current_questions.append(row)
	    	else:
	    		cat_names.append(str(row[1]))
	    		all_categories.append(current_questions)
	    		current_questions = []

	phrases_fn = 'googlesim/questions-phrases.txt'
	comp_phrases = []
	all_categories.append(current_questions)
	current_questions = []
	with open(phrases_fn, newline='') as phrases:
		spamreader = csv.reader(phrases, delimiter=' ')
		for row in list(spamreader):
			if row[0] != ':':
				current_questions.append(row)
			else:
				cat_names.append(str(row[1]))
				all_categories.append(current_questions)
				current_questions = []
	i = 0

	print("Evaluating on GOOGLE Analogy dataset:")
	print("Total number of relation categories", len(all_categories))
	print("Evaluating for first", args.num_evals, "in category")

	for section in all_categories:
		l = len(section)
		if cat_names[i] in ['gram1-adjective-to-adverb', 'gram3-comparative', 'gram8-plural', 'gram9-plural-verbs']:
			print("Evaluating for category", cat_names[i])
			ev.evaluate(min(l, args.num_evals), args.topn, section, jdic, us)
			if args.google:
				ev.evaluate(min(l, args.num_evals), args.topn, section, gdic, google)
		i += 1
		if i >= args.num_evals:
			break

##############################################
##############################################
#				EVAL 3						 #
##############################################
##############################################
if eval3:
	sims_fn = 'men/MEN_dataset_lemma_form_full.txt'
	test_sims = []
	with open(sims_fn, newline='') as sims:
	    spamreader = csv.reader(sims, delimiter=' ')   
	    for row in list(spamreader):
	    	if row[0] != ':':
	        	test_sims.append(row)
	        	
	print("Evaluating on MEN Similarity dataset")
	print("Total number of relations", len(test_sims))
	print("Evaluating for first", args.num_evals)
	ev.evaluate_sim(min(args.num_evals, len(test_sims)), test_sims, jdic, us)
	if args.google:
		ev.evaluate_sim(min(args.num_evals, len(test_sims)), test_sims, gdic, google)
	
	