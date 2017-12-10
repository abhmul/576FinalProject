import os
import store_word_dict as swd
import csv
# Load the word-vector dictionary
import sys
from word_vec_dict import WordVecDict
import tqdm

def evaluate(num_evals, topn, test_set, dic):
	num_correct = 0
	evaluation = 0
	for row in tqdm(test_set[:num_evals]):
	    evaluation += 1
	    pair1 = row[:2]
	    pair2 = row[2:4]
	    
	    if dic.has_words(*pair1) and dic.has_words(*pair2):
	        pair2_guess_words = dic.get_most_similar(pair1[0], pair1[1], pair2[0], topn)
	        
	        guesses = [t[0] for t in pair2_guess_words]
	        if dic.check_top(guesses, pair2[1], topn):
	            num_correct += 1
	            # print("actual D:", pair2[1])
	            # print("guesses: ", pair2_guess_words)
	    print(evaluation, "done")
	print("Correctness: ",num_correct/float(num_evals)*100)

print('BEGIN')

##############################################
##############################################
#				LOAD						 #
##############################################
##############################################

dic = WordVecDict()
dict_file_name = 'GoogleNews-vectors-negative300.bin'
dic.make_dict(dict_file_name, True)
print('DICT LOADED')
# print(dic.get_dict())

##############################################
##############################################
#				EVAL 1						 #
##############################################
##############################################
eval1 = False
if eval1:

### load analogies ###
	ratings_file_name = 'relsim/relsim_mean_ratings.csv'
	comp_ratings = []
	with open(ratings_file_name, newline='') as ratings:
	    spamreader = csv.reader(ratings, delimiter=',')
	    
	    for row in list(spamreader)[1:]:
	        comp_ratings.append(row)


	###Attempt to guess word D in an analogy from the dataset.
	num_evals = 200
	topn = 3
	print("Evaluating on RELSIM dataset")
	print("Total number of relations", len(comp_ratings))
	print("Evaluating for first", num_evals)

	num_correct = 0
	evaluation = 0
	for row in comp_ratings[:num_evals]:
	    evaluation += 1
	    comp = row[2:]
	    mean_rating = row[10]
	    rel1 = comp[0]
	    rel2 = comp[3]
	    pair1 = comp[1:3]
	    pair2 = comp[4:6]
	    
	    if dic.has_words(*pair1) and dic.has_words(*pair2):
	        reltype1 = int(rel1[:-1])
	        reltype2 = int(rel2[:-1])
	        if reltype1 == reltype2:   # within-type comparisons
	            pair2_guess_words = dic.get_most_similar(pair1[0], pair1[1], pair2[0], topn)
	            
	            guesses = [t[0] for t in pair2_guess_words]
	            if dic.check_top(guesses, pair2[1], topn):
	                num_correct += 1
	                # print("actual D:", pair2[1])
	                # print("guesses: ", pair2_guess_words)

	    print(evaluation, "done")
	                
	print("Correctness: ",num_correct/float(num_evals)*100)

##############################################
##############################################
#				EVAL 2						 #
##############################################
##############################################
eval2 = True
if eval2:

	questions_fn = 'googlesim/questions-words.txt'
	comp_questions = []
	current_questions = []
	all_categories = []
	with open(questions_fn, newline='') as questions:
	    spamreader = csv.reader(questions, delimiter=' ')
	    for row in list(spamreader):
	    	if row[0] != ':':
	    		comp_questions.append(row)
	    		current_questions.append(row)
	    	else:
	    		all_categories.append(current_questions)
	    		current_questions = []

	num_evals = 10
	topn = 10
	print("Evaluating on GOOGLE Analogy dataset: QUESTIONS")
	print("Total number of relations", len(comp_questions))
	print("Evaluating for first", num_evals)
	evaluate(num_evals, topn, comp_questions, dic)


	# phrases_fn = 'googlesim/questions-phrases.txt'
	# comp_phrases = []
	# with open(phrasess_fn, newline='') as phrases:
	#     spamreader = csv.reader(phrases, delimiter=' ')
	#     for row in list(spamreader):
	#     	if row[0] != ':':
	#         	comp_phrases.append(row)

	# num_evals = 10
	# topn = 10
	# print("Evaluating on GOOGLE Analogy dataset: PHRASES")
	# print("Total number of relations", len(comp_phrases))
	# print("Evaluating for first", num_evals)
	# evaluate(num_evals, topn, comp_questions, dic)


##############################################
##############################################
#				EVAL 3						 #
##############################################
##############################################
eval3 = True
if eval3:

	error = 0
	sims_fn = 'men/MEN_dataset_lemma_form_full.txt'
	test_sims = []
	with open(sims_fn, newline='') as sims:
	    spamreader = csv.reader(sims, delimiter=' ')   
	    for row in list(spamreader):
	    	if row[0] != ':':
	        	test_sims.append(row)

	num_evals = 10
	print("Evaluating on MEN Similarity dataset")
	print("Total number of relations", len(test_sims))
	print("Evaluating for first", num_evals)

	for row in tqdm(test_sims[:num_evals]):
	    word1 = row[0]
	    word2 = row[1]
	    real_sim = row[2]
	    
	    if dic.has_words(*pair1) and dic.has_words(*pair2):
	        guess_sim = dic.get_similarity_score(word1, word2) * 50
	        
	        error += abs(real_sim - guess_sim)
	        print(real_sim, guess_sim)
	print("Correctness: ",num_correct/float(num_evals)*100)
	





