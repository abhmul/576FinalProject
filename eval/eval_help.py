from tqdm import tqdm

def evaluate(num_evals, topn, test_set, dic, ntwk_name):
	if not dic.has_dict():
		return
	num_correct = 0
	num_evaled = 0
	for row in tqdm(test_set[:num_evals]):
	    pair1 = row[:2]
	    pair2 = row[2:4]
	    
	    if dic.has_words(*pair1) and dic.has_words(*pair2):
	        pair2_guess_words = dic.get_most_similar(pair1[0], pair1[1], pair2[0], topn)
	        
	        guesses = [t[0] for t in pair2_guess_words]
	        if dic.check_top(guesses, pair2[1], topn):
	            num_correct += 1
	            # print("actual D:", pair2[1])
	            # print("guesses: ", pair2_guess_words)
	        num_evaled += 1

	if num_evaled == 0:
		print('No analogies evaluated.')
		return
	print(ntwk_name, "Correctness: ", num_correct/float(num_evaled)*100, "on", num_evaled)
	return num_correct/float(num_evaled)*100

def evaluate_relsim(num_evals, topn, comp_ratings, dic, ntwk_name):
	num_correct = 0
	num_evaled = 0
	for row in tqdm(comp_ratings[:num_evals]):
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

	            num_evaled += 1
	                
	print(ntwk_name,"Correctness: ", num_correct/float(num_evaled)*100, "on", num_evaled)
	return num_correct/float(num_evaled)*100


def evaluate_sim(num_evals, test_sims, dic, ntwk_name):
	error = 0
	num_evaled = 0
	for row in tqdm(test_sims[:num_evals]):
	    word1 = row[0][:-2]
	    word2 = row[1][:-2]
	    real_sim = float(row[2])
	    
	    if dic.has_words(word1) and dic.has_words(word2):
	        guess_sim = dic.get_similarity_score(word1, word2) * 50
	        
	        error += abs(real_sim - guess_sim)
	        # print(real_sim, guess_sim)
	        num_evaled += 1

	print(ntwk_name, "Avg Error: ", error/(float(num_evaled)*50))
	print(ntwk_name, "Correctness: ", 100 * (1 - error / (float(num_evaled) * 50)))
	return 100 * (1 - error / (float(num_evaled) * 50))

