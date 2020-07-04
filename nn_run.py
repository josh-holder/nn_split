from __future__ import absolute_import, division, print_function, unicode_literals
#from SPLT_TF import get_gs
import core
from splt_mcts import Node, Edge, MCTS
from splt_model import Residual_CNN
import nn_config
from copy import deepcopy, copy
import random
import os
import atexit
import argparse
import numpy as np
np.set_printoptions(suppress=True)
import pickle
from collections import deque
import time
import threading
from multiprocessing import cpu_count
from tensorflow import keras
from splt_loss import softmax_cross_entropy_with_logits
import pandas as pd

def _build_parser():
	parser = argparse.ArgumentParser(description='Run NN to generate ideal playing strategy for SPLT.')

	parser.add_argument(
		'-r','--run_name',
		help="Will name the output directory for the results of the run",
		type=str,
		default="run",
	)

	return parser

def make_move_MCTS(gameBoard,nn_version,choose_best,split_num):
	"""
	Given a board state, chooses a next split to make using NN-guided MCTS.
	
	Parameters:
	----------
	gameBoard - Board object 
		the gameBoard object representing the current state of the board
	choose_best - boolean
		the placeholder for temperature. When choose_best = True, the most visited node is chosen as the next move,
		while when choose_best = False, the node is visited randomly according to the visitation distribution
	Returns:
	----------
	split - int
		an integer representing the box that must be split
	search_probs - (16*8,1) array
		list containing the results of the move. Each entry in the list is a tuple
		corresponding to a move in the position, of the form (number of visits, action number)
	"""
	start = time.time()
	root = Node(gameBoard)
	mcts = MCTS(root,nn_config.CPUCT,current_max_splits)
	search_probs = np.zeros((16*8,1),dtype=float)

	nn_time = 0
	splt_time = 0
	backfills_time = 0
	exs_tot_time = 0
	moves_time = 0

	for i in range(nn_config.MCTS_SEARCH_LEN):
		#print("Rollout "+str(i+1)+"/"+str(nn_config.MCTS_SEARCH_LEN),end='\r')
		leaf, breadcrumbs, move_time = mcts.moveToLeaf()
		value, ex_nn_time, ex_splt_time, ex_tot_time = mcts.expandLeaf(leaf,nn_version)
		backfill_time = mcts.backFill(leaf,value,breadcrumbs)

		nn_time += ex_nn_time
		splt_time += ex_splt_time
		exs_tot_time += ex_tot_time
		moves_time += move_time
		backfills_time += backfill_time
		

	#creates array with the probabilities of each box in the array
	for move in root.edges:
		split_box = root.board.box[move.action]
		prob = move.stats['N']/(nn_config.MCTS_SEARCH_LEN-1)
		prob_pos = split_box.y*8+split_box.x
		
		search_probs[prob_pos] = prob


	split, choose_time = choose_split(root,choose_best)
	
	end = time.time()
	rollout_time = end-start
	"""
	print("SPLIT "+str(split_num)+" ROLLOUT TIMING STATS:")
	print("- Total rollout time:",rollout_time,"-\n")
	print("Total moveToLeaf time:",moves_time,moves_time/rollout_time*100,"%")
	print("Total expandLeaf time:",exs_tot_time,exs_tot_time/rollout_time*100,"%")
	print("Total backFill time:",backfill_time,backfill_time/rollout_time*100,"%")
	print("Total choose_split time:",choose_time,choose_time/rollout_time*100,"%\n")
	print("Total time spent evaluating NN:",nn_time,nn_time/rollout_time*100,"%")
	print("Total time spent updating SPLT:",splt_time,splt_time/rollout_time*100,"%")
	overhead_time = rollout_time - nn_time - splt_time
	print("Total time spent on other MCTS overhead:",overhead_time,\
		overhead_time/rollout_time*100,"%\n")
	"""
	return split, search_probs, nn_time, splt_time, rollout_time

def choose_split(node,choose_best):
	"""Chooses the next move to make based on the results of the MCTS+NN rollout.
	
	Parameters:
	----------
	node - Node object
		The node with information on next moves and their visitation statistics
	choose_best - bool
		Placeholder for temperature. When choose_best = True, the most visited node is chosen as the next move,
		while when choose_best = False, the node is visited randomly according to the visitation distribution
	
	Returns:
	----------
	split - int
		an integer representing the index of the move that should be made
	"""
	start = time.time()
	if choose_best: #chooses highest n value of move options
		max_n = 0
		best_moves = [] #creates list of all moves that are visited the most. Randomly chooses a best move at the end
		for move in node.edges:
			if move.stats['N'] > max_n:
				max_n = move.stats['N']
				best_moves = [move.action]
			elif move.stats['N'] == max_n:
				best_moves.append(move.action)
		split = random.choice(best_moves)
	else: #chooses random n value based on their numbers
		n_list = []
		action_list = []
		for move in node.edges:
			n_list.append(move.stats['N'])
			action_list.append(move.action)
		remainingDistance = random.random()*sum(n_list) #chooses by random walk
		for i,weight in enumerate(n_list):
			remainingDistance -= weight
			if remainingDistance < 0:
				split = action_list[i]
				break
	end = time.time()
	choose_time = end-start
	return split, choose_time

def play_games(nn_version,j,titer):
	"""
	Given a NN version, plays 1000 games of SPLT from start to finish and records their states, using a MCTS guided by nn_version.
	Saves the states in memory.

	Parameters:
	----------
	nn_version - NN object
		NN object that will be playing the game

	Returns:
	----------
	memory - deque object
		Effectively a list of states. Each state is a list of the form [gameBoard, search_probs, splits]
	"""
	global new_max_splits, memory, lck

	timedf_dict = {}

	for game_num in range(int(nn_config.NUM_SELFTRAIN_GAMES/cpu_count())):
		print("PLAYING GAME NUM "+str(game_num)+" ON CORE "+str(j))
		gameBoard = core.Board()
		core.makeMove(gameBoard,0) #makes first trivial split
		split = 1

		split_sequence = [0] #in case this ends up being a next best sequence
		states = [] #stores each state in the game
		
		nn_times = []
		splt_times = []
		total_times = []

		while len(gameBoard.getMoveOptions()) != 0:
			if split>=30: choose_best = True
			else: choose_best = False
			#print("-------------")
			print("Game "+str(game_num) + ", Split "+str(split)+" on core "+str(j))
			move, search_probs, nn_time, splt_time, rollout_time =\
				make_move_MCTS(gameBoard,nn_version,choose_best,split)

			nn_times.append(nn_time)
			splt_times.append(splt_time)
			total_times.append(rollout_time)

			split_sequence.append(move)
			gameState = np.squeeze(gameBoard.get_input_array())
			states.append([gameState,search_probs])

			core.makeMove(gameBoard,move)

			split += 1
		
		#add final split amount to state information, then save it in memory
		for state in states:
			state.append(split/1000)
		
		lck.acquire()
		memory.extend(states) #add the new states to memory

		if split >= new_max_splits: #save a sequence of new maximum length
			new_max_splits = split
			pth = os.path.join(os.getcwd(),args.run_name,"highest_splits.txt")
			with open(pth,'a') as f:
				f.write(str(split)+": " + str(split_sequence)+'\n')

		lck.release()

		game_df = pd.DataFrame(data={'nn_time':nn_times,'splt_time':splt_times,'total_time':total_times})
		timedf_dict[game_num] = game_df
	
	time_df = pd.concat(timedf_dict,axis=1)
	time_path = args.run_name+'/trainiter'+str(titer)+'_core'+str(j)+'_timedata_df.pkl'
	with open(time_path,'wb') as f:
		pickle.dump(time_df,f)

def retrain(nn_version,train_overall_loss,train_value_loss,train_policy_loss):
	"""Given the nn_version and the memory, selects a random minibatch of games
	to train with, and retrains the NN accordingly.

	Parameters:
	----------
	nn_version - NN object
		The version of the NN that you are aiming to retrain
	train_overall_loss - list of floats
		tracks the average loss of the NN for each retraining step
	train_value_loss - list of floats
		tracks the loss of the value function of the NN for each retraining step
	train_policy_loss - list of floats
		tracks the loss of the policy function of the NN for each retraining step
	
	Returns:
	----------
	nn_retrained - NN object
		The new version of the NN, with retrained weights
	train_overall_loss - list of floats
		updated overall loss history
	train_value_loss - list of floats
		updated value loss history
	train_policy_loss - list of floats
		updated policy loss history
	"""
	nn_retrained = copy(nn_version)

	for i in range(nn_config.TRAINING_LOOPS):
		minibatch = random.sample(memory,min(nn_config.BATCH_SIZE, len(memory))) #selects BATCH_SIZE unique states to retrain the NN on
		training_states = np.array([state[0] for state in minibatch]) #states that will be evaluated with NN
		training_targets = {'value_head': np.array([state[2] for state in minibatch])
							, 'policy_head': np.squeeze(np.array([state[1] for state in minibatch]))}

		"""
		print("--Input shapes:--")
		print(np.shape(minibatch))
		print("Value head shape:")
		print(np.shape(training_targets['value_head']))
		print("Policy head shape:")
		print(np.shape(training_targets['policy_head']))
		print(training_targets['policy_head'])
		"""

		fit = nn_retrained.model.fit(training_states,training_targets,epochs=nn_config.EPOCHS,verbose=1,validation_split=0,batch_size=32)
		print("New loss: "+str(fit.history))

		train_overall_loss.append(round(fit.history['loss'][nn_config.EPOCHS - 1],4))
		train_value_loss.append(round(fit.history['value_head_loss'][nn_config.EPOCHS - 1],4))
		train_policy_loss.append(round(fit.history['policy_head_loss'][nn_config.EPOCHS - 1],4)) 
	
	return nn_retrained, train_overall_loss, train_value_loss, train_policy_loss

def learn():
	global current_max_splits,new_max_splits,memory

	try: os.mkdir(args.run_name)
	except FileExistsError: pass

	mem_path = os.path.join(os.getcwd(),args.run_name,"game_memory.pkl")
	if os.path.exists(mem_path):
		print("Loading existing memory.")
		with open(mem_path,'rb') as f:
			memory = pickle.load(f)
	else:
		print("Previous memory not found: generating empty memory list.")
		memory = deque(maxlen=nn_config.MEMORY_SIZE)

	nn_path = os.path.join(os.getcwd(),args.run_name,'best_nn.h5')
	if os.path.exists(nn_path):
		print("Loading existing NN.")
		current_NN = keras.models.load_model(nn_path,custom_objects={'softmax_cross_entropy_with_logits': softmax_cross_entropy_with_logits})
	else:
		print("Previous NN not found: generating NN from scratch.")
		current_NN = Residual_CNN(nn_config.REG_CONST, nn_config.LEARNING_RATE, (16,8,3), 16*8, nn_config.HIDDEN_CNN_LAYERS)
		current_NN.model.save(nn_path)
	
	split_hist_path = os.path.join(os.getcwd(),args.run_name,"split_history.pkl")
	if os.path.exists(split_hist_path):
		print("Loading existing NN split history.")
		with open(split_hist_path,'rb') as f:
			split_hist = pickle.load(f)
		current_max_splits = max(split_hist)
	else:
		print("Previous NN split history not found.")
		split_hist = [0]
		current_max_splits = max(split_hist)

	new_max_splits = int(current_max_splits) #variable which makes sure to track any new records that might be obtained along the training process

	train_overall_loss = []
	train_value_loss = []
	train_policy_loss = []

	loop = True
	global lck
	lck = threading.Lock()

	i = 1

	while loop: #run training loop
		print("---TRAINING LOOP "+str(i)+"---")

		threads = []
		start_selfplay = time.time()
		#Playing games against self to generate memory
		for j in range(cpu_count()):
			print("Starting MCTS thread:")
			t = threading.Thread(target=play_games,args=(current_NN,j,i))
			threads.append(t)
			t.start()
		
		for t in threads:
			t.join()

		end_selfplay = time.time()

		print(str(nn_config.NUM_SELFTRAIN_GAMES) + " games completed in " + str(end_selfplay-start_selfplay)+" seconds.")

		if len(memory) >= nn_config.MEMORY_SIZE:
			print("Enough games in memory: updating NN based on this data.")
			start_update = time.time()
			new_NN,train_overall_loss,train_value_loss,train_policy_loss = retrain(current_NN,train_overall_loss,train_value_loss,train_policy_loss)
			end_update = time.time()
			print("Retraining completed in " + str(end_update-start_update)+" seconds.")

			if new_max_splits > current_max_splits:
				print("This version of the NN is better:")
				print("NN record = "+str(new_max_splits)+", old record = "+str(current_max_splits))
				current_NN = new_NN
				current_max_splits = int(new_max_splits)
			else:
				print("This version of the NN is not better.")
				print("NN record = "+str(new_max_splits)+", old record = "+str(current_max_splits))
				new_max_splits = int(current_max_splits)

			print("Saving NN loss, split record data...")
			#save nn split record history
			with open(split_hist_path,'wb') as f:
				pickle.dump(split_hist,f)
			
			#save loss data
			loss_data = [train_overall_loss,train_value_loss,train_policy_loss]
			loss_path = os.path.join(os.getcwd(),args.run_name,"loss_history.pkl")
			with open(loss_path,'wb') as f:
				pickle.dump(loss_data,f)
		else:
			print("Only "+str(len(memory))+" states in memory. Will continue to selfplay before evaluating.")

		#save memory
		print("Saving new information to memory.")
		with open(mem_path,'wb') as f:
			pickle.dump(memory,f)

		#save NN
		current_NN.model.save(nn_path)
		
		i += 1

if __name__ == "__main__":
	parser = _build_parser()
	args = parser.parse_args()
	learn()
	"""
	global current_max_splits
	current_max_splits = 0
	current_NN = Residual_CNN(nn_config.REG_CONST, nn_config.LEARNING_RATE, (16,8,3), 16*8, nn_config.HIDDEN_CNN_LAYERS)
	play_games(current_NN,0)
	"""

def exit_handler():
	"""
	Saves the memory in case of a premature stoppage.
	"""
	pth = os.path.join(os.getcwd(),args.run_name,'game_memory.pkl')
	with open(pth,'wb') as f:
		pickle.dump(memory,f)

atexit.register(exit_handler)









