import numpy as np
import time

####Implement the architecture for a deep learning approach to SPL-T:
def get_preds(gameBoard,current_NN,max_splits):
	"""Given a gameboard and a NN, returns probabilities for making each legal
	move on the board given the output of the NN.

	Parameters:
	----------
	gameBoard - Board object
		gameboard that represents the state that probabilities will be generated for
	current_NN - neural network object
		NN that will be used to generate move probabilities
	max_splits - int
		Integer representing the current record number of splits. This will be used to
		scale the value provided by the NN to [-1,1]

	Returns:
	----------
	value - float
		float between 0 and 1, depending on how far away from the best_splits
	probs - numpy array
		array that represents the probabilities of possible moves on the given gameboard.
	boxes - list of ints
		list of ints representing the boxes that can be split at the current state.
		Passed back out of the function to save computation.
	"""
	nn_time = 0
	splt_time = 0

	start = time.time()
	input_array = gameBoard.get_input_array()
	end = time.time()
	splt_time += end-start

	start = time.time()
	preds = current_NN.predict(input_array) #get unprocessed NN results
	end = time.time()
	nn_time += end-start

	#retrieve and scale value
	value_array = preds[0]
	value = value_array[0]
	value = (value*1000-max_splits)/100 #max value is >100 splits past record, min value is >100 splits below
	if value > 1: value = 1
	elif value < -1: value = -1
	
	#retrieve and filter probs
	logits_array = preds[1]
	logits = logits_array[0]

	start = time.time()
	boxes = gameBoard.getMoveOptions()
	allowedActions = [gameBoard.box[box_num].y*8+gameBoard.box[box_num].x for box_num in boxes]
	end = time.time()
	splt_time += end - start

	mask = np.ones(logits.shape,dtype=bool)
	mask[allowedActions] = False
	logits[mask] = -100

	#SOFTMAX
	odds = np.exp(logits)
	probs = odds/np.sum(odds)
	end = time.time()

	return value, probs, boxes, nn_time, splt_time

if __name__ == "__main__":
	gameBoard = core.Board()
	core.makeMove(gameBoard,0)
	core.makeMove(gameBoard,0)
	array = get_gs(gameBoard)