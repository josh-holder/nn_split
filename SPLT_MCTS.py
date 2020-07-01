import numpy as np
import logging
import config
from copy import deepcopy
from SPLT_TF import get_preds
import core
import time
#import loggers as lg

class Node():

	def __init__(self, gameBoard):
		self.id = None #will later be an integer equal to the place which it was added to the tree in
		self.board = gameBoard
		self.edges = []

	def isLeaf(self):
		if len(self.edges) > 0:
			return False
		else:
			return True

class Edge():

	def __init__(self, inNode, outNode, action, prior):
		self.id = str(inNode.id)+"|"+str(outNode.id)
		self.inNode = inNode
		self.outNode = outNode
		self.action = action

		self.stats =  {
					'N': 0,
					'W': 0,
					'Q': 0,
					'P': prior,
				}

class MCTS():
	def __init__(self,root,cpuct,max_splits):
		self.root=root
		self.tree={}
		self.cpuct = cpuct
		self.max_splits = max_splits
		self.num_nodes = 0
		self.addNode(root)

	def __len__(self):
		return(self.num_nodes)
	
	def addNode(self,node):
		#number nodes sequentially by the order in which they were added to the tree, starting at 0
		node.id= self.num_nodes
		self.tree[node.id] = node
		self.num_nodes += 1

	def moveToLeaf(self):
		#print("moving to leaf")
		#lg.logger_mcts.info('------MOVING TO LEAF------')
		start = time.time()
		currentNode = self.root #start at root node
		breadcrumbs = [] #tracks visited nodes on the way to leaf node

		split_num = 0

		while not currentNode.isLeaf():
			#lg.logger_mcts.info('------LOOKING FOR LEAF AT SPLTNUM %d------',split_num)
			split_num += 1
			#print("split_num = " + str(split_num))
			maxQU = -99999

			#calculates max QU
			if currentNode == self.root:
				epsilon = config.EPSILON
				nu = np.random.dirichlet([config.ALPHA] * len(currentNode.edges))
			else:
				epsilon = 0
				nu = [0] * len(currentNode.edges)

			Nb = 0
			for edge in currentNode.edges:
				Nb = Nb + edge.stats['N']

			for idx, edge in enumerate(currentNode.edges):
				#print("Edge: "+edge.id)
				#print("prior = "+str(edge.stats['P']))
				#print("N = "+str(edge.stats['N']))
				U = self.cpuct * \
					((1-epsilon) * edge.stats['P'] + epsilon * nu[idx] )  * \
					np.sqrt(Nb) / (1 + edge.stats['N'])
					
				Q = edge.stats['Q']

				#lg.logger_mcts.info('N = %d, P = %f, nu = %f, adjP = %f, W = %f, Q = %f, U = %f, Q+U = %f'
				#	,edge.stats['N'], np.round(edge.stats['P'],6), np.round(nu[idx],6), ((1-epsilon) * edge.stats['P'] + epsilon * nu[idx] )
				#	, np.round(edge.stats['W'],6), np.round(Q,6), np.round(U,6), np.round(Q+U,6))
				
				#print("Q = "+str(Q))
				#print("U = "+str(U))
				if Q + U > maxQU:
					maxQU = Q + U
					chosenEdge = edge
					#print("edge "+edge.id +" better than old maxQU: "+str(maxQU))
				#else:
					#print("Node QU "+str(Q+U) +" less than old maxQU of "+str(maxQU))
				
			#lg.logger_mcts.info('action with highest Q + U...%d', simulationAction)

			currentNode = chosenEdge.outNode
			breadcrumbs.append(chosenEdge)
			#print("selected edge "+chosenEdge.id + " to continue the search")

		#print("Leaf found:")
		end = time.time()
		move_time = end-start
		return currentNode, breadcrumbs, move_time
	
	
	def backFill(self, leaf, value, breadcrumbs):
		#lg.logger_mcts.info('------DOING BACKFILL------')
		start = time.time()
		for edge in breadcrumbs:
			edge.stats['N'] = edge.stats['N'] + 1
			edge.stats['W'] = edge.stats['W'] + value
			edge.stats['Q'] = edge.stats['W'] / edge.stats['N']

			#lg.logger_mcts.info('updating edge with value %f ... N = %d, W = %f, Q = %f'
			#	, value 
			#	, edge.stats['N']
			#	, edge.stats['W']
			#	, edge.stats['Q']
			#	)
		end = time.time()
		return end-start
	
	def expandLeaf(self,leaf,nn_version):
		"""
		Given a leaf (Node), find the possible moves that can come from this gameBoard.
		Adds nodes for each new gameboard that results from these splits, as well as edges connecting them.
		Calls neural network to get evaluations for the new nodes, adds them to the edges.
		"""
		start_expand = time.time()
		nn_time = 0
		splt_time = 0

		value, probs, moves, pred_nn_time, pred_splt_time = get_preds(leaf.board,nn_version,self.max_splits)
		
		nn_time += pred_nn_time
		splt_time += pred_splt_time

		for move in moves:
			start = time.time()
			box = leaf.board.box[move]
			policy_pos = box.x+box.y*8
			new_board = deepcopy(leaf.board)
			core.makeMove(new_board,move)
			end = time.time()
			splt_time += end-start
			new_node = Node(new_board)
			self.addNode(new_node)
			
			prior = probs[policy_pos]
			#add edge to graph
			new_edge = Edge(leaf,new_node,move,prior)
			leaf.edges.append(new_edge)
		
		end_expand = time.time()
		total_time = end_expand - start_expand

		return value, nn_time, splt_time, total_time

			