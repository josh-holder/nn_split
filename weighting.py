import math
import copy
import time
from collections import defaultdict

weightverbose = 0
timingInfo = 0
HORIZONTAL='-'
VERTICAL='|'

def findCluster(gameBoard):
	#Given a gameBoard, finds the clusters that would be caused by a split of any box in the board
	#INPUT: gameBoard, the gameboard object
	#affectedColumns, a list of boolean values that determines if boxes in a given column need to have their clusters updated
	#OUTPUT: clusters, a list of sets. Each inner list contains the indices of boxes that are involved in a cluster when a given box is split.
	#The inner list is an empty list when splitting a given box results in no cluster

	if timingInfo: start = time.time()
	clusters = []

	#iterates through every box in the board and splits them
	for boxToBeCheckedIndex, boxToBeChecked in enumerate(gameBoard.box):
		#only operates updates boxes that could have had their splits affected
		
		if boxToBeChecked.splitPossible(gameBoard.splitAction):
			newClusters = set()

			newgameBoard = copy.deepcopy(gameBoard)

			copiedboxToBeChecked = newgameBoard.box[boxToBeCheckedIndex] #need to redefine the new copied object within the new copied gameboard

			newgameBoard.split(copiedboxToBeChecked) #creates a copy of the board with the given box split

			lastCreatedBox=newgameBoard.box[-1]

			#iterate through every box in the new list of boxes to see if it is in the upper left corner of a cluster
			for clusterBoxIndex, clusterBox in enumerate(newgameBoard.box):
				if clusterBox == lastCreatedBox:
					

					clusterMembers = set()
					clusterMembers.add(clusterBoxIndex)

					for otherBoxIndex, otherBox in enumerate(newgameBoard.box):
						if otherBox == lastCreatedBox:

							# If otherbox is beside box
							if otherBox.x==(clusterBox.x+clusterBox.width) and otherBox.y==clusterBox.y:	clusterMembers.add(otherBoxIndex)
							# If otherbox is diagonal to box
							elif otherBox.x==(clusterBox.x+clusterBox.width) and otherBox.y==(clusterBox.y+clusterBox.height):	clusterMembers.add(otherBoxIndex)
							# If otherbox is below box
							elif otherBox.x==clusterBox.x and otherBox.y==(clusterBox.y+clusterBox.height):	clusterMembers.add(otherBoxIndex)
					
					#This catches groups of 6 and 8 boxes as multiple groups of 4 boxes, which just doublecounts boxes but does not negatively impact accuracy
					if len(clusterMembers) == 4:
						newClusters = newClusters.union(clusterMembers)
			clusters.append(newClusters)
		
		#otherwise, keep the old.
		else:
			clusters.append(set())
			
		
	if timingInfo: 
		end = time.time()
		print("Time to find clusters for turn "+str(len(gameBoard.splitRecord))+": "+str(end-start))
	return clusters

def findSplitsUntilFall(gameBoard):
	"""
	Given a gameboard object, finds the points until each block falls.
	INPUT: gameBoard object
	OUTPUT: splitsUntilFall, a dictionary with form {blockIndex:splits_until_fall}
	pointBlocksBelow, a dictionary with form {blockIndex:[points of point blocks below]}
	"""
	if timingInfo: start = time.time()
	splitsUntilFall = {}
	pointBlocksBelow = defaultdict(list)

	explosionImminent = 0
	for box in gameBoard.box:
		if box.points == 1:
			explosionImminent = 1

	sufverbose = 0

	if sufverbose: print("Initializing splitsUntilFall of bottom row")

	#initialize the bottom row first
	for boxIndex in gameBoard.boxesInRow[gameBoard.height]:
		box = gameBoard.box[boxIndex]
		splitsUntilFall[boxIndex] = box.points
		pointBlocksBelow[boxIndex] = [box.points]

	#iterate through every other row, box by box
	for row in range(gameBoard.height-1,0,-1):
		if sufverbose: print("~~~Finding splits until fall of row " + str(row)+ "~~~")

		for boxIndex in gameBoard.boxesInRow[row]:
			if sufverbose: print("Calculating for box index " + str(boxIndex))

			box = gameBoard.box[boxIndex]
			maxSplitsToFall = 0

			#iterates through boxes in row directly below
			for otherBoxIndex in gameBoard.boxesInRow[row+1]:
				otherBox = gameBoard.box[otherBoxIndex]

				#SPECIAL CASE: When the split is vertical AND the box does not have points AND there is a point block about to explode AND a cluster would be formed, 
				#it actually will fall in two parts and matter
				#if explosionImminent and box.points == 0 and gameBoard.splitAction==VERTICAL:
					#find the splitsuntilfall for both sides of the block seperately, then take the smaller one of the two and make that splitsUntillFall.
					#NEEDS TO BE IMPLEMENTED, probably doesnt matter in most scenarios

				#only selects boxes in the correct columns to be below the box you're looking at
				if otherBox.x+otherBox.width > box.x and otherBox.x < box.x+box.width:
					if sufverbose: print("Box with index " + str(otherBoxIndex) + " is directly below. This box has " + str(splitsUntilFall[otherBoxIndex]) + " splits until fall.")
					
					if splitsUntilFall[otherBoxIndex] == 0:
						if sufverbose: print("This box below will not fall, so therefore the box above will not fall either.")
						maxSplitsToFall = 0
						break
					#finds box directly under the current box that will fall the latest
					if splitsUntilFall[otherBoxIndex] > maxSplitsToFall:
						if sufverbose: print("This is the new high splits until fall, so we update the max.")
						maxSplitsToFall = splitsUntilFall[otherBoxIndex]
						pointBlocksBelow[boxIndex] = copy.copy(pointBlocksBelow[otherBoxIndex])
					
					
			


			#the box with fall when its points run out, or a block below falls: whichever comes first. However, we only want to do this when the points are nonzero.
			if box.points != 0 and maxSplitsToFall != 0:

				#if it was also a point block, just one of higher value than the minimum, add it's points to pointBlocksBelow
				if box.points > maxSplitsToFall and box.points not in pointBlocksBelow[boxIndex]:
					pointBlocksBelow[boxIndex].append(box.points)

				splitsUntilFall[boxIndex] = min([box.points,maxSplitsToFall])
			else:
				splitsUntilFall[boxIndex] = maxSplitsToFall

			if sufverbose: print("We choose the minimum between " + str(box.points)+" and "+str(maxSplitsToFall)+" (excluding zero), yielding "+str(splitsUntilFall[boxIndex])+" for box "+str(boxIndex))
	
	if timingInfo: 
		end = time.time()
		print("Time to find splits until fall for turn "+str(len(gameBoard.splitRecord))+": "+str(end-start))

	return splitsUntilFall, pointBlocksBelow
	


def findWeights(gameBoard):
	"""
	Given a gameboard and thus the list of boxes in the gameboard, gameBoard.box, finds the weights of each box in the gameboard.
	Factors that are considered:
	-Split Imbalance: the difference between the number of vertical and horizontal lights in the board, in proportion to the total splits,
	is considered. For example, creating many vertical splits with a split when there are already many more vertical split opportunities on the board
	recieves a negative weight.
	-Aspect Ratio: An aspect ratio of 1 is ideal. Splitting blocks such that the aspect ratio increases recieves a penalty. 
	No bonus is recieved for lowering the aspect ratio.
	-Point Block Locations: creating clusters above point blocks that are about to explode recieve large positive weights.
	Splitting boxes without creating clusters is still prioritized above point blocks that are about to explode
	-Split Height: Splitting blocks and making point blocks at lower locations on the board is prioritized
	INPUTS: gameBoard, the current gameBoard object
	OUTPUTS: weights, a list of numbers corresponding to the statistical weighting of a given box being split
	"""
	if timingInfo: start = time.time()

	offset = 10
	weights = []

	total_splits = gameBoard.hor_splits+gameBoard.ver_splits
	splitImbalance = gameBoard.hor_splits-gameBoard.ver_splits

	splitsUntilFall,pointBlocksBelow = findSplitsUntilFall(gameBoard)

	if weightverbose: 
		print("_________________________________________________________")
		print("Calculating weights for move "+str(len(gameBoard.splitRecord)))
		print("_________________________________________________________")

	for boxToBeWeightedIndex, boxToBeWeighted in enumerate(gameBoard.box):
		#only assign a weight to boxes that don't already have points and that can be split this turn
		if boxToBeWeighted.points == 0 and boxToBeWeighted.splitPossible(gameBoard.splitAction):
			weight = 0 #default weight

			createPoints = boxToBeWeightedIndex in gameBoard.clusters[boxToBeWeightedIndex]

			#----------------------IMBALANCE WEIGHT-------------------------------- 
			#weight depending on the balance of horizontal to vertical splits in the board

			if weightverbose: 
				print(" ")
				print("BLOCK "+str(boxToBeWeightedIndex))
				print("~~~Finding weight from imbalance for block "+str(boxToBeWeightedIndex)+":~~~") 
				print(str(splitImbalance) + " more hor than ver splits for split " + str(len(gameBoard.splitRecord)))
				

			if gameBoard.splitAction == HORIZONTAL:
				#ensures that splitting the box wouldnt make a cluster: if it does, this weight is irrelevant
				if not createPoints:
					verticalSplitsToAdd = (boxToBeWeighted.width-1)
					if weightverbose: print("Splitting box " + str(boxToBeWeightedIndex) + " would create " + str(verticalSplitsToAdd) + " new vertical splits.")

					#if there are more vertical than horizontal splits, negative weighting increases linearly with vertical splits added
					if splitImbalance <= 0:
						
						weightToAdd = (-15/7)*verticalSplitsToAdd #Eyeballed to provide -15 weight when adding the max
						#Scales based off how urgent the imbalance is relative to how many total splits are left. 2x weight when the imbalance is 50% of the total splits left
						weightToAdd = weightToAdd*(4*abs(splitImbalance)/total_splits)

						if weightverbose: print("More vertical than horizontal splits, so we add weight of " + str(weightToAdd))

					#if there are more horizontal than vertical splits, positive weighting follows parabola
					elif splitImbalance >0:
						#quadratic regression is eyeballed to be 0 at 0%, 5 at 200%, 15 at 100% 
						percent = verticalSplitsToAdd/abs(splitImbalance)
						weightToAdd = -12.5*percent**2+27.5*percent

						#Scales based off how urgent the imbalance is relative to how many total splits are left. 2x weight when the imbalance is 50% of the total splits left
						weightToAdd = weightToAdd*(4*abs(splitImbalance)/total_splits)

						if weightverbose: print("More horizontal than vertical splits, so we add weight of " + str(weightToAdd))
					
					weight += weightToAdd

				else:
					if weightverbose: print("Splitting this box would cause a cluster, so we don't weight it.")	
					pass
				
				#
			
			#calculate imbalance weight for vertical splits
			else:
				#ensures that splitting the box wouldnt make a cluster: if it does, this weight is irrelevant
				if not createPoints:
					horizontalSplitsToAdd = (boxToBeWeighted.height-1)
					if weightverbose: print("Splitting box " + str(boxToBeWeightedIndex) + " would create " + str(horizontalSplitsToAdd) + " new horizontal splits.")

					#if there are more horizontal than vertical splits, negative weighting increases linearly with horizontal splits added
					if splitImbalance >= 0:
						
						weightToAdd = (-15/7)*horizontalSplitsToAdd #Eyeballed to provide -15 weight when adding the 7 new horizontal splits
						#Scales based off how urgent the imbalance is relative to how many total splits are left. 2x weight when the imbalance is 50% of the total splits left
						weightToAdd = weightToAdd*(4*abs(splitImbalance)/total_splits)

						if weightverbose: print("More horizontal than vertical splits, so we add weight of " + str(weightToAdd))

					#if there are more vertical than horizontal splits, positive weighting follows parabola
					elif splitImbalance <0:
						#quadratic regression is eyeballed to be 0 at 0%, 5 at 200%, 15 at 100% 
						percent = horizontalSplitsToAdd/abs(splitImbalance)
						weightToAdd = -12.5*percent**2+27.5*percent

						#Scales based off how urgent the imbalance is relative to how many total splits are left. 2x weight when the imbalance is 50% of the total splits left
						weightToAdd = weightToAdd*(4*abs(splitImbalance)/total_splits)

						if weightverbose: print("More vertical than horizontal splits, so we add weight of " + str(weightToAdd))
					
					weight += weightToAdd

				else:
					if weightverbose: print("Splitting this box would cause a cluster, so weighting based on imbalance is irrelevant.")
					pass
			

			#----------------------------------ASPECT RATIO WEIGHT---------------------------------
			if weightverbose: 
				print("Total weight is now "+str(weight))
				print("~~~Finding weight from aspect ratio:~~~")

			aspectRatio = max(boxToBeWeighted.width,boxToBeWeighted.height)/min(boxToBeWeighted.width,boxToBeWeighted.height)

			weightToAdd = 0
			if gameBoard.splitAction == HORIZONTAL:
				newaspectRatio = max(boxToBeWeighted.width,boxToBeWeighted.height/2)/min(boxToBeWeighted.width,boxToBeWeighted.height/2)
			else:
				newaspectRatio = max(boxToBeWeighted.width/2,boxToBeWeighted.height)/min(boxToBeWeighted.width/2,boxToBeWeighted.height)
			
			#only applies a penalty to making a worse aspect ratio
			if newaspectRatio - aspectRatio > 1:
				#eyeballed to remove 10 points when taking an aspect ratio from 4 to 8, remove 2 points when taking an aspect ratio from 2 to 4
				weightToAdd = -.75*(newaspectRatio - aspectRatio)**2+.5*(newaspectRatio - aspectRatio)
				if weightverbose: 
					print("Aspect ratio is getting worse, going from "+str(aspectRatio)+" to "+str(newaspectRatio))
					print("We add a penalty weight of "+str(weightToAdd))
			else:
				if weightverbose: print("Aspect ratio is getting no worse, so no penalty is added")
			
			weight += weightToAdd


			#--------------------------------------POINT BLOCK WEIGHT------------------------------------
			#Weight based on how soon the next point block explosion in the boxes column will occur
			if weightverbose: 
				print("Total weight is now "+str(weight))
				print("~~~Finding weight from point block explosions:~~~")
				print("There are "+str(splitsUntilFall[boxToBeWeightedIndex])+" splits until the box falls.")


			weightToAdd = 0
			#only assigns weight if the box is sitting on a point block
			if splitsUntilFall[boxToBeWeightedIndex] != 0:
				
				extraSplitsBelow = len(pointBlocksBelow[boxToBeWeightedIndex])-1
				difference = (len(gameBoard.splitRecord)+1) - (pointBlocksBelow[boxToBeWeightedIndex][-1]-1) 
				

				#now we add weight to boxes differently depending on whether splitting the box will cause a cluster.
				if createPoints:
					#Determine if adding another cluster in this row will cause another halving to occur - only happens when the difference is more than 2^number of halvings
					#If so, apply a small extra weight to splitting this  box, and a large extra weight to the box if splitting it would cause a cluster
					#weight this extra weight based on the number of splits left until the block falls. Use ceil(splitsUntilFall/2)

					if weightverbose: print("There are "+str(extraSplitsBelow)+" extra halvings occuring on this box before it reaches 0. The difference between the next lowest point block in the column is "+str(difference)+".")

					if difference >= 2**extraSplitsBelow:

						if weightverbose: print("This difference is enough to cause another halving, so we increase the block's weight.")

						#eyeballed to add 40 points of weight when this is the last chance to create the cluster, exponentially decreasing to 20 points when it is not urgent
						#If the difference is less than 2^(splitsUntilFall+1), then decrease the weight increase by 15% because its harder to 
						#take full advantage of the double split, but it's still a double split
						turns = math.ceil(splitsUntilFall[boxToBeWeightedIndex]/2)

						weightToAdd = (29.53/(1+.1518*math.exp(1.14*turns)))+20

						if weightverbose: 
							print("Splitting this box WILL cause a cluster, so we add weight accordingly.")
							print("There are "+str(turns)+" opportunities to make this cluster until the split is made, so we add "+str(weightToAdd)+" weight.")

						if difference < 2**(extraSplitsBelow+1):
							weightToAdd = weightToAdd*.85
							if weightverbose: print("-Difference is small enough that the blocks will end up one apart. We penalize the weight by 15%, yielding: "+str(weightToAdd))
					
					else:
						if weightverbose: print("Making this point block will not lead to another halving, so we assign no extra weight to it.")
				
				#otherwise, if splitting the box won't cause a cluster, we prioritize splitting boxes in this row, but less aggressively
				else:
					#Eyeball the weight to be +10 when the box has 10 turns to fall, decreasing quadratically to 0 at 2 splits and 18 splits
					if 2 < splitsUntilFall[boxToBeWeightedIndex] <= 18:
						#weightToAdd = -.15625*splitsUntilFall[boxToBeWeightedIndex]**2+3.125*splitsUntilFall[boxToBeWeightedIndex]-5.625
						weightToAdd = -.15625*splitsUntilFall[boxToBeWeightedIndex]**2+3.125*splitsUntilFall[boxToBeWeightedIndex]-5.625
					else:
						weightToAdd = 0
					
					if weightverbose: 
						print("Splitting this box WILL NOT cause a cluster, so we add weight accordingly.")
						print("There are "+str(splitsUntilFall[boxToBeWeightedIndex])+" splits until the block is halved, so we add "+str(weightToAdd)+" weight.")

					#if you're too late to make this split, calculate based on the next split that will happen
					if splitsUntilFall[boxToBeWeightedIndex] <=2 and len(pointBlocksBelow[boxToBeWeightedIndex]) > 1:
						newPointBlocksBelow = pointBlocksBelow[boxToBeWeightedIndex][1:]
						tillFall = min(newPointBlocksBelow)
						
						#weight the same way as above, but using the new number of splits until fall
						if 2 < tillFall <= 18:
							weightToAdd = -.15625*splitsUntilFall[boxToBeWeightedIndex]**2+3.125*splitsUntilFall[boxToBeWeightedIndex]-5.625
						else:
							weightToAdd = 0
						
						if weightverbose:
							print("This block will fall in "+str(splitsUntilFall[boxToBeWeightedIndex])+" splits, so there's no chance we make a point block before this.")
							print("Entire list of point blocks beneath, minus the lowest:")
							print(newPointBlocksBelow)
							print("New splits until falling: "+str(tillFall))
							print("Using this number of splitsUntilFall, we add "+str(weightToAdd)+" weight.")	
						
			else:
				if weightverbose: print("This block is sitting on no point blocks, so we assign no extra split weight.")

			weight += weightToAdd

			if weightverbose: 
				print("Total box weight is now "+str(weight))
				print("~~~Finding weight from height:~~~")

			#--------------------------HEIGHT WEIGHT-----------------------------
			#Adds bonus weight of up to 7.5 when splitting blocks at the bottom of the board. 
			#This factor is extra important if the split would create a cluster. (200%)

			if createPoints:

				#determine the lowest height of the cluster that is created 
				maxHeight = 0
				for boxIndex in gameBoard.clusters[boxToBeWeightedIndex]:

					#clusters includes the box that would be created in the split in the list, so if it tries to use this we just ignore it
					try:
						candHeight = gameBoard.box[boxIndex].y + gameBoard.box[boxIndex].height
					except:
						candHeight = 0
						pass
					
					if candHeight> maxHeight:
						maxHeight = candHeight

				weightToAdd = 20*(maxHeight/gameBoard.height)**2

				if weightverbose: 
					print("This will create point blocks with a bottom at "+str(maxHeight)+".")
					print("Add " + str(weightToAdd) + " weight.")
			
			else:
				weightToAdd = 10*((boxToBeWeighted.y+boxToBeWeighted.height)/gameBoard.height)**2

				if weightverbose: 
					print("Bottom of the block is at "+str(boxToBeWeighted.y+boxToBeWeighted.height)+", but will not create a point block.")
					print("Add " + str(weightToAdd) + " weight.")
			
			weight += weightToAdd

			if weightverbose: 
				print("Total box weight is now "+str(weight))
				print("~~~Finding weight from point block size:~~~")

			#----------------------POINT BLOCK SIZE WEIGHT----------------------------------
			if createPoints:
				#determine how many total points are left in the board
				max_splits = 0
				for box in gameBoard.box:
					if box.points == 0:
						max_splits += box.width*box.height-1

				#determine how many squares the cluster will be
				if gameBoard.splitAction == HORIZONTAL:
					splitsAvailable = (box.width*box.height)-1
				else:
					splitsAvailable = (box.width*box.height)-1

				clusterSquares = len(gameBoard.clusters[boxToBeWeightedIndex])/2*splitsAvailable

				#if clustersquares is not the minimum (4), apply a penalty based on the proportion 
				# of the remaining squares that the cluster takes up
				if clusterSquares > 4:
					weightToAdd = -60*(clusterSquares/total_splits)*(len(gameBoard.splitRecord)/200)
				else:
					weightToAdd=0
				
				weight += weightToAdd

				if weightverbose:
					print("This cluster takes up "+str(clusterSquares)+" available splits.")
					print("There are "+str(max_splits)+" splits left on the board.")
					print("We add "+str(weightToAdd)+" to the weight.")
					print("Final weight value: "+str(weight))

			else:
				if weightverbose: print("This block doesn't create point blocks, so we do nothing.")
			
			#----------------------UNEVEN CLUSTER WEIGHT PENALTY----------------
			#determines if the cluster is a group of 6
			if len(gameBoard.clusters[boxToBeWeightedIndex]) == 6:
				weightToAdd = -10
			else:
				weightToAdd = 0

			weight += weightToAdd


			weights.append(weight)


		else:
			weights.append(0)

	#to prevent negative weights, we take the lowest weight and add it to the entire list, making the worst block have no shot at being chosen 
	#and other negative blocks with a low chance at being chosen
	minWeight = min(weights)
	if minWeight <=0:
		minWeight = -minWeight
		weights = [x+minWeight+.1+offset for x in weights]
	
	for boxind, box in enumerate(gameBoard.box):
		if not box.splitPossible(gameBoard.splitAction):
			weights[boxind] = 0
	

		

	if weightverbose: 
		print("Final weights vector:") 
		print(weights)

	if timingInfo: 
		end = time.time()
		print("Time to calculate weights until fall for turn "+str(len(gameBoard.splitRecord))+": "+str(end-start))
	return weights