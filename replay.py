import core_update as core
import sys
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtGui import QPainter, QColor, QPen,QBrush
from PyQt5.QtCore import *


##########################################
class gameBoardDisplay(QWidget):
##########################################
	
	def __init__(self,gameBoard):
		super().__init__()

		self.minimumTileSize=40
		self.initUI(gameBoard)
		self.gameBoard=gameBoard
		self.highLightedBox=0
		
	def initUI(self,gameBoard):      

		self.setGeometry(300, 300, self.minimumTileSize*gameBoard.width, self.minimumTileSize*gameBoard.height)
		self.setWindowTitle('Score: {0}\tSplits:{1}'.format(gameBoard.score,len(gameBoard.splitRecord)))
		self.show()
		
	def Update(self,gameBoard):
		self.gameBoard=gameBoard
		self.setWindowTitle('Score: {0}\tSplits:{1}'.format(gameBoard.score,len(gameBoard.splitRecord)))
		self.update()

	def HighlightBox(self,highlightedBox):
		self.highLightedBox=highlightedBox

	def paintEvent(self, e):

		qp = QPainter()
		qp.begin(self)

		self.drawBackground(qp)

		for boxindex,box in enumerate(self.gameBoard.box):
			self.drawBox(qp,box.x,box.y,box.width,box.height,boxindex,box.points)
			if boxindex==self.highLightedBox:
				self.drawBoxHighlight(qp,box.x,box.y,box.width,box.height)
		qp.end()
	
	def drawBackground(self,qp):

		brush = QBrush()
		brush.setStyle(Qt.SolidPattern)
		brush.setColor(QColor(230, 230, 230))
		qp.setBrush(brush)

		rect1=QRect(0, 0, self.minimumTileSize*self.gameBoard.width, self.minimumTileSize*self.gameBoard.height)
		qp.drawRect(rect1)  

	def drawBox(self,qp,x,y,width,height,index,points):
		
		brush = QBrush()
		brush.setStyle(Qt.SolidPattern)
		brush.setColor(QColor(150, 150, 150))
		qp.setBrush(brush)

		pen = QPen(Qt.black, 2, Qt.SolidLine)
		qp.setPen(pen)

		rect1=QRect(x*self.minimumTileSize, y*self.minimumTileSize, width*self.minimumTileSize, height*self.minimumTileSize)

		if points>0:
			brush.setStyle(Qt.Dense2Pattern)
			qp.setBrush(brush)
			qp.drawRect(rect1)             
			qp.drawText(rect1, Qt.AlignCenter,"[{0}]\n{1}".format(index,points))    
		else:
			qp.drawRect(rect1) 
			qp.drawText(rect1, Qt.AlignCenter,"[{0}]".format(index)) 
   
	def drawBoxHighlight(self,qp,x,y,width,height):

		brush = QBrush()
		brush.setStyle(Qt.NoBrush)
		qp.setBrush(brush)

		pen = QPen(Qt.red, 8, Qt.DashLine)
		qp.setPen(pen)

		rect1=QRect(x*self.minimumTileSize, y*self.minimumTileSize, width*self.minimumTileSize, height*self.minimumTileSize)
		qp.drawRect(rect1)


##########################################
def replaySequence(graphicalDisplay,sequence):
##########################################
	gameBoard=core.Board(8,16)
	
	while 1:

		moveOptions=gameBoard.getMoveOptions()

		if len(moveOptions)==0:
			return gameBoard.score,gameBoard.splitRecord
			break

		nextMove=sequence[len(gameBoard.splitRecord)]
		
		print("Move {0} of {1}: Split box #{2}".format(len(gameBoard.splitRecord),len(sequence),nextMove))
		
		graphicalDisplay.HighlightBox(nextMove)
		graphicalDisplay.Update(gameBoard)
		
		tempDelay=input("Press any key to continue")
		if nextMove not in moveOptions:
			print("------ IMPOSSIBLE MOVE REQUESTED -----")
			break

		if gameBoard.weights[nextMove] != max(gameBoard.weights):
			print("~~~Made move not recommended by algorithm.~~~")
			indices = [i for i, x in enumerate(gameBoard.weights) if x == max(gameBoard.weights)]
			print("Box(es) recommended:")
			print(indices)
			print("Weight of box(es) recommended:")
			print(max(gameBoard.weights))
			print("Weight of box chosen:")
			print(gameBoard.weights[nextMove])


		core.makeMove(gameBoard,nextMove)       

##########################################
if __name__ == '__main__':
##########################################
	try:
		sequenceFile=sys.argv[1]
	except:
		print("Didn't get the parameters I expected.\n\nExpected usage is replay.py <sequence filename>\n")
		sys.exit(1)

	print("\n--------Replay SPL-T--------\n")

	try:
		f = open(sequenceFile, "r")
	except:
		print("Couldn't open file",sequenceFile,", does it exist?")
		sys.exit(1)

	line=f.readline()
	f.close()
	sequence=(line.rstrip('\n').split(', '))
	sequence=[int(ii) for ii in sequence]

	print("This sequence has a length of",len(sequence))
	

	app = QApplication(sys.argv)

	gameBoard=core.Board(8,16)	

	graphicalDisplay = gameBoardDisplay(gameBoard)

	score,path=replaySequence(graphicalDisplay,sequence)

	print("\tScore: ",score,"\tLength: ",len(path))

	sys.exit(app.exec_())
