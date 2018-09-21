# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        foodList = newFood.asList()
        minDistance = float("inf") #list of distances to all food pellets
        for food in foodList:
            minDistance = min(minDistance, util.manhattanDistance(newPos, food))

        ghostDistances = 1
        ghostsClose = 0
#        print(successorGameState.getGhostPositions())
        for ghostPos in successorGameState.getGhostPositions():
            distanceToGhost = util.manhattanDistance(newPos, ghostPos)
            ghostDistances += distanceToGhost
            if distanceToGhost <= 1:
                ghostsClose += 1

        minDistanceNormalized = (1 / float(minDistance)) #Normalized so we have a value between 0-1
        ghostDistancesNormalized = (1 / float(ghostDistances)) #Normalized so we have a value between 0-1
        evaluation = successorGameState.getScore() + minDistanceNormalized + ghostDistancesNormalized - ghostsClose
        return evaluation

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def maxValue(gameState,depth):       #Pacman
            depth += 1
            v = float("-inf")
            actions = gameState.getLegalActions(0)
            if gameState.isWin() or gameState.isLose() or depth == self.depth:   # if the state is a terminal state: return the state’s utility
                return self.evaluationFunction(gameState)
            for action in actions:
                successor= gameState.generateSuccessor(0,action)
                v = max(v,minValue(1,successor,depth))
            return v

        def minValue(agent,gameState,depth):        #Ghosts
            v = float("inf")
            actions = gameState.getLegalActions(agent)
            if gameState.isWin() or gameState.isLose() or depth == self.depth:   # if the state is a terminal state: return the state’s utility
                return self.evaluationFunction(gameState)
            for action in actions:
                successor= gameState.generateSuccessor(agent,action)
                nextAgent = agent + 1
                if nextAgent == gameState.getNumAgents():   #Reset nextAgent so we know the next one to move is pacman
                    nextAgent = 0
                    v = min(v,maxValue(successor,depth))
                else:
                    v = min(v,minValue(nextAgent,successor,depth))
            return v

        action = Directions.STOP
        score = float("-inf")
        for agentState in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0,agentState)
            utility = minValue(1,nextState,0)
            if utility > score:
                action = agentState
                score = utility
        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def maxValue(gameState,depth,alpha,beta):       #Pacman
            depth += 1
            v = float("-inf")
            actions = gameState.getLegalActions(0)
            if gameState.isWin() or gameState.isLose() or depth == self.depth:   #Terminal State Test
                return self.evaluationFunction(gameState)
            for action in actions:
                successor= gameState.generateSuccessor(0,action)
                v = max(v,minValue(1,successor,depth,alpha,beta))
                if v > beta:
                    return v
                alpha = max(alpha,v)
            return v

        #For all ghosts.
        def minValue(agent,gameState,depth,alpha,beta):
            v = float("inf")
            actions = gameState.getLegalActions(agent)
            if gameState.isWin() or gameState.isLose() or depth == self.depth:   #Terminal State Test
                return self.evaluationFunction(gameState)
            for action in actions:
                successor= gameState.generateSuccessor(agent,action)
                nextAgent = agent + 1
                if nextAgent == gameState.getNumAgents():
                    nextAgent = 0
                    v = min(v,maxValue(successor,depth,alpha,beta))
                    if v < alpha:
                        return v
                    beta = min(beta,v)
                else:
                    v = min(v,minValue(nextAgent,successor,depth,alpha,beta))
                    if v < alpha:
                        return v
                    beta = min(beta,v)
            return v

        # Alpha-Beta Pruning
        action = Directions.STOP
        score = float("-inf")
        alpha = float("-inf")
        beta = float("inf")
        for agentState in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0,agentState)
            utility = minValue(1,nextState,0,alpha,beta)
            if utility > score:
                action = agentState
                score = utility
            if utility > beta:
                return returnAction
            alpha = max(alpha,utility)
        return action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(agent, depth, gameState): #Same as minimax nbut instead of min we do a division between the amount of actions available
            actions = gameState.getLegalActions(agent)
            numberOfAgents = gameState.getNumAgents()
            if gameState.isLose() or gameState.isWin() or depth == self.depth:  # return the utility in case the defined depth is reached or the game is won/lost.
                return self.evaluationFunction(gameState)
                #If the agent is pacman, then we max
            if agent == 0:
                cost = []
                for action in gameState.getLegalActions(agent):
                    cost.append(expectimax(1, depth, gameState.generateSuccessor(agent, action)))
                return max(cost)
                #If the  agent is a ghost, then we expectimax
            else:
                nextAgent = agent + 1  #  calculate the next agent to see if its a ghost or pacman and increase depth accordingly.
                if numberOfAgents == nextAgent:
                    nextAgent = 0 #Reset nextAgent so we know the next one to move is pacman
                if nextAgent == 0: #Increase depth of the game
                    depth += 1
                cost = []
                for action in gameState.getLegalActions(agent):
                    cost.append(expectimax(nextAgent, depth, gameState.generateSuccessor(agent, action)))
                return (sum(cost) / float(len(actions)))

        maxValue = -(float("inf"))
        action = Directions.STOP
        for agentState in gameState.getLegalActions(0):
            utility = expectimax(1, 0, gameState.generateSuccessor(0, agentState))
            if utility > maxValue:
                maxValue = utility
                action = agentState
        return action

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    Returns the utilty of choosing an action from the current game state.

    This evaluation function uses the distance to the closest pellet and normalizes
    it so that the smaller the distance, the better the score

    It also normalizes the total sum of ghost distances, normalizes it to have
    a similar weigth as the distance to the closest pellet, and substracts it
    to the score.

    The function also substracts the amount of power capsules left, and the amount
    of food pellets left.

    Finally, if a ghost is getting too close, the function also substracts a value
    to the score.

    """
    "*** YOUR CODE HERE ***"
    #First we calculate distances to food and ghosts and amount of power capsules on the board
    currentPos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    foodList = food.asList()
    minFoodDistance = float("inf")
    capsules = len(currentGameState.getCapsules())
    ghostsDistances = 1
    ghostsProximity = 0
    for food in foodList:
        minFoodDistance = min(minFoodDistance,util.manhattanDistance(currentPos, food))
    for ghosts in currentGameState.getGhostPositions():
        distance = util.manhattanDistance(currentPos, ghosts)
        ghostsDistances += distance
        if distance <= 1:
            ghostsProximity += 1

    #Values to be returned
    score = currentGameState.getScore()
    minFoodDistanceNormalized = (2 / float(minFoodDistance)) #Normalized so we have a value between 0-2
    ghostDistancesNormalized = (1 / float(ghostsDistances)) #Normalized so we have a value between 0-1
    betterEvaluation = score + minFoodDistanceNormalized - ghostDistancesNormalized - ghostsProximity - capsules - len(foodList)
    return  betterEvaluation

# Abbreviation
better = betterEvaluationFunction
