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

        print("New Position:", newPos)
        "*** YOUR CODE HERE ***"

        foodList = newFood.asList()
        print(foodList)
        distance = [float("inf")] #list of distances to all food pellets
        for food in foodList:
            distance.append(util.manhattanDistance(newPos, food))

        minDistance = min(distance) #Chooses the closest food pellet

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
        evaluation = successorGameState.getScore() + minDistanceNormalized + ghostDistancesNormalized - 2*ghostsClose
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
        def minimax(agent, depth, gameState):
            numberOfAgents = gameState.getNumAgents()
            if gameState.isLose() or gameState.isWin() or depth == self.depth:  # if the state is a terminal state: return the state’s utility
                return self.evaluationFunction(gameState)
            #If the agent is pacman, then we max
            if agent == 0:
                cost = []
                for action in gameState.getLegalActions(agent):
                    cost.append(minimax(1, depth, gameState.generateSuccessor(agent, action)))
                return max(cost)
            #If the  agent is a ghost, then we minimize
            else:
                nextAgent = agent + 1  # calculate the next agent to see if its a ghost or pacman and increase depth accordingly.
                if numberOfAgents == nextAgent:
                    nextAgent = 0 #Reset nextAgent so we know the next one to move is pacman
                if nextAgent == 0: #Increase depth of the game
                   depth += 1
                cost = []
                for action in gameState.getLegalActions(agent):
                    cost.append(minimax(nextAgent, depth, gameState.generateSuccessor(agent, action)))
                return min(cost)

        maxValue = -(float("inf"))
        action = Directions.STOP
        actions = gameState.getLegalActions()
        for agentState in actions:
            utility = minimax(1, 0, gameState.generateSuccessor(0, agentState))
            if utility > maxValue:
                maxValue = utility
                action = agentState
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
        def maxValue(agent, depth, gameState, alpha, beta):  # maxValue function
            v = -(float("inf"))
            legalActions =  gameState.getLegalActions(agent)
            for succesor in legalActions:
                v = max(v, abPruning(1, depth, gameState.generateSuccessor(agent, succesor), alpha, beta)) #(1 to call for the ghost)
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v

        def minValue(agent, depth, gameState, alpha, beta):  # minValue function
            v = float("inf")
            ##### Validate next agents (ghosts)######
            nextAgent = agent + 1
            numberOfAgents = gameState.getNumAgents()
            if numberOfAgents == nextAgent:
                nextAgent = 0
            if nextAgent == 0:
                depth += 1
            #########################################
            legalActions =  gameState.getLegalActions(agent)
            for succesor in legalActions:
                v = min(v, abPruning(nextAgent, depth, gameState.generateSuccessor(agent, succesor), alpha, beta))
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v

            def abPruning(agent, depth, gameState, alpha, beta):
                if gameState.isLose() or gameState.isWin() or depth == self.depth:  # if the state is a terminal state: return the state’s utility
                    return self.evaluationFunction(gameState)
                #If the agent is pacman, then we max
                if agent == 0:  # maximize for pacman
                    return maxValue(agent, depth, gameState, alpha, beta)
                #If the  agent is a ghost, then we minimize
                else:
                    return minValue(agent, depth, gameState, alpha, beta)

            utility = -(float("inf"))
            alpha = -(float("inf"))
            beta = float("inf")
            legalActions =  gameState.getLegalActions(0)
            action = Directions.EAST
            for agentState in legalActions:
                ghostValue = abPruning(1, 0, gameState.generateSuccessor(0, agentState), alpha, beta)
                if ghostValue > utility:
                    utility = ghostValue
                    action = agentState
                if utility > beta:
                    return utility
                alpha = max(alpha, utility)

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
                return max(expectimax(1, depth, gameState.generateSuccessor(agent, succesor)) for succesor in actions)
                #If the  agent is a ghost, then we expectimax
            else:
                nextAgent = agent + 1  #  calculate the next agent to see if its a ghost or pacman and increase depth accordingly.
                if numberOfAgents == nextAgent:
                    nextAgent = 0 #Reset nextAgent so we know the next one to move is pacman
                if nextAgent == 0: #Increase depth of the game
                    depth += 1
                return sum(expectimax(nextAgent, depth, gameState.generateSuccessor(agent, succesor)) for succesor in actions) / float(len(actions))

        maximum = -(float("inf"))
        action = Directions.STOP
        for agentState in gameState.getLegalActions(0):
            utility = expectimax(1, 0, gameState.generateSuccessor(0, agentState))
            if utility > maximum or maximum == float("-inf"):
                maximum = utility
                action = agentState

        return action

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
