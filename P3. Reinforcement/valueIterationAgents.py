# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range (self.iterations):
            prevValues=self.values.copy()
            for state in self.mdp.getStates():
                self.values[state] = float('-inf')
                for action in  self.mdp.getPossibleActions(state):
                    value=0
                    for (newState, probs) in self.mdp.getTransitionStatesAndProbs(state,action):
                        value += probs * (self.mdp.getReward(state, action, newState) + self.discount*prevValues[newState])
                    self.values[state] = max(self.values[state], value)
                if self.values[state] == float('-inf'):
                    self.values[state] = 0.0

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        qValue = 0
        for (newState, probs) in self.mdp.getTransitionStatesAndProbs(state, action):
            qValue += probs * (self.mdp.getReward(state, action, newState) + self.discount*self.getValue(newState))
        return qValue


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.mdp.getPossibleActions(state)
        qValues = util.Counter()
        for a in actions:
            qValues[a] = self.getQValue(state, a)
        return qValues.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        for i in range(self.iterations):
            state = self.mdp.getStates()[i % len(self.mdp.getStates())]
            action = self.computeActionFromValues(state)
            if action is None:
                v = 0
            else:
                v = 0
                for (newState, probs) in self.mdp.getTransitionStatesAndProbs(state, action):
                    v += probs * (self.mdp.getReward(state, action, newState) + self.discount*self.getValue(newState))
                self.values[state] = v

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):

        states = self.mdp.getStates()
        predecessors = dict()
        for s in states:
            predecessors[s] = []
        for s in states:
            actions = self.mdp.getPossibleActions(s)
            for a in actions:
                for newState, probs in self.mdp.getTransitionStatesAndProbs(s, a):
                    if probs > 0:
                        predecessors[newState].append(s)

        priorityQueue = util.PriorityQueue()
        for s in states:
            qValues = util.Counter()
            for a in self.mdp.getPossibleActions(s):
                qValues[a] = self.computeQValueFromValues(s, a)
            diff = abs(self.values[s] - qValues[qValues.argMax()])
            priorityQueue.push(s, -diff)

        for i in range(self.iterations):
            if priorityQueue.isEmpty():
                return;
            s = priorityQueue.pop()
            qValues = util.Counter()
            for a in self.mdp.getPossibleActions(s):
                qValues[a] = self.computeQValueFromValues(s, a)
            self.values[s] = qValues[qValues.argMax()]

            for p in predecessors[s]:
                qValues = util.Counter()
                for a in self.mdp.getPossibleActions(p):
                    qValues[a] = self.computeQValueFromValues(p, a)
                diff = abs(self.values[p] - qValues[qValues.argMax()])
                if diff > self.theta:
                    priorityQueue.update(p, -diff)
