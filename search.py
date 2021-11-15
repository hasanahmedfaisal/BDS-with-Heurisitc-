# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)
    
class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    new_found = []
    expanded = []
    #start state here
    startstate = problem.getStartState()
    startnode = (startstate, [])
    new_found.append(startnode)
    while len(new_found)>=0:
        currentstate, actions = new_found[len(new_found)-1]
        new_found.pop(-1)

        if currentstate not in expanded:
            expanded.append(currentstate)

            if problem.isGoalState(currentstate):
                return actions
            else:
                successors = problem.getSuccessors(currentstate)

                for succState, succAction, succCost in successors:
                    newaction = actions + [succAction]
                    newnode = (succState, newaction)
                    new_found.append(newnode)
    #util.raiseNotDefined()

def breadthFirstSearch(problem):
    q = util.Queue()

    #visitedPos holds all of the visited positions already (this is required for
    #the graph-search implementation of BFS)
    visitedPos = []

    #push starting state onto the stack with an empty path
    q.push((problem.getStartState(),[]))

    #Then we can start looping, note our loop condition is if the stack is empty
    #if the stack is empty at any point we failed to find a solution
    while(not q.isEmpty()):

        #since our stack elements contain two elements
        #we have to fetch them both like this
        currentPos,currentPath = q.pop()
        #print("Currently Visiting:", currentPos, "\nPath=", end="");
        #print(currentPath);
        #then we append the currentPos to the list of visited positions
        visitedPos.append(currentPos)

        #check if current state is a goal state, if it is, return the path
        if (problem.isGoalState(currentPos)):
            return currentPath;

        #obtain the list of successors from our currentPos
        successors = problem.getSuccessors(currentPos)

        #if we have successors, note that these successors have a position and the path to get there
        if (len(successors) != 0):
            #iterate through them
            for state in successors:
                #if we find one that has not already been visisted
                if ((state[0] not in visitedPos) and (state[0] not in (stateQ[0] for stateQ in q.list))):
                    #calculate the new path (currentPath + path to reach new state's position)
                    newPath = currentPath + [state[1]]
                    #push it onto the stack with the new path
                    q.push((state[0],newPath))
                    
    #util.raiseNotDefined()
    #added to increase effectiveness
def uniformCostSearch(problem):
    q = util.PriorityQueue()

    visitedPos = []

    q.push((problem.getStartState(),[]), 0)

    while(not q.isEmpty()):


        currentPos,currentPath = q.pop()

        visitedPos.append(currentPos)

        if (problem.isGoalState(currentPos)):
            return currentPath;

        successors = problem.getSuccessors(currentPos)

        if (len(successors) != 0):
            for state in successors:
                if (state[0] not in visitedPos) and (state[0] not in (stateQ[2][0] for stateQ in q.heap)):
                    newPath = currentPath + [state[1]]
                    q.push((state[0],newPath),problem.getCostOfActions(newPath))

                elif (state[0] not in visitedPos) and (state[0] in (stateQ[2][0] for stateQ in q.heap)):
                    for stateQ in q.heap:
                        if stateQ[2][0] == state[0]:
                            oldPriority = problem.getCostOfActions(stateQ[2][1])

                    newPriority = problem.getCostOfActions(currentPath + [state[1]])

                    # State is cheaper with his hew father -> update and fix parent #
                    if oldPriority > newPriority:
                        newPath = currentPath + [state[1]]
                        q.update((state[0],newPath),newPriority)  
    
    #util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0
from util import PriorityQueue
class PriorityQ_and_Function(PriorityQueue):
    """
    Implements a priority queue with the same push/pop signature of the
    Queue and the Stack classes. This is designed for drop-in replacement for
    those two classes. The caller has to provide a priority function, which
    extracts each item's priority.
    """
    def  __init__(self, problem, priorityFunc):
        "priorityFunction (item) -> priority"
        self.priorityFunc = priorityFunc      # store the priority function
        PriorityQueue.__init__(self)        # super-class initializer
        self.problem = problem
    def push(self, item, heuristic):
        "Adds an item to the queue with priority from the priority function"
        PriorityQueue.push(self, item, self.priorityFunc(self.problem,item,heuristic))

# Calculate f(n) = g(n) + h(n) #
def f(problem,state,heuristic):

    return problem.getCostOfActions(state[1]) + heuristic(state[0],problem)
def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    queueXY = PriorityQ_and_Function(problem,f)

    path = [] # Every state keeps it's path from the starting state
    visited = [] # Visited states


    # Check if initial state is goal state #
    if problem.isGoalState(problem.getStartState()):
        return []

    # Add initial state. Path is an empty list #
    element = (problem.getStartState(),[])

    queueXY.push(element,heuristic)

    while(True):

        # Terminate condition: can't find solution #
        if queueXY.isEmpty():
            return []

        # Get informations of current state #
        xy,path = queueXY.pop() # Take position and path

        # State is already been visited. A path with lower cost has previously
        # been found. Overpass this state
        if xy in visited:
            continue

        visited.append(xy)

        # Terminate condition: reach goal #
        if problem.isGoalState(xy):
            return path

        # Get successors of current state #
        succ = problem.getSuccessors(xy)

        # Add new states in queue and fix their path #
        if succ:
            for item in succ:
                if item[0] not in visited:

                    # Like previous algorithms: we should check in this point if successor
                    # is a goal state so as to follow lectures code

                    newPath = path + [item[1]] # Fix new path
                    element = (item[0],newPath)
                    queueXY.push(element,heuristic)
                    
direction = {'North': 'South', 'East': 'West', 'South': 'North', 'West': 'East'}

#here we implement the Bidirectional Search as per the paper we were supposed to refer
def BDSMM0(problem): #here MM0 is just your normal BFS in the forward direction
    #that is, it is from the root node to the goal state(if it exists)
    #here we need 2 queues and 2 visited dictionaries
    q1 = util.Queue()#since BFS, we initialize as queues since queues follow
    q2 = util.Queue()#FIFO like BFS
    #now we intialize the dictionaries, where each key is the node we visited
    #and the values corresponding to the key is the path we followed
    visited1 = {}
    visited2 = {}

    #here we fill in the initial states:
    #q1 starts from the root (fwd BFS)
    #q2 starts from the goal (bkd BFS)

    q1.push(problem.getStartState())
    q2.push(problem.goal)

    #marking the nodes as visited
    visited1[problem.getStartState()] = '' #no path exists as of now
    visited2[problem.goal] = '' #obviously, this being the first node, no path
    #exists as of now, hence the values to these keys are empty
    #but these have been recorded into the 'visited list'
    #while loop till the completion of the search or till when no path exists
    while not q1.isEmpty() and not q2.isEmpty():
        while not q1.isEmpty():
            #pop the current node, since it has been recorded as explored
            #and hence we get it off the queue as we do on normal BFS
            s = q1.pop()
            #checking for a goal state
            if problem.isGoalState(s, visited2):
                rev = [direction[n] for n in visited2[s]]
                #reverse the path taken by the other search to meet in the middle and append
                return visited1[s] + rev[::-1]
            #if not a goal state, expand further
            for state in problem.getSuccessors(s): #order is being managed
                if state[0] in visited1: #if state has been visited before, dont visit again
                    continue
                q1.push(state[0])
                visited1[state[0]] = list(visited1[s]) + [state[1]] #appending the next state in the 'visited list' we had
    
        while not q2.isEmpty():# this is the same, but searching in reverse
            s2 = q2.pop()
            if problem.isGoalState(s2, visited1):
                return [direction[n] for n in visited1[s2]][::-1] + visited2[s2] #reversing the order we got
            for state in problem.getSuccessors(s2):
                if state[0] in visited2: # Again leaving the nodes we already visited
                    continue

                q2.push(state[0])
                visited2[state[0]] = list(visited2[s2]) + [state[1]]

def BDSMM(problem, heuristic): #this is just a* in both the directions


    q1 = util.PriorityQueue()
    q2 = util.PriorityQueue()

    # Declare dictionaries to store visited positions: 1 stores for forward traversal and 2 stands for backward traversal
    visited1 = {}
    visited2 = {}

    # Add both starting states to visited Dicts
    visited1[problem.getStartState()] = [] #we dont't have any corresponding values for these keys just added
    visited2[problem.goal] = []

    # We use a priority que to store nodes in the frontier with the A* cost metric of f(n)=h(n)+g(n)
    # The priority queue helps us to maintain the order - from the higher priority to lower priority
    # problem.getCostOfActions() gives us the g(n)
    # while heuristic(state, problem) gives us the  f(n)
    q1.push((problem.getStartState()), (problem.getCostOfActions({}) + heuristic(problem.getStartState(), problem, "g")))
    q2.push((problem.goal), (problem.getCostOfActions({}) + heuristic(problem.goal, problem, "s")))

    # Run while both frontier's are not empty and return [] in the case the goal is not reachable from the start
    while not q1.isEmpty() and not q2.isEmpty():

        # Run both searches at simultaneously
        s = q1.pop()

        if problem.isGoalState(s, visited2):
            rev = [direction[n] for n in visited2[s]]
            return visited1[s] + rev[::-1] # the reversed path that the search took to get to the middle to meet

        successors = problem.getSuccessors(s)

        for state in successors:  # priority queue manages order for us so we don't have to use if statements
            if state[0] in visited1:
                continue

            visited1[state[0]] = list(visited1[s]) + [state[1]]
            q1.push(state[0], (problem.getCostOfActions(visited1[state[0]]) + heuristic(state[0], problem, "g")))

        s2 = q2.pop()

        if problem.isGoalState(s2, visited1):
            return visited1[s2] + [direction[d] for d in visited2[s2]][::-1]

        successors = problem.getSuccessors(s2)

        for state in successors:  # order is being managed again
            if state[0] in visited2:
                continue

            visited2[state[0]] = list(visited2[s2]) + [state[1]]
            q2.push(state[0],
                    (problem.getCostOfActions(visited2[state[0]]) + heuristic(state[0], problem, "s")))

    
# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
BD0 = BDSMM0
BD = BDSMM

