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

from game import Directions
import util


class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0, item)

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


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.
    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.
    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    new_found = []
    expanded = []
    # start state here
    startstate = problem.getStartState()
    startnode = (startstate, [])
    new_found.append(startnode)
    while len(new_found) >= 0:
        currentstate, actions = new_found[len(new_found) - 1]
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

def breadthFirstSearch(problem):
    # Update, not to take the found/frontier list as a set since it ain't hashable and created problems in Q5
    q = util.Queue()

    # visitedPos holds all of the visited positions already (this is required for
    # the graph-search implementation of BFS)
    visited = []

    # push starting state onto the stack with an empty path
    q.push((problem.getStartState(), [])) #second space is for the path with the current node

    # Then we can start looping, note our loop condition is if the stack is empty
    # if the stack is empty at any point we failed to find a solution
    while (not q.isEmpty()):

        # since our stack elements contain two elements
        # we have to fetch them both like this
        currentPos, currentPath = q.pop()
        # print("Currently Visiting:", currentPos, "\nPath=", end="");
        # print(currentPath);
        # then we append the currentPos to the list of visited positions
        visited.append(currentPos)

        # check if current state is a goal state, if it is, return the path
        if (problem.isGoalState(currentPos)):
            return currentPath;

        # obtain the list of successors from our currentPos
        successors = problem.getSuccessors(currentPos)

        # if we have successors, note that these successors have a position and the path to get there
        if (len(successors) != 0):
            # iterate through them
            for state in successors:
                # if we find one that has not already been visisted
                if ((state[0] not in visited) and (state[0] not in (stateQ[0] for stateQ in q.list))):
                    # calculate the new path (currentPath + path to reach new state's position)
                    newPath = currentPath + [state[1]]
                    # push it onto the stack with the new path
                    q.push((state[0], newPath))

    # util.raiseNotDefined()

def uniformCostSearch(problem):
    q = util.PriorityQueue()

    visitedPos = []

    q.push((problem.getStartState(), []), 0)

    while (not q.isEmpty()):

        currentPos, currentPath = q.pop()

        visitedPos.append(currentPos)

        if (problem.isGoalState(currentPos)):
            return currentPath;

        successors = problem.getSuccessors(currentPos)

        if (len(successors) != 0):
            for state in successors:
                if (state[0] not in visitedPos) and (state[0] not in (stateQ[2][0] for stateQ in q.heap)):
                    newPath = currentPath + [state[1]]
                    q.push((state[0], newPath), problem.getCostOfActions(newPath))

                elif (state[0] not in visitedPos) and (state[0] in (stateQ[2][0] for stateQ in q.heap)):
                    for stateQ in q.heap:
                        if stateQ[2][0] == state[0]:
                            oldPriority = problem.getCostOfActions(stateQ[2][1])

                    newPriority = problem.getCostOfActions(currentPath + [state[1]])

                    # State is cheaper with his hew father -> update and fix parent #
                    if oldPriority > newPriority:
                        newPath = currentPath + [state[1]]
                        q.update((state[0], newPath), newPriority)

                        # util.raiseNotDefined()

from util import PriorityQueue
class PriorityQ_and_Function(PriorityQueue):
    """
    Implements a priority queue with the same push/pop signature of the
    Queue and the Stack classes. This is designed for drop-in replacement for
    those two classes. The caller has to provide a priority function, which
    extracts each item's priority.
    """

    def __init__(self, problem, priorityFunc):
        "priorityFunction (item) -> priority"
        self.priorityFunc = priorityFunc  # store the priority function
        PriorityQueue.__init__(self)  # super-class initializer
        self.problem = problem

    def push(self, item, heuristic):
        "Adds an item to the queue with priority from the priority function"
        PriorityQueue.push(self, item, self.priorityFunc(self.problem, item, heuristic))


def nullHeuristic(state, problem=None, goal = None): #modified the null heuristic to make it work with the newer versions of the BDS with heuristics
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

# Calculate f(n) = g(n) + h(n) #
def f(problem, state, heuristic):
    return problem.getCostOfActions(state[1]) + heuristic(state[0], problem)


def aStarSearch(problem, heuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    queueXY = PriorityQ_and_Function(problem, f)

    path = []  # Every state keeps it's path from the starting state
    visited = []  # Visited states

    # Check if initial state is goal state #
    if problem.isGoalState(problem.getStartState()):
        return []

    # Add initial state. Path is an empty list #
    element = (problem.getStartState(), [])

    queueXY.push(element, heuristic)

    while (True):

        # Terminate condition: can't find solution #
        if queueXY.isEmpty():
            return []

        # Get informations of current state #
        xy, path = queueXY.pop()  # Take position and path

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

                    newPath = path + [item[1]]  # Fix new path
                    element = (item[0], newPath)
                    queueXY.push(element, heuristic)


directions = {'North': 'South', 'East': 'West', 'South': 'North', 'West': 'East'}

def BDSMM0(problem):
    """
    Bi directional search - MM0.
    Two simple BFS in both direction.
    """
    def __reversedPath(p):
       # this function reverses a list and returns it #
        return [Directions.REVERSE[x] for x in p][::-1] # the -1 signifies that we are starting from the back

    from util import Queue
    # Init two ques and visited sets.
    q1 = Queue() #making 2 queues since BDS is just BFS in 2 directions and BFS is implemented via queues (First come First Serve)
                   #the node found first is expanded first
    q2 = Queue()
    visited1 = dict() #making dictionaries so that we have a key (node) and a value assigned to that key (the path from that node towards the goal)
    visited2 = dict() #similarly the second one for the this purpose

    q1.push((problem.getStartState(), [], 0)) #this queue is for the forward diretion search and hence this starts from th stat state
    q2.push((problem.goal, [], 0)) # this is for the backward seach and hence this starts from the gol state , we try to make them intersect
    visited1[problem.getStartState()] = '' #empty at first
    visited2[problem.goal] = '' #empty at first
    expanding_level1 = 0
    expanding_level2 = 0

    # Two simple BFS in each while no of the queues are empty
    while True:
        # First BFS, from start to goal
        # Will expand one level of nodes
        if q1.isEmpty():
            return [] #stop
        #############################################
        #print(q1.list[0][0]) #node number and distance from the goal
        #print(q1.list[0][1]) #directions taken also show the path
        #print(q1.list[0][2]) #level
        #print("hello")
        #############################################
        while (not q1.isEmpty()) and q1.list[0][2] == expanding_level1: #node wise taking steps
            # Get current state
            cur_state, path, level = q1.pop()

            # Check result
            if problem.isGoalStateBi(cur_state, visited2):# from the searh agents file checks whether the search is going back to the already
                return path + __reversedPath(visited2[cur_state])# visited nodes

            # Expand valid neighbors
            valid_neighbor = filter(lambda x: x[0] not in visited1, problem.getSuccessors(cur_state))# just gives us the neighbors and they should also
            for nxt in valid_neighbor: # be unvisited
                q1.push((nxt[0], path+[nxt[1]], level+1))
                visited1[nxt[0]] = path+[nxt[1]]
        expanding_level1 += 1 #number of levels that have been expanded

        ''''get the level number from here
        ###########################################################
        #print("Level here {}".format(expanding_level1)) 
        ###############################################################
        '''''
        # Second BFS, from foal to start
        # Will expand one level of nodes
        if q2.isEmpty():
            return []

        #############################################
        #print(q2.list[0][0]) #node number and distance from the goal
        #print(q2.list[0][1]) #directions taken also show the path
        #print(q2.list[0][2]) #level
        #print("hello")
        #############################################

        while (not q2.isEmpty()) and q2.list[0][2] == expanding_level2: #starting to expand levels from the bottom now
            # Get current state
            cur_state, path, level = q2.pop()

            # Check result
            if problem.isGoalStateBi(cur_state, visited1):
                return __reversedPath(visited1[cur_state]) + path

            # Expand valid neighbors
            valid_neighbor = filter(lambda i: i[0] not in visited2, problem.getSuccessors(cur_state))#getting the valid neighbors
            for nxt in valid_neighbor:
                q2.push((nxt[0], path+[nxt[1]], level+1))
                visited2[nxt[0]] = path+[nxt[1]]
        expanding_level2 += 1
    return []


def BDSMM(problem, heuristic): #now this part is going to have the a* search in 2 directions

    def reversedPath(p):

        return [Directions.REVERSE[x] for x in p][::-1] #again, we want the reversed path list function for the

    from util import PriorityQueue
    # Init two priority queues and visited dicts
    pq1 = PriorityQueue()
    pq2 = PriorityQueue()
    visited1 = dict()
    visited2 = dict()

    pq1.push((problem.getStartState(), [], 0), heuristic(problem.getStartState(), problem, 'goal')) #this heuristic is the foodHeuristic
    #we defined earlier
    pq2.push((problem.goal, [], 0), heuristic(problem.goal, problem, 'start'))
    visited1[problem.getStartState()] = [] #empty
    visited2[problem.goal] = [] #empty

    # Run two a* search in each while round
    while True:
        # First a* search, from start to goal
        # Only expand one node
        if pq1.isEmpty():
            return []
        cur_state, path, level = pq1.pop()

        if problem.isGoalStateBi(cur_state, visited2):
            return path + reversedPath(visited2[cur_state])

        valid_neighbor = filter(lambda x: x[0] not in visited1, problem.getSuccessors(cur_state)) # getting the valid neighbors here
        for nxt in valid_neighbor: # the 'filter' keyword helps us get only those values that are satisfying the condition
            np = heuristic(nxt[0], problem, 'goal') + problem.getCostOfActions(path+[nxt[1]]) #choose path and nodes on the basis of the heuristic
            pq1.push((nxt[0], path+[nxt[1]], level+1), np)
            visited1[nxt[0]] = path+[nxt[1]]

        # Second a* search, from goal to start
        # Only expand one node
        # this is the same search again but only from the other side, that is, from the goal state to the start state
        if pq2.isEmpty():
            return []

        cur_state, path, level = pq2.pop()

        if problem.isGoalStateBi(cur_state, visited1):
            return reversedPath(visited1[cur_state]) + path

        valid_neighbor = filter(lambda x: x[0] not in visited2, problem.getSuccessors(cur_state))
        for new in valid_neighbor:
            np = heuristic(new[0], problem, 'start') + problem.getCostOfActions(path+[new[1]])
            pq2.push((new[0], path+[new[1]], level+1), np)
            visited2[new[0]] = path+[new[1]]

    return []

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
bd0 = BDSMM0
bd = BDSMM
