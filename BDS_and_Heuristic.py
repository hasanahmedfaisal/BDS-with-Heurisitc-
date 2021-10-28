#BDS has been implemented with heuristic
import queue
from queue import PriorityQueue
n = 15
graph =  [[]for i in range(n)] #list of n variables . It is a graph of n spaces

def BFSf(source, target, n):
    visitedf = []
    visitedb = []
    pqf = PriorityQueue()
    pqf.put((0, source))
    pqb = PriorityQueue()
    pqb.put((14, target))
    while not pqf.empty() and not pqb.empty():
        s = pqf.get()[1] #getting the path with the least cost
        print(s, end = ' ')
        if s == target:
            break

        for v, c in graph[s]:
            if v not in visitedf:
                visitedf.append(v)
                pqf.put((c,v))

        #print()
    return s
    #return pqf
        #sb = pqb.get()[0]

def BFSb(source, target, n):
    visitedf = []
    visitedb = []
    pqb = PriorityQueue()
    pqb.put((0, target))
    #pqb = PriorityQueue()
    #pqb.put((14, target))
    while not pqb.empty():
        s = pqb.get()[1] #getting the path with the least cost
        print(s, end = ' ')
        if s == source:
            break

        for v, c in graph[s]:
            if v not in visitedf:
                visitedf.append(v)
                pqb.put((c,v))

    return s
    #return pqb
def addedge(x, y, cost):
	graph[x].append((y, cost))
	graph[y].append((x, cost))

def intersect():
    lf = []
    lb = []
    while not pqf.empty():
        lf = (pqf.get()[1])
        #lb.append(pqb.get())
    print("funit " + str(lf))
    #print(lb)
    #q = set(lf).intersection(set(lb))
    #return q



addedge(0, 1, 3)
addedge(0, 2, 2)
addedge(0, 3, 5)
addedge(1, 4, 9)
addedge(1, 5, 8)
addedge(2, 6, 4)
addedge(2, 7, 14)
addedge(3, 8, 7)
addedge(8, 9, 5)
addedge(8, 10, 6)
addedge(9, 11, 1)
addedge(9, 12, 10)
addedge(9, 13, 1)

source = 0
target = 9
print("Forward")
pqf = BFSf(source, target, n)
print("Backward")
pqb = BFSb(source, target, n)
#print('Ha')
#print(pqf)
#print(intersect())