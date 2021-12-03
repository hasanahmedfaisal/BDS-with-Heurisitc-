"""Search the shallowest nodes in the search tree first."""
"*** YOUR CODE HERE ***"
new_found = Queue()  # custom queue class on top used to implement BFS
new_found.enqueue((problem.getStartState(), [], 0))
found = set([])

while not new_found.isEmpty():
    node, path, cost = new_found.dequeue()
    if problem.isGoalState(node):
        return path
    if node in found:
        continue
    found.add(node)
    for n, p, c in problem.getSuccessors(node):
        if n not in new_found.items and n not in found:
            new_found.enqueue((n, path + [p], c))
return False