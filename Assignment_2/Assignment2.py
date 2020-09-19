'''
Function tri_traversal - performs DFS, UCS and A* traversals and returns the path for each of these traversals 

n - Number of nodes in the graph
m - Number of goals ( Can be more than 1)
1<=m<=n
Cost - Cost matrix for the graph of size (n+1)x(n+1)
IMP : The 0th row and 0th column is not considered as the starting index is from 1 and not 0. 
Refer the sample test case to understand this better

Heuristic - Heuristic list for the graph of size 'n+1' 
IMP : Ignore 0th index as nodes start from index value of 1
Refer the sample test case to understand this better

start_point - single start node
goals - list of size 'm' containing 'm' goals to reach from start_point

Return : A list containing a list of all traversals [[],[],[]]
1<=m<=n
cost[n][n] , heuristic[n][n], start_point, goals[m]

NOTE : you are allowed to write other helper functions that you can call in the given function
'''

def tri_traversal(cost, heuristic, start_point, goals):
    l = []
    t1 = DFS_Traversal(cost, start_point, goals)
    t2 = UCS_Traversal(cost, start_point, goals)
    t3 = A_star_Traversal(cost, heuristic, start_point, goals)

    l.append(t1)
    l.append(t2)
    l.append(t3)
    return l

def DFS_Traversal(cost, start_point, goals):
    visited = set()
    nodeStack = []
    nodeStack.append((start_point, [start_point]))
    n = len(cost)
    while(len(nodeStack) != 0):
        (node, path) = nodeStack.pop()
        visited.add(node)
        if(node in goals):
            return path
        for i in range(n-1, 0, -1):
            if (cost[node][i] != -1) and (i not in visited):
                nodePath = path[:]
                nodePath.append(i)
                nodeStack.append((i, nodePath))
    return []

def UCS_Traversal(cost, start_point, goals):
    n = len(cost)
    explored = set()
    frontier = []
    frontier.append((0, start_point, [start_point]))
    while frontier:
        (pathCost, node, path) = frontier.pop(0)
        if node in goals:
            return path
        if(node not in explored):
            explored.add(node)
            for i in range(1, n):
                if(cost[node][i]!=-1 and i not in explored):
                    nodePath = path[:]
                    nodePath.append(i)
                    frontier.append((pathCost+cost[node][i], i, nodePath))
            frontier.sort()
    return []

def A_star_Traversal(cost, heuristic, start_point, goals):
    n = len(cost)
    explored = set()
    frontier = []
    frontier.append((0, heuristic[start_point], start_point, [start_point]))
    while frontier:
        (pathCost, pathHeur, node, path) = frontier.pop(0)
        if node in goals:
            return path
        # if(node not in explored):
        explored.add(node)
        for i in range(1, n):
            if(cost[node][i]!=-1 and i not in explored):
                nodePath = path[:]
                nodePath.append(i)
                nodePathCost = pathCost+cost[node][i]
                frontier.append((nodePathCost, nodePathCost+heuristic[i], i, nodePath))
        frontier.sort(key=lambda x: (x[1], x[2], x[3]))
        # print(frontier)
    return []