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

# Helper Function for A_star_Traversal
def get_neighbours(cost, v, n):
	neigh = []
	for i in range(1, n+1):
		if( cost[v][i] != -1 and i != v ):
			neigh.append([i,cost[v][i]])
	return neigh

def A_star_Traversal(cost, heuristic, start_point, goals):
	num = len(cost) - 1 #in this case since cost is (n+1) n is the actual size
	frontier = set([start_point]) # nodes visited but neighbours not explored, initially =to start_point
	explored = set([]) # nodes visited and neighbours explored
	
	g = {} # g(n) -> current distance from start_node to n
	g[start_point] = 0
	parents = {}
	parents[start_point] = start_point

	l = []
	mincost = -1 #required to compare prev goal state's f(n) with new f(n')
	while(len(frontier) > 0): # A* and GBFS always do exhaustive search
		n = None
		# find a node with the lowest value of f()
		for v in frontier:
			if(n == None or g[v] + heuristic[v] < g[n] + heuristic[n]):
				n = v

		if (n == None):
			#print("No path")
			return None
		'''
		if the current node is the goal
        then trace back from child to parent and so on until start_point
        thus this will be in reverse order. 
		'''
		if(n in goals):
			orig = n
			if(mincost==-1 or mincost > g[n]+heuristic[n]):
				#print(g[n])
				mincost = g[n] + heuristic[n]
				l = []
				while parents[n] != n:
					l.append(n)
					n = parents[n]
				l.append(start_point)
				l.reverse()

			n = orig
			#print('Path found: {}'.format(l))
			#return l

		# n->parent m->child
		for (m, weight) in get_neighbours(cost, n, num ):
			#print(m,weight)
			if ((m not in frontier) and (m not in explored)):
				frontier.add(m)
				parents[m] = n
				g[m] = g[n] + weight
			else:
				if( g[m] > g[n] + weight):
					g[m] = g[n] + weight
					parents[m] = n

					if(m in explored):
						explored.remove(m)
						frontier.add(m)
		#print(parents)
		#now n has been explored, including its neighbours
		frontier.remove(n)
		explored.add(n)

	return l