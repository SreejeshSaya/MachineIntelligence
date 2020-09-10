import sys

debug = int(sys.argv[1])
def get_neighbours(cost, v, n):
	neigh = []
	for i in range(1, n+1):
		if( cost[v][i] != -1 and i != v ):
			neigh.append([i,cost[v][i]])
	return neigh




def astar(cost, heuristic, start_point, goals):
	#f(n) = g(n) + h(n)
	fin = []
	num = len(cost) - 1 #in this case since cost is (n+1) n is the actual size
	frontier = set([start_point]) # nodes visited but neighbours not explored, initially =to start_point
	explored = set([]) # nodes visited and neighbours explored
	
	g = {} # g(n) -> current distance from start_node to n
	g[start_point] = 0
	#debug = 1
	parents = {}
	parents[start_point] = start_point

	temp_soln = []
	mincost = -1 #required to compare prev goal state's f(n) with new f(n')
	while(len(frontier) > 0): # A* and GBFS always do exhaustive search
		n = None

		# find a node with the lowest value of f()
		for v in frontier:
			if(n == None or g[v] + heuristic[v] < g[n] + heuristic[n]):
				n = v
		if(debug):
			print("before: FRONTIER -- ", frontier)
			print("before: EXPLORED --  ", explored);
			print("node : ", n)

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
			if(debug):
				print("mincost: {0}, g[n]+h[n]: {1}".format(mincost,g[n]+heuristic[n]))
			if(mincost==-1 or mincost > g[n]+heuristic[n]):
				if(debug):
					print("in")
				#print(g[n])
				mincost = g[n] + heuristic[n]
				temp_soln = []
				while parents[n] != n:
					temp_soln.append(n)
					n = parents[n]
				temp_soln.append(start_point)
				temp_soln.reverse()
			n = orig
			#print('Path found: {}'.format(temp_soln))
			#return temp_soln

		# n->parent m->child
		for (m, weight) in get_neighbours(cost, n, num ):
			#print(m,weight)
			if ( (m not in frontier) and (m not in explored)):
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

		if(debug):
			print("after: FRONTIER -- ", frontier)
			print("after: EXPLORED -- ", explored); print()
	return temp_soln
	return None

cost = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 5, 9, -1, 6, -1, -1, -1, -1, -1],
            [0, -1, 0, 3, -1, -1, 9, -1, -1, -1, -1], 
            [0, -1, 2, 0, 1, -1, -1, -1, -1, -1, -1],
            [0, 6, -1, -1, 0, -1, -1, 5, 7, -1, -1],
            [0, -1, -1, -1, 2, 0, -1, -1, -1, 2, -1],
            [0, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1],
            [0, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1],
            [0, -1, -1, -1, -1, 2, -1, -1, 0, -1, 8],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, 0, 7],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0]]
cost1 = [[0,0,0,0,0,0,0,0],
	[0,0,3,-1,-1,-1,-1,2],
	[0,-1,0,5,10,-1,-1,-1],
	[0,-1,-1,0,2,-1,1,-1],
	[0,-1,-1,-1,0,4,-1,-1],
	[0,-1,-1,-1,-1,0,-1,-1],
	[0,-1,-1,-1,-1,3,0,-1],
	[0,-1,-1,1,-1,-1,4,0]]
heuristic1 = [0, 5, 7, 3, 4, 6, 0, 0, 6, 5, 0]
heuristic2 = [0,7,9,4,2,0,3,5]
#check = get_neighbours(cost, 1, 10)
#for (m,weight)  in check:
#	print(m,weight)

print(astar(cost, heuristic1, 1, [8]))
print(astar(cost1, heuristic2, 1, [2]))
'''
verifies with unit 1 (video on PESU) : Problem solving by Searching- Informed Search , time - 19:37
for goal state 	6 - [1,2,6] g(n) = 14
			   	7 - [1,5,4,7] g(n) = 13
			   	10 - [1,5,9,10] g(n) = 15
'''
