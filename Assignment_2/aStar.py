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