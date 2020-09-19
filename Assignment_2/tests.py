from Assignment2 import *


def test(cal, exp, case ):
	print("\n")
	ok = {0: "DFS", 1: "UCS", 2: "A*S"}
	print("+++++++++++++++Test Case : ",case)
	for i in range(3):
		if(exp[i] == cal[i]):
			print("{0} : PASS".format(ok[i]))
		else:
			print("{0} : FAIL ---- Expected: {1}  Got: {2}".format(ok[i],exp[i],cal[i]))
			
cost1 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
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
        
cost2 = [[0,0,0,0,0,0,0,0],
	[0,0,3,-1,-1,-1,-1,2],
	[0,-1,0,5,10,-1,-1,-1],
	[0,-1,-1,0,2,-1,1,-1],
	[0,-1,-1,-1,0,4,-1,-1],
	[0,-1,-1,-1,-1,0,-1,-1],
	[0,-1,-1,-1,-1,3,0,-1],
	[0,-1,-1,1,-1,-1,4,0]] #https://www.geeksforgeeks.org/search-algorithms-in-ai/

cost3 = [
    [0, 0, 0, 0, 0, 0],
    [0, 0, 6, -1, -1, -1],
    [0, 6, 0, 3, 3, -1],
    [0, -1, 3, 0, 1, 7],
    [0, -1, 3, 1, 0, 8],
    [0, -1, -1, 7, 8, 0],
]

heuristic1 = [0, 5, 7, 3, 4, 6, 0, 0, 6, 5, 0]
heuristic2 = [0,7,9,4,2,0,3,5]
heuristic3 = [0, 10, 8, 7, 7, 3]


x = tri_traversal(cost1, heuristic1, 1, [1])
test(x,[[1],[1],[1]], 1)


x = tri_traversal(cost1, heuristic1, 1, [2])
test(x,[[1,2],[1,2],[1,2]], 2)


x = tri_traversal(cost1, heuristic1, 1, [3])
test(x,[[1,2,3],[1,2,3],[1,2,3]], 3)


x = tri_traversal(cost1, heuristic1, 1, [4])
test(x,[[1,2,3,4],[1,5,4],[1,5,4]], 4)


x = tri_traversal(cost1, heuristic1, 1, [5])
test(x,[[1,2,3,4,8,5],[1,5],[1,5]], 5)


x = tri_traversal(cost1, heuristic1, 1, [6])
test(x,[[1,2,6],[1,2,6],[1,2,6]], 6)


x = tri_traversal(cost1, heuristic1, 1, [7])
test(x,[[1,2,3,4,7],[1,5,4,7],[1,5,4,7]], 7)


x = tri_traversal(cost1, heuristic1, 1, [8])
test(x,[[1,2,3,4,8],[1,5,4,8],[1,5,4,8]], 8)


x = tri_traversal(cost1, heuristic1, 1, [9])
test(x,[[1,2,3,4,8,5,9],[1,5,9],[1,5,9]], 9)


x = tri_traversal(cost1, heuristic1, 1, [10])
test(x,[[1,2,3,4,8,5,9,10],[1,5,9,10],[1,5,9,10]], 10)


x = tri_traversal(cost1, heuristic1, 1, [6,7,10])
test(x,[[1,2,3,4,7],[1,5,4,7],[1,5,4,7]], 11)


x = tri_traversal(cost1, heuristic1, 1, [3,4,7,10])
test(x,[[1,2,3],[1,2,3],[1,2,3]], 12)


x = tri_traversal(cost1, heuristic1, 1, [5,9,4])
test(x,[[1,2,3,4],[1,5],[1,5]], 13)


x = tri_traversal(cost1, heuristic1, 1, [4,8,10])
test(x,[[1,2,3,4],[1,5,4],[1,5,4]], 14)


x = tri_traversal(cost1, heuristic1, 1, [2,8,5])
test(x,[[1,2],[1,2],[1,2]], 15)


x = tri_traversal(cost1, heuristic1, 1, [7,9,10])
test(x,[[1,2,3,4,7],[1,5,9],[1,5,4,7]], 16) # a* != dfs here


x = tri_traversal(cost1, heuristic1, 1, [10,6,8,4])
test(x,[[1,2,3,4],[1,5,4],[1,5,4]], 17)


x = tri_traversal(cost1, heuristic1, 1, [9,7,5,10])
test(x,[[1,2,3,4,7],[1,5],[1,5]], 18)


x = tri_traversal(cost2, heuristic2, 1, [1])
test(x,[[1],[1],[1]], 19)


x = tri_traversal(cost2, heuristic2, 1, [2])
test(x,[[1,2],[1,2],[1,2]], 20)


x = tri_traversal(cost2, heuristic2, 1, [3])
test(x,[[1,2,3],[1,7,3],[1,7,3]], 21)


x = tri_traversal(cost2, heuristic2, 1, [4])
test(x,[[1,2,3,4],[1,7,3,4],[1,7,3,4]], 22)


x = tri_traversal(cost2, heuristic2, 1, [5])
test(x,[[1,2,3,4,5],[1,7,3,6,5],[1,7,3,6,5]], 23)


x = tri_traversal(cost2, heuristic2, 1, [6])
test(x,[[1,2,3,6],[1,7,3,6],[1,7,3,6]], 24)


x = tri_traversal(cost2, heuristic2, 1, [7])
test(x,[[1,7],[1,7],[1,7]], 25)


x = tri_traversal(cost2, heuristic2, 1, [4,5,6])
test(x,[[1,2,3,4],[1,7,3,6],[1,7,3,4]], 26)# i donno a* can be = [1,7,3,6] cuz it costs the same f(n)


x = tri_traversal(cost2, heuristic2, 1, [3,6,7])
test(x,[[1,2,3],[1,7],[1,7]], 27)


x = tri_traversal(cost2, heuristic2, 1, [4,6])
test(x,[[1,2,3,4],[1,7,3,6],[1,7,3,4]], 28) # i donno a* can be = [1,7,3,6] cuz it costs the same f(n)


x = tri_traversal(cost2, heuristic2, 1, [2,3,7])
test(x,[[1,2],[1,7],[1,7]], 29)


x = tri_traversal(cost2, heuristic2, 4, [3])
test(x,[[],[],[]], 30)


x = tri_traversal(cost3, heuristic3, 1, [5])
test(x,[[1,2,3,4,5],[1,2,3,5],[1,2,3,5]], 31)

