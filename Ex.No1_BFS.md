# Ex.No: 1  Implementation of Breadth First Search 
### DATE:                                                                            
### REGISTER NUMBER : 212222220022
### NAME : MANIKANDAN R
### AIM: 
To write a python program to implement Breadth first Search. 
### Algorithm:
1. Start the program
2. Create the graph by using adjacency list representation
3. Define a function bfs and take the set “visited” is empty and “queue” is empty
4. Search start with initial node and add the node to visited and queue.
5. For each neighbor node, check node is not in visited then add node to visited and queue list.
6.  Creating loop to print the visited node.
7.   Call the bfs function by passing arguments visited, graph and starting node.
8.   Stop the program.
### Program:

graph = {
    '5': ['3'],
    '3': ['2', '4'],
    '7': ['8'],
    '2': [],
    '4': ['8'],
    '8': []
}

visited = []  
queue = []   

def bfs(visited, graph, node):  
    visited.append(node)
    queue.append(node)

    while queue:
        m = queue.pop(0)
        print(m)

        for neighbour in graph[m]:
            if neighbour not in visited:
                visited.append(neighbour)
                queue.append(neighbour)


bfs(visited, graph, '5')


### Output:

<img width="827" height="247" alt="image" src="https://github.com/user-attachments/assets/e8f2fcde-5bea-4b53-a67e-b3ac2d3b3d2d" />


### Result:
Thus the breadth first search order was found sucessfully.
