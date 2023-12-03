#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <unordered_map>
#include <set>
#include <string>
#include <windows.h>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <climits>

using namespace std;

void clearScreen()
{
    system("CLS");
}

void displayMenu()
{
    cout << "---User Menu---" << endl << endl;
    cout << "1-787. Cheapest Flights Within K Stops" << endl;
    cout << "2-1489. Find Critical and Pseudo-Critical Edges in Minimum Spanning Tree" << endl;
    cout << "3-1697. Checking Existence of Edge Length Limited Paths" << endl;
    cout << "4-Cablaj" << endl;
    cout << "5-D. Jzzhu and Cities" << endl;
    cout << "6-Rusuoaica" << endl;
    cout << "7-Camionas" << endl;
    cout << "8-Bile" << endl;
    cout << "9-Oracol" << endl;
    cout << "10-Apm2" << endl;
    cout << "11-Exit" << endl;
}

class Graph
{
private:

    int n; /// The number of nodes of the graph.
    vector<vector<int>> adjList; /// This vector stores the adjacency list representation of the graph.

    /// Private function to perform the Bellman-Ford algorithm with a limited number of relaxations.
    /// \param prices Reference to a vector storing the minimum prices to each node.
    /// \param flights A vector of vectors representing the flights with [from, to, price].
    /// \param k An integer representing the maximum number of relaxations (stops) allowed.
    void bellmanFord(vector<int>& prices, const vector<vector<int>>& flights, int k);

    /// Private function to initialize a Union-Find data structure with 'n' nodes.
    /// \param n An integer representing the number of nodes in the Union-Find set.
    /// \param par A reference to a vector storing the parent nodes in the disjoint set.
    /// \param rank A reference to a vector storing the rank of each node in the disjoint set.
    /// \param maxRank To initialize current maximum rank
    void unionFindConstructor(int n, vector<int>& par, vector<int>& rank, int& maxRank);

    /// Private function to find the representative (parent) of a node in a disjoint set.
    /// Additionally, performs path compression to optimize future find operations.
    /// \param node An integer representing the node for which to find the representative.
    /// \param par A reference to a vector storing the parent nodes in the disjoint set.
    /// \return An integer representing the representative (parent) of the given node.
    int find(int node, vector<int>& par);

    /// Private function to perform the union operation in a disjoint set.
    /// \param node1 An integer representing the first node.
    /// \param node2 An integer representing the second node.
    /// \param par A reference to a vector storing the parent nodes in the disjoint set.
    /// \param rank A reference to a vector storing the ranks of nodes in the disjoint set.
    /// \param maxRank To store the current maximum rank
    /// \return A boolean indicating whether the union operation was successful (true) or not (false)
    bool union1(int node1, int node2, vector<int>& par, vector<int>& rank, int& maxRank);

    /// Private function to perform the Kruskal's algorithm and find critical and pseudo-critical edges.
    /// \param n An integer representing the number of nodes.
    /// \param edges A reference to a vector of vectors representing the edges with [from, to, weight].
    /// \param critical A reference to a vector to store critical edges.
    /// \param pseudo A reference to a vector to store pseudo-critical edges.
    /// \return An integer representing the weight of the minimum spanning tree.
    int kruskalMST(int n, vector<vector<int>>& edges, vector<int>& critical, vector<int>& pseudo);

    /// Private sorting function to compare edges based on their weights.
    /// \param x A constant reference to the first edge.
    /// \param y A constant reference to the second edge.
    /// \return A boolean indicating whether the weight of the first edge is less than the weight of the second edge.
    static bool sortEdges(const vector<int>& x, const vector<int>& y);

    /// Private function to calculate the distance between two points.
    /// \param a Coordinates of the first point.
    /// \param b Coordinates of the second point.
    /// \return distance between points a and b.
    double distance(const pair<int, int>& a, const pair<int, int>& b);

    /// Private function to perform Prim's algorithm to find the Minimum Spanning Tree.
    /// \param n Number of localities.
    /// \param localities Vector of coordinates for each locality.
    /// \param minDist Vector to store the minimum distances.
    /// \param visited Vector to track visited nodes.
    void primMST(int n, const vector<pair<int, int>>& localities, vector<double>& minDist, vector<bool>& visited);

    /// Private function to perform Dijkstra's algorithm to find the shortest paths from the capital to all cities,
    /// considering the train routes and roads. Update the result with the maximum train routes
    /// that can be closed without changing the shortest paths.
    /// \param source The capital city (starting node).
    /// \param n The number of cities.
    /// \param adjList The adjacency list representing roads between cities.
    /// \param pq The priority queue to handle train routes and their lengths.
    /// \param result The reference to the variable to store the maximum number of train routes that can be closed.
    void dijkstraCustom(int source, int n, vector<vector<pair<int, int>>>& adjList, priority_queue<pair<pair<int, int>, bool>>& pq, int& result);

    /// Private function to perfom Kruskal's algorithm to find the minimum spanning tree cost
    /// \param N Number of stations
    /// \param M Number of existing tunnels
    /// \param A Cost of building a new tunnel
    /// \param edgeList List of edges (tunnels) represented as (station1, station2, cost)
    /// \return The minimum cost of building a connected metro network
    int kruskalTotalCost(int N, int M, int A, vector<vector<int>> &edgeList);

    /// Private function to find the shortest path using Dijkstra's algorithm
    /// \param src Source node
    /// \param end Destination node
    /// \param N Number of nodes in the graph
    /// \param adjList Adjacency list representing the graph
    /// \return Shortest distance from src to end
    int dijkstra(int src, int end, int N, vector<vector<pair<int, int>>> &adjList);

    /// Private function to check if the given block is inside a grid of size N x N
    /// \param N The size of the grid (both rows and columns are N)
    /// \param block A pair representing the coordinates of a block in the grid (row, column)
    /// \return Returns a boolean value - true if the block is inside the grid, false otherwise
    bool inside(int N, pair<int, int> block);

    /// Private function to perfom Kruskal's algorithm to find the minimum spanning tree cost
    /// \param N Number of nodes
    /// \param edgeList List of edges  represented as (node1, node2, cost)
    /// \return The minimum cost of the MST
    int kruskal(int N, vector<vector<int>>& edges);

    /// Private function to perform Kruskal's algorithm to find the maximum cost for each new edge.
    /// \param N The number of nodes (cities) in the graph.
    /// \param edges The initial edges of the graph.
    /// \param newEdges The additional edges introduced by Marele Lider.
    void kruskalMaximumCost(int N, vector<vector<int>> &edges, vector<vector<int>> &newEdges);






public:

    /// Constructor to initialize the graph with given edges and directed flag.
    Graph(int n, vector<vector<int>>& edges, bool directed);
    /// Constructor without parameters.
    Graph();

    /// Public function to find the cheapest price with a limited number of stops.
    /// \param n An integer representing the number of nodes.
    /// \param flights A vector of vectors representing the flights with [from, to, price].
    /// \param src An integer representing the source node.
    /// \param dst An integer representing the destination node.
    /// \param k An integer representing the maximum number of stops allowed.
    /// \return An integer representing the minimum price to reach the destination.
    /// https://leetcode.com/problems/cheapest-flights-within-k-stops/
    int findCheapestPrice(int n, vector<vector<int>>& flights, int src, int dst, int k);

    /// Public function to find critical and pseudo-critical edges in the graph.
    /// \param n An integer representing the number of nodes in the graph.
    /// \param edges A vector of vectors representing the edges with [from, to, weight].
    /// \return A vector of two vectors: critical edges and pseudo-critical edges.
    /// https://leetcode.com/problems/find-critical-and-pseudo-critical-edges-in-minimum-spanning-tree/
    vector<vector<int>> findCriticalAndPseudoCriticalEdges(int n, vector<vector<int>>& edges);

    /// Public function to determine the existence of distance-limited paths between nodes for given queries.
    /// This function uses Union-Find data structure to efficiently process queries based on edge weights.
    /// \param n An integer representing the number of nodes in the graph.
    /// \param edgeList A vector of vectors representing edges in the graph with [node1, node2, weight].
    /// \param queries A vector of vectors representing queries with [source, destination, weightLimit].
    /// \return A vector of boolean values indicating whether there is a path for each query within the weight limit.
    /// https://leetcode.com/problems/checking-existence-of-edge-length-limited-paths/
    vector<bool> distanceLimitedPathsExist(int n, vector<vector<int>>& edgeList, vector<vector<int>>& queries);

    /// Public function to calculate the minimum cable length required for the network.
    /// \param n Number of localities.
    /// \param localities Vector of coordinates for each locality.
    /// \return Total minimum cable length.
    /// https://www.infoarena.ro/problema/cablaj
    double minimumCableLength(int n, const vector<pair<int, int>>& localities);

    /// Public function to calculate the maximum number of train routes that can be closed
    /// without changing the shortest paths from every city to the capital.
    /// \param n The number of cities.
    /// \param adjList The adjacency list representing roads between cities.
    /// \param pq The priority queue to handle train routes and their lengths.
    /// \return The maximum number of train routes that can be closed.
    /// https://codeforces.com/contest/450/problem/D
    int jzzhuAndCities(int n, vector<vector<pair<int, int>>>& adjList, priority_queue<pair<pair<int, int>, bool>>& pq);

    /// Public function to solve the Rusuoaica problem
    /// \param N Number of stations
    /// \param M Number of existing tunnels
    /// \param A Cost of building a new tunnel
    /// \param edgeList List of edges (tunnels) represented as (station1, station2, cost)
    /// \return The minimum cost of building a connected metro network
    /// https://www.infoarena.ro/problema/rusuoaica
    int rusuoaica(int N, int M, int A, vector<vector<int>> &edgeList);

    /// Public function to solve the camionas problem
    /// \param N Number of nodes in the graph
    /// \param adjList Adjacency list representing the graph
    /// \return Minimum number of roads whose resistance must be increased
    /// https://www.infoarena.ro/problema/camionas
    int camionas(int N, vector<vector<pair<int, int>>> &adjList);

    /// Public function to process the balls movements and calculate maximum ranks at each step
    /// \param N The size of the grid (both rows and columns are N)
    /// \param bile A vector of pairs representing the bile movements (row, column)
    /// \return Returns a stack of integers representing the maximum rank at each step
    /// https://www.infoarena.ro/problema/bile
    stack<int> bile(int N, vector<pair<int, int>>& bile);

    /// Public function to solve oracol problem
    /// \param N Number of nodes
    /// \param edgeList List of edges  represented as (node1, node2, cost)
    /// \return The minimum cost of the MST
    /// https://www.infoarena.ro/problema/oracol
    int oracol(int N, vector<vector<int>>& edges);

    /// Public function to solve apm2 problem.
    /// \param N The number of nodes (cities) in the graph.
    /// \param edges The initial edges of the graph.
    /// \param newEdges The additional edges introduced by Marele Lider.
    /// https://www.infoarena.ro/problema/apm2
    void apm2(int N, vector<vector<int>> &edges, vector<vector<int>> &newEdges);



};


/// Constructor to initialize the graph with given edges and directed flag.
Graph::Graph(int n, vector<vector<int>>& edges, bool directed)
{
    this->n = n; /// Initialize the number of nodes.
    this->adjList.resize(this->n + 1); /// Resize the adjacency list to accommodate 'n' nodes.

    /// Iterate through the given edges and populate the adjacency list.
    for (const auto& edge : edges)
    {
        this->adjList[edge[0]].push_back(edge[1]); /// Add edge from node 'edge[0]' to node 'edge[1]'.

        /// If the graph is undirected, add reverse edge from 'edge[1]' to 'edge[0]'.
        if (directed == false)
        {
            this->adjList[edge[1]].push_back(edge[0]);
        }
    }
}

/// Constructor without parameters.
Graph::Graph()
{
    this->n = 0; /// Initialize the number of nodes to 0 for the default constructor.
    this->adjList = vector<vector<int>>(); /// Initialize the adjacency list as an empty vector of vectors.
}


/// Private function to perform the Bellman-Ford algorithm with a limited number of relaxations.
/// \param prices Reference to a vector storing the minimum prices to each node.
/// \param flights A vector of vectors representing the flights with [from, to, price].
/// \param k An integer representing the maximum number of relaxations (stops) allowed.
void Graph::bellmanFord(vector<int>& prices, const vector<vector<int>>& flights, int k)
{
    /// Iterate for k relaxations (including the initial prices).
    for (int i = 0; i <= k; i++)
    {
        /// Create a temporary vector to store updated prices for this relaxation stage.
        vector<int> tmpPrices = prices;

        /// Iterate through each flight and update the temporary prices.
        for (const auto& flight : flights)
        {
            int from = flight[0];
            int to = flight[1];
            int price = flight[2];

            /// Check if the source node has been reached in previous relaxation stages.
            if (prices[from] == INT_MAX)
                continue;

            /// Update the temporary prices if a shorter path is found.
            if (prices[from] + price < tmpPrices[to])
                tmpPrices[to] = prices[from] + price;
        }

        /// Update the main prices vector with the results of this relaxation stage.
        prices = tmpPrices;
    }
}

/// Private function to initialize a Union-Find data structure with 'n' nodes.
/// \param n An integer representing the number of nodes in the Union-Find set.
/// \param par A reference to a vector storing the parent nodes in the disjoint set.
/// \param rank A reference to a vector storing the rank of each node in the disjoint set.
/// \param maxRank To initialize current maximum rank
void Graph::unionFindConstructor(int n, vector<int>& par, vector<int>& rank, int& maxRank)
{
    par.resize(n); /// Resize the parent vector to accommodate 'n' nodes.
    rank.resize(n, 1); /// Initialize the rank vector with each node having a rank of 1.
    maxRank=0; /// Initialize the current maximum rank.
    for (int i = 0; i < n; i++)
        par[i] = i; /// Initialize each node as its own parent (representative).
}

/// Private function to find the representative (parent) of a node in a disjoint set.
/// Additionally, performs path compression to optimize future find operations.
/// \param node An integer representing the node for which to find the representative.
/// \param par A reference to a vector storing the parent nodes in the disjoint set.
/// \return An integer representing the representative (parent) of the given node.
int Graph::find(int node, vector<int>& par)
{
    while (node != par[node])
    {
        par[node] = par[par[node]]; /// Path compression: Make the grandparent of the current node its parent.
        node = par[node]; /// Move to the parent node.
    }
    return node; /// Return the representative (parent) of the given node.
}

/// Private function to perform the union operation in a disjoint set.
/// \param node1 An integer representing the first node.
/// \param node2 An integer representing the second node.
/// \param par A reference to a vector storing the parent nodes in the disjoint set.
/// \param rank A reference to a vector storing the ranks of nodes in the disjoint set.
/// \param maxRank To store the current maximum rank
/// \return A boolean indicating whether the union operation was successful (true) or not (false).
bool Graph::union1(int node1, int node2, vector<int>& par, vector<int>& rank, int& maxRank)
{
    int p1 = this->find(node1, par); /// Find the representative (parent) of node1.
    int p2 = this->find(node2, par); /// Find the representative (parent) of node2.

    if (p1 == p2)
        return false; /// If both nodes have the same representative, no union is needed.

    if (rank[p1] > rank[p2])
    {
        par[p2] = p1; /// Set the parent of node2 to be the representative of node1.
        rank[p1] += rank[p2]; /// Update the rank of node1.
        maxRank=max(maxRank,rank[p1]); /// Calculate the new maximum rank.
    }
    else
    {
        par[p1] = p2; /// Set the parent of node1 to be the representative of node2.
        rank[p2] += rank[p1]; /// Update the rank of node2.
        maxRank=max(maxRank,rank[p2]); /// Calculate the new maximum rank.
    }

    return true; /// Return true to indicate a successful union operation.
}

/// Private function to perform the Kruskal's algorithm and find critical and pseudo-critical edges.
/// \param n An integer representing the number of nodes.
/// \param edges A reference to a vector of vectors representing the edges with [from, to, weight].
/// \param critical A reference to a vector to store critical edges.
/// \param pseudo A reference to a vector to store pseudo-critical edges.
/// \return An integer representing the weight of the minimum spanning tree.
int Graph::kruskalMST(int n, vector<vector<int>>& edges, vector<int>& critical, vector<int>& pseudo)
{
    int mst_weight = 0; /// Initialize the weight of the minimum spanning tree.
    vector<int> par1, rank1; /// Arrays to store disjoint set information.
    int maxRank1=0;

    /// Initialize disjoint sets for each node.
    this->unionFindConstructor(n, par1, rank1, maxRank1);

    /// Iterate through the edges to build the minimum spanning tree.
    for (int i = 0; i < edges.size(); i++)
        if (this->union1(edges[i][0], edges[i][1], par1, rank1, maxRank1))
            mst_weight += edges[i][2];

    /// Iterate through the edges to identify critical and pseudo-critical edges.
    for (int i = 0; i < edges.size(); i++)
    {
        int weight = 0;
        vector<int> par2, rank2; /// Temporary arrays for each iteration.
        int maxRank2=0;

        /// Initialize disjoint sets for each iteration.
        this->unionFindConstructor(n, par2, rank2, maxRank2);

        /// Check the impact of excluding the current edge on the minimum spanning tree weight.
        for (int j = 0; j < edges.size(); j++)
            if (i != j && this->union1(edges[j][0], edges[j][1], par2, rank2, maxRank2))
                weight += edges[j][2];

        /// If the excluded edge results in a different tree or increases the weight, mark it as critical.
        if (maxRank2 != n || weight > mst_weight)
        {
            critical.push_back(edges[i][3]);
            continue;
        }

        vector<int> par3, rank3; /// Temporary arrays for each iteration.
        int maxRank3=0;

        /// Initialize disjoint sets for each iteration.
        this->unionFindConstructor(n, par3, rank3, maxRank3);


        /// Check the impact of including the current edge on the minimum spanning tree weight.
        this->union1(edges[i][0], edges[i][1], par3, rank3, maxRank3);
        int weight1 = edges[i][2];
        for (int j = 0; j < edges.size(); j++)
            if (this->union1(edges[j][0], edges[j][1], par3, rank3, maxRank3))
                weight1 += edges[j][2];

        /// If the included edge results in the same tree weight, mark it as pseudo-critical.
        if (weight1 == mst_weight)
        {
            pseudo.push_back(edges[i][3]);
        }
    }

    return mst_weight; /// Return the weight of the minimum spanning tree.
}

/// Private sorting function to compare edges based on their weights.
/// \param x A constant reference to the first edge.
/// \param y A constant reference to the second edge.
/// \return A boolean indicating whether the weight of the first edge is less than the weight of the second edge.
bool Graph::sortEdges(const vector<int>& x, const vector<int>& y)
{
    /// Compare edges based on their weights (third element in each vector).
    return x[2] < y[2];
}

/// Private function to calculate the distance between two points.
/// \param a Coordinates of the first point.
/// \param b Coordinates of the second point.
/// \return distance between points a and b.
double Graph::distance(const pair<int, int>& a, const pair<int, int>& b)
{
    return sqrt(pow(a.first - b.first, 2) + pow(a.second - b.second, 2));
}

/// Private function to perform Prim's algorithm to find the Minimum Spanning Tree.
/// \param n Number of localities.
/// \param localities Vector of coordinates for each locality.
/// \param minDist Vector to store the minimum distances.
/// \param visited Vector to track visited nodes.
void Graph::primMST(int n, const vector<pair<int, int>>& localities, vector<double>& minDist, vector<bool>& visited)
{
    /// Start with the first node (arbitrarily chosen as the source)
    minDist[0] = 0;

    /// Loop to select the next node and update distances
    for (int i = 0; i < n; i++)
    {
        /// Find the unvisited node with the smallest minDist value
        int u = -1;
        for (int j = 0; j < n; j++)
        {
            if (visited[j] == false && (u == -1 || minDist[j] < minDist[u]))
            {
                u = j;
            }
        }

        /// Mark the selected node as visited
        visited[u] = true;

        /// Update distances to neighbors of the selected node
        for (int v = 0; v < n; v++)
        {
            double dist = distance(localities[u], localities[v]);
            /// If the neighbor is unvisited and the new distance is smaller, update minDist
            if (visited[v] == false && dist < minDist[v])
            {
                minDist[v] = dist;
            }
        }
    }
}

/// Private function to perform Dijkstra's algorithm to find the shortest paths from the capital to all cities,
/// considering the train routes and roads. Update the result with the maximum train routes
/// that can be closed without changing the shortest paths.
/// \param source The capital city (starting node).
/// \param n The number of cities.
/// \param adjList The adjacency list representing roads between cities.
/// \param pq The priority queue to handle train routes and their lengths.
/// \param result The reference to the variable to store the maximum number of train routes that can be closed.
void Graph::dijkstraCustom(int source, int n, vector<vector<pair<int, int>>>& adjList, priority_queue<pair<pair<int, int>, bool>>& pq, int& result)
{
    /// Vector to store the distances from the capital to each city.
    vector<int> dist(n, INT_MAX);

    /// Vector to mark if a city is visited during the algorithm.
    vector<bool> visited(n, false);

    /// Initialize the priority queue with the starting node and the information that no train is used (false).
    pq.push(make_pair(make_pair(0, source), false));
    dist[source] = 0;

    /// Main loop of Dijkstra's algorithm
    while (!pq.empty())
    {
        /// Extract the top element from the priority queue
        int currentNode = pq.top().first.second;
        int currentDist = -pq.top().first.first;
        bool usingTrain = pq.top().second;
        pq.pop();

        /// Check if the current node is visited and the train is used
        if (dist[currentNode] <= currentDist && usingTrain == true)
        {
            /// If the train route is used and the distance is not improved, increment the result
            result++;
            continue;
        }

        /// If the train is used and the distance is improved, update the distance
        else if (dist[currentNode] > currentDist && usingTrain == true)
        {
            dist[currentNode] = currentDist;
        }

        /// If the node is already visited, skip the rest of the loop
        if (visited[currentNode] == true)
        {
            continue;
        }

        /// Mark the current node as visited
        visited[currentNode] = true;

        /// Iterate through neighbors of the current node
        for (const auto& neighbor : adjList[currentNode])
        {
            int nextNode = neighbor.first;
            int edgeWeight = neighbor.second;

            /// Relaxation step: If a shorter path is found, update the distance and push into the priority queue
            if (currentDist + edgeWeight < dist[nextNode])
            {
                dist[nextNode] = currentDist + edgeWeight;
                pq.push(make_pair(make_pair(-dist[nextNode], nextNode), false));
            }
        }
    }
}

/// Private function to perfom Kruskal's algorithm to find the minimum spanning tree cost
/// \param N Number of stations
/// \param M Number of existing tunnels
/// \param A Cost of building a new tunnel
/// \param edgeList List of edges (tunnels) represented as (station1, station2, cost)
/// \return The minimum cost of building a connected metro network
int Graph::kruskalTotalCost(int N, int M, int A, vector<vector<int>> &edgeList)
{
    /// Sort edges based on weights
    sort(edgeList.begin(), edgeList.end(), this->sortEdges);

    int cost = 0, cnt = 0; /// Initialize variables for cost and edge count

    /// Initialize Union-Find data structure
    vector<int> par, rank;
    int maxRank;
    this->unionFindConstructor(N+1,par,rank,maxRank);


    /// Iterate through sorted edges
    for (int i = 0; i < M; i++)
    {
        if (this->find(edgeList[i][0],par) != this->find(edgeList[i][1], par) && edgeList[i][2] <= A)
        {
            /// If adding the edge doesn't create a cycle and its weight is within limit A
            this->union1(this->find(edgeList[i][0],par), this->find(edgeList[i][1], par), par, rank, maxRank);
            cost += edgeList[i][2];
            cnt++;
        }
        else
        {
            /// If adding the edge creates a cycle or its weight is beyond limit A
            cost -= edgeList[i][2];
        }
    }

    /// Calculate and return the total cost considering additional connections
    int result = cost + (N - cnt - 1) * A;
    return result;
}

/// Private function to find the shortest path using Dijkstra's algorithm
/// \param src Source node
/// \param end Destination node
/// \param N Number of nodes in the graph
/// \param adjList Adjacency list representing the graph
/// \return Shortest distance from src to end
int Graph::dijkstra(int src, int end, int N, vector<vector<pair<int, int>>> &adjList)
{
    /// Initialize distance array to store the shortest distances from the source
    vector<int> dist(N, INT_MAX);
    dist[src] = 0;

    /// Priority queue to keep track of the nodes with the smallest tentative distances
    /// Each element in the priority queue is a pair representing {-distance, node}
    priority_queue<pair<int, int>> pq;
    pq.push({0, src});

    /// Dijkstra's algorithm loop
    while (!pq.empty())
    {
        /// Extract the node with the smallest tentative distance from the priority queue
        int currNode = pq.top().second;
        pq.pop();

        /// Explore neighbors of the current node
        for (auto neighbor : adjList[currNode])
        {
            int nextNode = neighbor.first;
            int weight = -neighbor.second; /// Negate the weight as we are using a min-heap

            /// Relaxation step: Update the distance if a shorter path is found
            if (dist[currNode] != INT_MAX && dist[currNode] + weight < dist[nextNode])
            {
                dist[nextNode] = dist[currNode] + weight;
                /// Add the updated distance and the next node to the priority queue
                pq.push({-dist[nextNode], nextNode});
            }
        }
    }

    /// Return the shortest distance to the destination node
    return dist[end];
}

/// Private function to check if the given block is inside a grid of size N x N
/// \param N The size of the grid (both rows and columns are N)
/// \param block A pair representing the coordinates of a block in the grid (row, column)
/// \return Returns a boolean value - true if the block is inside the grid, false otherwise
bool Graph::inside(int N, pair<int, int> block)
{
    return (block.first >= 1 && block.second >= 1 && block.first <= N && block.second <= N);
}

/// Private function to perfom Kruskal's algorithm to find the minimum spanning tree cost
/// \param N Number of nodes
/// \param edgeList List of edges  represented as (node1, node2, cost)
/// \return The minimum cost of the MST
int Graph::kruskal(int N, vector<vector<int>>& edges)
{
    /// Sort edges based on weights
    sort(edges.begin(), edges.end(), this->sortEdges);

    int result = 0;

    /// Initialize Union-Find data structure
    vector<int> par, rank;
    int maxRank;
    this->unionFindConstructor(N+1,par,rank,maxRank);

    /// Iterate through sorted edges
    for (int i=0; i < edges.size(); i++)
        if(this->union1(edges[i][0],edges[i][1],par,rank,maxRank)) /// Check if adding the current edge forms a cycle or not.
            result += edges[i][2];

    return result;

}

/// Private function to perform Kruskal's algorithm to find the maximum cost for each new edge.
/// \param N The number of nodes (cities) in the graph.
/// \param edges The initial edges of the graph.
/// \param newEdges The additional edges introduced by Marele Lider.
void Graph::kruskalMaximumCost(int N, vector<vector<int>> &edges, vector<vector<int>> &newEdges)
{
    /// Sort the edges based on weights.
    sort(edges.begin(), edges.end(), sortEdges);

    /// Initialize Union-Find data structure.
    vector<int> par, rank;
    int maxRank;
    this->unionFindConstructor(N,par,rank,maxRank);

    /// Iterate through sorted edges.
    for (int i = 0; i < edges.size(); i++)
    {
        /// Check if adding the edge creates a cycle in the Minimum Spanning Tree (MST).
        if (this->find(edges[i][0],par) != this->find(edges[i][1],par))
        {
            /// Union the sets of the nodes connected by the current edge in the MST.
            this->union1(this->find(edges[i][0],par), this->find(edges[i][1],par),par,rank,maxRank);

            /// Check each new edge to see if it connects nodes already in the MST and hasn't been processed.
            for (int j = 0; j < newEdges.size(); j++)
            {
                /// Check if the new edge connects the same sets as the current edge in the MST and hasn't been processed yet.
                if (this->find(newEdges[j][0],par) == this->find(newEdges[j][1],par) && newEdges[j][2] == -1)
                {
                    /// Update the cost of the new edge to be one less than the weight of the current edge.
                    newEdges[j][2] = edges[i][2] - 1;
                }
            }
        }
    }

}






/// Public function to find the cheapest price with a limited number of stops.
/// \param n An integer representing the number of nodes.
/// \param flights A vector of vectors representing the flights with [from, to, price].
/// \param src An integer representing the source node.
/// \param dst An integer representing the destination node.
/// \param k An integer representing the maximum number of stops allowed.
/// \return An integer representing the minimum price to reach the destination.
/// https://leetcode.com/problems/cheapest-flights-within-k-stops/
int Graph::findCheapestPrice(int n, vector<vector<int>>& flights, int src, int dst, int k)
{
    /// Initialize prices vector with maximum values, and set the source price to 0.
    vector<int> prices(n, INT_MAX);
    prices[src] = 0;

    /// Use the Bellman-Ford algorithm to update prices with a limited number of stops.
    this->bellmanFord(prices, flights, k);

    /// Check if the destination is reachable and return the result.
    if (prices[dst] == INT_MAX)
        return -1;
    else
        return prices[dst];
}

/// Public function to find critical and pseudo-critical edges in the graph.
/// \param n An integer representing the number of nodes in the graph.
/// \param edges A vector of vectors representing the edges with [from, to, weight].
/// \return A vector of two vectors: critical edges and pseudo-critical edges.
/// https://leetcode.com/problems/find-critical-and-pseudo-critical-edges-in-minimum-spanning-tree/
vector<vector<int>> Graph::findCriticalAndPseudoCriticalEdges(int n, vector<vector<int>>& edges)
{
    /// Add index information to each edge for identification after sorting.
    for (int i = 0; i < edges.size(); i++)
        edges[i].push_back(i);

    /// Sort edges based on weight using the custom comparator function.
    sort(edges.begin(), edges.end(), this->sortEdges);

    /// Vector to store indices of critical and pseudo-critical edges.
    vector<int> critical, pseudo;

    /// Perform Kruskal's algorithm to find critical and pseudo-critical edges.
    this->kruskalMST(n, edges, critical, pseudo);

    /// Create a vector to store the result, with the first element as critical edges and the second as pseudo-critical edges.
    vector<vector<int>> result;
    result.push_back(critical);
    result.push_back(pseudo);

    return result;
}

/// Public function to determine the existence of distance-limited paths between nodes for given queries.
/// This function uses Union-Find data structure to efficiently process queries based on edge weights.
/// \param n An integer representing the number of nodes in the graph.
/// \param edgeList A vector of vectors representing edges in the graph with [node1, node2, weight].
/// \param queries A vector of vectors representing queries with [source, destination, weightLimit].
/// \return A vector of boolean values indicating whether there is a path for each query within the weight limit.
/// https://leetcode.com/problems/checking-existence-of-edge-length-limited-paths/
vector<bool> Graph::distanceLimitedPathsExist(int n, vector<vector<int>>& edgeList, vector<vector<int>>& queries)
{
    /// Add an index to each query for later identification
    for(int i = 0; i < queries.size(); i++)
        queries[i].push_back(i);

    /// Sort queries and edgeList based on weights using the sortEdges comparison function
    sort(queries.begin(), queries.end(), sortEdges);
    sort(edgeList.begin(), edgeList.end(), sortEdges);

    /// Initialize the result vector with false values
    vector<bool> result(queries.size(), false);

    /// Initialize Union-Find data structure with parent and rank arrays
    vector<int> par, rank;
    int maxRank;
    this->unionFindConstructor(n, par, rank, maxRank);

    /// Index for iterating through edgeList
    int j = 0;

    /// Process each query and update Union-Find structure accordingly
    for(int i = 0; i < queries.size(); i++)
    {
        /// Process edges until the weight limit of the current query
        while(j < edgeList.size() && edgeList[j][2] < queries[i][2])
        {
            /// Union nodes if the edge weight is within the limit
            this->union1(edgeList[j][0], edgeList[j][1], par, rank, maxRank);
            j++;
        }

        /// Check if the source and destination nodes of the query are in the same connected component
        if(this->find(queries[i][0], par) == this->find(queries[i][1], par))
            result[queries[i][3]] = true; /// Set the result to true if there is a path within the weight limit
    }

    return result;
}

/// Public function to calculate the minimum cable length required for the network.
/// \param n Number of localities.
/// \param localities Vector of coordinates for each locality.
/// \return Total minimum cable length.
/// https://www.infoarena.ro/problema/cablaj
double Graph::minimumCableLength(int n, const vector<pair<int, int>>& localities)
{
    /// Initialize vectors for visited nodes and minimum distances
    vector<bool> visited(n, false);
    vector<double> minDist(n, INT_MAX);

    /// Apply Prim's algorithm to find the minimum spanning tree
    this->primMST(n, localities, minDist, visited);

    /// Calculate the total length of the minimum cable required
    double result = 0.0;
    for (int i = 0; i < n; i++)
    {
        result += minDist[i];
    }

    return result;
}

/// Public function to calculate the maximum number of train routes that can be closed
/// without changing the shortest paths from every city to the capital.
/// \param n The number of cities.
/// \param adjList The adjacency list representing roads between cities.
/// \param pq The priority queue to handle train routes and their lengths.
/// \return The maximum number of train routes that can be closed.
/// https://codeforces.com/contest/450/problem/D
int Graph::jzzhuAndCities(int n, vector<vector<pair<int, int>>>& adjList, priority_queue<pair<pair<int, int>, bool>>& pq)
{
    int result = 0;

    /// Call the private dijkstra function to calculate the result
    this->dijkstraCustom(0, n, adjList, pq, result);

    return result;
}

/// Public function to solve the Rusuoaica problem
/// \param N Number of stations
/// \param M Number of existing tunnels
/// \param A Cost of building a new tunnel
/// \param edgeList List of edges (tunnels) represented as (station1, station2, cost)
/// \return The minimum cost of building a connected metro network
/// https://www.infoarena.ro/problema/rusuoaica
int Graph::rusuoaica(int N, int M, int A, vector<vector<int>> &edgeList)
{
    int result;
    result = this->kruskalTotalCost(N, M, A, edgeList);
    return result;
}

/// Public function to solve the camionas problem
/// \param N Number of nodes in the graph
/// \param adjList Adjacency list representing the graph
/// \return Minimum number of roads whose resistance must be increased
/// https://www.infoarena.ro/problema/camionas
int Graph::camionas(int N, vector<vector<pair<int, int>>> &adjList)
{
    /// Call the dijkstra function to find the shortest path
    int result;
    result = this->dijkstra(0, N - 1, N, adjList);
    return result;
}

/// Public function to process the balls movements and calculate maximum ranks at each step
/// \param N The size of the grid (both rows and columns are N)
/// \param bile A vector of pairs representing the bile movements (row, column)
/// \return Returns a stack of integers representing the maximum rank at each step
/// https://www.infoarena.ro/problema/bile
stack<int> Graph::bile(int N, vector<pair<int, int>>& bile)
{
    /// Create an instance of the UnionFind class with the total number of nodes in the grid
    vector<int> par,rank;
    int maxRank=0;
    this->unionFindConstructor(N*N+1,par,rank,maxRank);
    /// Stack to store the maximum ranks at each step
    stack<int> result;

    /// Directions for adjacent movements: Up, Left, Right, Down
    vector<vector<int>> direct = {{-1, 0}, {0, -1}, {0, 1}, {1, 0}};
    /// Keeps track of whether a block is taken or not
    vector<bool> takenStatus(N * N, false);

    /// Process balls movements in reverse order (from last to first)
    for (int i = N * N - 1; i >= 0; i--)
    {
        /// Push the current maximum rank into the result stack
        result.push(maxRank);
        int r = bile[i].first;
        int c = bile[i].second;

        /// Mark the current position as taken
        takenStatus[(r - 1) * N + c] = true;

        /// Check adjacent positions to perform union if they are taken
        for (const auto &dir : direct)
        {
            int newR = r + dir[0];
            int newC = c + dir[1];
            /// Check if the adjacent position is inside the grid
            if (this->inside(N, {newR, newC}))
            {
                /// Check if the adjacent position is also taken
                if (takenStatus[(newR - 1) * N + newC] == true)
                {
                    /// Union the current position with the adjacent taken position
                    this->union1((r - 1) * N + c, (newR - 1) * N + newC, par, rank, maxRank);
                }
            }
        }
    }

    /// Return the stack containing maximum ranks at each step
    return result;
}

/// Public function to solve oracol problem
/// \param N Number of nodes
/// \param edgeList List of edges  represented as (node1, node2, cost)
/// \return The minimum cost of the MST
/// https://www.infoarena.ro/problema/oracol
int Graph::oracol(int N, vector<vector<int>>& edges)
{

    int result;
    result=this->kruskal(N,edges);
    return result;

}

/// Public function to solve apm2 problem.
/// \param N The number of nodes (cities) in the graph.
/// \param edges The initial edges of the graph.
/// \param newEdges The additional edges introduced by Marele Lider.
/// https://www.infoarena.ro/problema/apm2
void Graph::apm2(int N, vector<vector<int>> &edges, vector<vector<int>> &newEdges)
{
    this->kruskalMaximumCost(N,edges,newEdges);
}



int main()
{
    int cnt=0;
    while(true)
    {
        displayMenu();
        int command;
        cin>>command;
        switch(command)
        {
        case 1:
        {
            clearScreen();
            cout<<"n= ";
            int n;
            cin>>n;
            int x, y, z;
            cout<<endl<<"flights (enter -1 to stop)= ";
            vector<vector<int>> flights;
            while (cin>>x && x!=-1 && cin>>y && cin>>z)
            {
                flights.push_back({x, y, z});
            }
            cout<<endl;
            int src;
            cout<<"src= ";
            cin>>src;

            int dst;
            cout<<"dst= ";
            cin>>dst;

            int k;
            cout<<"k= ";
            cin>>k;

            Graph G;
            cout<<endl;
            cout<<G.findCheapestPrice(n, flights, src, dst, k);
            cout<<endl;
            cout<<"Press 'Enter' to return to the menu."<<endl;
            cin.ignore();
            while(cin.get() != '\n');
            clearScreen();
            break;


        }

        case 2:
        {
            clearScreen();
            cout<<"n= ";
            int n;
            cin>>n;
            int x, y, z;
            cout<<endl<<"edges (enter -1 to stop)= ";
            vector<vector<int>> edges;
            while (cin>>x && x!=-1 && cin>>y && cin>>z)
            {
                edges.push_back({x, y, z});
            }
            cout<<endl;
            Graph G;
            vector<vector<int>> result;
            result=G.findCriticalAndPseudoCriticalEdges(n,edges);
            for(int i=0; i<result.size(); i++)
            {
                for(int j=0; j<result[i].size(); j++)
                    cout<<result[i][j]<<" ";
                cout<<endl;
            }


            cout<<"Press 'Enter' to return to the menu."<<endl;
            cin.ignore();
            while(cin.get() != '\n');
            clearScreen();
            break;
        }

        case 3:
        {
            clearScreen();
            cout<<"n= ";
            int n;
            cin>>n;
            int x, y, z;
            cout<<endl<<"edgeList (enter -1 to stop)= ";
            vector<vector<int>> edgeList;
            while (cin>>x && x!=-1 && cin>>y && cin>>z)
            {
                edgeList.push_back({x, y, z});
            }
            cout<<endl<<"queries (enter -1 to stop)= ";
            vector<vector<int>> queries;
            while (cin>>x && x!=-1 && cin>>y && cin>>z)
            {
                queries.push_back({x, y, z});
            }
            cout<<endl;
            Graph G;
            vector<bool> result;
            result=G.distanceLimitedPathsExist(n,edgeList,queries);
            for(int i=0; i<result.size(); i++)
            {
                cout<<result[i]<<" ";
            }
            cout<<endl;

            cout<<"Press 'Enter' to return to the menu."<<endl;
            cin.ignore();
            while(cin.get() != '\n');

            clearScreen();
            break;
        }
        case 4:
        {
            clearScreen();


            cout<<"n= ";
            int n;
            cin >> n;
            int x, y;
            vector<pair<int, int>> localities;
            cout<<endl;
            cout<<"localities= ";
            for (int i = 0; i < n; i++)
            {
                cin >> x >> y;
                localities.emplace_back(x, y);
            }
            cout<<endl;
            Graph G;
            double result = G.minimumCableLength(n, localities);
            cout << fixed << setprecision(4) << result;

            cout<<endl;
            cout<<"Press 'Enter' to return to the menu."<<endl;
            cin.ignore();
            while(cin.get() != '\n');
            clearScreen();
            break;
        }
        case 5:
        {
            clearScreen();

            cout<<"n, m, k= ";
            int n, m, k;
            cin >> n >> m >> k;

            vector<vector<pair<int, int>>> adjList(n);
            priority_queue<pair<pair<int, int>, bool>> pq;

            cout<<endl<<"u, v, x= ";
            for (int i = 0; i < m; ++i)
            {
                int u, v, x;
                cin >> u >> v >> x;
                u--, v--;
                adjList[u].emplace_back(v, x);
                adjList[v].emplace_back(u, x);
            }

            cout<<endl<<"s, y= ";
            for (int i = 0; i < k; ++i)
            {
                int s, y;
                cin >> s >> y;
                s--;
                pq.push(make_pair(make_pair(-y, s), true));
            }

            cout<<endl;
            Graph G;
            int result = G.jzzhuAndCities(n, adjList, pq);

            cout << result;

            cout<<endl;
            cout<<"Press 'Enter' to return to the menu."<<endl;
            cin.ignore();
            while(cin.get() != '\n');
            clearScreen();
            break;
        }
        case 6:
        {
            clearScreen();

            int N, M, A;
            cout<<"N, M, A= ";
            cin >> N >> M >> A;
            vector<vector<int>> edgeList;
            cout<<endl;

            int t1, t2, t3;
            cout<<"t1, t2, t3= ";

            for (int i = 0; i < M; i++)
            {
                cin >> t1 >> t2 >> t3;
                edgeList.push_back({t1, t2, t3});
            }

            cout<<endl;
            Graph G;
            int result;
            result = G.rusuoaica(N, M, A, edgeList);
            cout<<result;

            cout<<endl;
            cout<<"Press 'Enter' to return to the menu."<<endl;
            cin.ignore();
            while(cin.get() != '\n');
            clearScreen();
            break;
        }
        case 7:
        {
            clearScreen();


            int N, M, G;
            cout<<"N, M, G= ";
            cin >> N >> M >> G;
            vector<vector<pair<int, int>>> adjList(N);
            cout<<endl;

            int x, y, g;
            cout<<"x, y, g= ";
            for (int i = 0; i < M; i++)
            {
                cin >> x >> y >> g;
                x--;
                y--;

                if (g >= G)
                {
                    adjList[x].emplace_back(y, 0);
                    adjList[y].emplace_back(x, 0);
                }
                else
                {
                    adjList[x].emplace_back(y, -1);
                    adjList[y].emplace_back(x, -1);
                }
            }

            cout<<endl;

            Graph G1;

            cout << G1.camionas(N, adjList);

            cout<<endl;
            cout<<"Press 'Enter' to return to the menu."<<endl;
            cin.ignore();
            while(cin.get() != '\n');

            clearScreen();
            break;
        }
        case 8:
        {
            clearScreen();


            int N;
            cout<<"N= ";
            cin >> N;
            cout<<endl;
            vector<pair<int, int>> bile;
            int L, C;
            cout<<"L, C= ";
            for (int i = 0; i < N * N; i++)
            {
                cin >> L >> C;
                bile.emplace_back(L, C);
            }

            Graph G;
            stack<int> result=G.bile(N,bile);

            cout<<endl;

            while (!result.empty())
            {
                if (result.top() == 0)
                {
                    if (result.size() == 1)
                    {
                        cout << result.top() << endl;
                    }
                    else
                    {
                        cout << 1 << endl;
                    }
                }
                else
                {
                    cout << result.top() << endl;
                }
                result.pop();
            }

            cout<<endl;
            cout<<"Press 'Enter' to return to the menu."<<endl;
            cin.ignore();
            while(cin.get() != '\n');

            clearScreen();
            break;
        }
        case 9:
        {
            clearScreen();




            int N;
            cout<<"N= ";
            cin>>N;
            cout<<endl;
            vector<vector<int>> edges;
            int C;
            cout<<"C= ";
            for(int i=0; i<N; i++)
                for(int j=i+1; j<=N; j++)
                {
                    cin>>C;
                    edges.push_back({i,j,C});
                }

            cout<<endl;
            Graph G;
            cout<<G.oracol(N,edges);

            cout<<endl;
            cout<<"Press 'Enter' to return to the menu."<<endl;
            cin.ignore();
            while(cin.get() != '\n');
            clearScreen();
            break;
        }
        case 10:
        {
            clearScreen();



            int N, M, Q;
            cout<<"N, M, Q= ";
            cin >> N >> M >> Q;
            vector<vector<int>> edges;
            cout<<endl;
            cout<<"X, Y, T= ";
            int X, Y, T;
            for (int i = 0; i < M; i++)
            {
                cin >> X >> Y >> T;
                X--;
                Y--;
                edges.push_back({X, Y, T});
            }

            vector<vector<int>> newEdges;
            cout<<endl;
            cout<<"A, B= ";
            int A, B;
            for (int i = 0; i < Q; i++)
            {
                cin >> A >> B;
                A--;
                B--;
                newEdges.push_back({A, B, -1});
            }
            cout<<endl;

            Graph G;
            G.apm2(N, edges, newEdges);

            for (int i = 0; i < newEdges.size(); i++)
            {
                cout << newEdges[i][2] << endl;
            }

            cout<<"Press 'Enter' to return to the menu."<<endl;
            cin.ignore();
            while(cin.get() != '\n');
            clearScreen();
            break;
        }

        case 11:
        {
            clearScreen();
            cnt=1;
            break;

        }

        }
        if(cnt==1)
        {
            clearScreen();
            break;
        }
    }

    return 0;
}
