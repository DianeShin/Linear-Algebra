import numpy as np
from numpy import linalg as la

np.set_printoptions(precision=3)
 
class DiGraph:
    """A class for representing directed graphs via their adjacency matrices.

    Attributes:
        (fill this out after completing DiGraph.__init__().)
    """
    # Problem 1
    def __init__(self, A, labels=None):
        """Modify A so that there are no sinks in the corresponding graph,
        then calculate Ahat. Save Ahat and the labels as attributes.

        Parameters:
            A ((n,n) ndarray): the adjacency matrix of a directed graph.
                A[i,j] is the weight of the edge from node j to node i.
            labels (list(str)): labels for the n nodes in the graph.
                If None, defaults to [0, 1, ..., n-1].                

        Examples
        ========
        >>> A = np.array([[0, 0, 0, 0],[1, 0, 1, 0],[1, 0, 0, 1],[1, 0, 1, 0]])
        >>> G = DiGraph(A, labels=['a','b','c','d'])
        >>> G.A_hat
        array([[0.   , 0.25 , 0.   , 0.   ],
               [0.333, 0.25 , 0.5  , 0.   ],
               [0.333, 0.25 , 0.   , 1.   ],
               [0.333, 0.25 , 0.5  , 0.   ]])
        >>> steady_state_1 = G.linsolve()
        >>> { k: round(steady_state_1[k],3) for k in steady_state_1}
        {'a': 0.096, 'b': 0.274, 'c': 0.356, 'd': 0.274}
        >>> steady_state_2 = G.eigensolve()
        >>> { k: round(steady_state_2[k],3) for k in steady_state_2}
        {'a': 0.096, 'b': 0.274, 'c': 0.356, 'd': 0.274}
        >>> steady_state_3 = G.itersolve()
        >>> { k: round(steady_state_3[k],3) for k in steady_state_3}
        {'a': 0.096, 'b': 0.274, 'c': 0.356, 'd': 0.274}
        >>> get_ranks(steady_state_3)
        ['c', 'b', 'd', 'a']
        """
        self.A = A
        self.labels = labels

        AZeroColumn = np.all(A == 0, axis=0)   
        ANoSink = np.copy(A)
        
        for column in range(np.shape(ANoSink)[1]):
            if AZeroColumn[column] == True:
                ANoSink[:, column] = np.ones(np.shape(A)[0], dtype=int)

        self.A_hat = ANoSink.astype(dtype= float)
        column_sum = np.sum(ANoSink, axis=0)
        for row in range(np.shape(A)[0]):
            for column in range(np.shape(A)[1]):
                self.A_hat[row][column] /= column_sum[column]


    def linsolve(self, epsilon=0.85):
        """Compute the PageRank vector using the linear system method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Returns:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        coeff = np.identity(np.shape(self.A)[0]) - epsilon * self.A_hat
        const = ((1-epsilon)/np.shape(self.A)[0]) * np.ones(np.shape(self.A)[0])
        x = np.linalg.solve(coeff, const)
        return dict(zip(self.labels, x))


    def eigensolve(self, epsilon=0.85):
        """Compute the PageRank vector using the eigenvalue method.
        Normalize the resulting eigenvector so its entries sum to 1.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        A_bar = epsilon * self.A_hat + ((1-epsilon)/np.shape(self.A)[0]) * np.ones(np.shape(self.A))
        
        eigval, eigvec = np.linalg.eig(A_bar)
        result = eigvec[:,0]
        for val in range(eigval.size):
            if (eigval[val] == 1.000e+00) :
                result =  eigvec[:,val]
        result = result / np.sum(result)
        return dict(zip(self.labels, result))

    def itersolve(self, epsilon=0.85, maxiter=100, tol=1e-12):
        """Compute the PageRank vector using the iterative method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.
            maxiter (int): the maximum number of iterations to compute.
            tol (float): the convergence tolerance.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """

        x_0 = (np.ones(np.shape(self.A)[0]) / np.shape(self.A)[0]).T
        x_0.shape = (np.shape(self.A)[0], 1)

        for iter in range (maxiter):
            x_1 = np.matmul(epsilon * self.A_hat + ((1-epsilon) / np.shape(self.A)[0]) * np.ones(np.shape(self.A)) , x_0) 
            x_1 = x_1 / np.linalg.norm(x_1)
            if np.linalg.norm(x_1 - x_0) < tol :
                break
            x_0 = x_1
        x_1 = x_1.T[0]
        x_1 = x_1 / np.sum(x_1)
        return dict(zip(self.labels, x_1))


def get_ranks(d):
    """Construct a sorted list of labels based on the PageRank vector.

    Parameters:
        d (dict(str -> float)): a dictionary mapping labels to PageRank values.

    Returns:
        (list) the keys of d, sorted by PageRank value from greatest to least.
    """    
    return list({k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}.keys())


# Task 2
def rank_websites(filename="web_stanford.txt", epsilon=0.85):
    """Read the specified file and construct a graph where node j points to
    node i if webpage j has a hyperlink to webpage i. Use the DiGraph class
    and its itersolve() method to compute the PageRank values of the webpages,
    then rank them with get_ranks().

    Each line of the file has the format
        a/b/c/d/e/f...
    meaning the webpage with ID 'a' has hyperlinks to the webpages with IDs
    'b', 'c', 'd', and so on.

    Parameters:
        filename (str): the file to read from.
        epsilon (float): the damping factor, between 0 and 1.

    Returns:
        (list(str)): The ranked list of webpage IDs.

    Examples
    ========
    >>> print(rank_websites()[0:5])
    ['98595', '32791', '28392', '77323', '92715']
    """

    #create label list  
    labelSet = set()
    labelList = []
    with open(filename) as file:
        for line in file:
            wordList = line.replace("\n", "").split('/')
            for word in wordList:
                labelSet.add(word)
    labelList = list(labelSet)

    #create label dictionary
    label_index_dict = {}
    for index in range(len(labelList)):
        label_index_dict[labelList[index]] = index

    #create adjacency matrix
    adjacency_matrix = np.zeros((len(labelList), len(labelList)))

    #fill adjacency matrix
    with open(filename) as file:
        for line in file:
            line.strip()
            wordList = line.replace("\n", "").split('/')
            for word in wordList:
                if (word == wordList[0]):
                    continue
                else :
                    adjacency_matrix[label_index_dict[word]][label_index_dict[wordList[0]]] = 1

    #create Digraph object
    graph = DiGraph(adjacency_matrix , labelList)
    return get_ranks(graph.itersolve(epsilon=epsilon))

# Task 3
def rank_uefa_teams(filename, epsilon=0.85):
    """Read the specified file and construct a graph where node j points to
    node i with weight w if team j was defeated by team i in w games. Use the
    DiGraph class and its itersolve() method to compute the PageRank values of
    the teams, then rank them with get_ranks().

    Each line of the file has the format
        A,B
    meaning team A defeated team B.

    Parameters:
        filename (str): the name of the data file to read.
        epsilon (float): the damping factor, between 0 and 1.

    Returns:
        (list(str)): The ranked list of team names.

    Examples
    ========
    >>> rank_uefa_teams("psh-uefa-2018-2019.csv",0.85)[0:5]
    ['Liverpool', 'Ath Madrid', 'Paris SG', 'Genk', 'Barcelona']
    """
    #create label list  
    labelSet = set()
    labelList = []
    with open(filename) as file:
        for line in file:
            wordList = line.split(',')
            labelSet.add(wordList[0])
            labelSet.add(wordList[1])
    labelList = list(labelSet)

    #create label dictionary
    label_index_dict = {}
    for index in range(len(labelList)):
        label_index_dict[labelList[index]] = index
    
    #create adjacency matrix
    adjacency_matrix = np.zeros((len(labelList), len(labelList)))
    
    #fill adjacency matrix
    with open(filename) as file:
        for line in file:
            wordList = line.replace("\n", "").split(',')
            if (int(wordList[2]) < int(wordList[3])):  # left lost to right -> node from left to right
                adjacency_matrix[label_index_dict[wordList[1]]][label_index_dict[wordList[0]]] += 1
            elif (int(wordList[2]) > int(wordList[3])): # right lost to left -> node from right to left
                adjacency_matrix[label_index_dict[wordList[0]]][label_index_dict[wordList[1]]] += 1

    #create Digraph object
    graph = DiGraph(adjacency_matrix , labelList)
    return get_ranks(graph.itersolve(epsilon=epsilon))

if __name__ == "__main__":
    import doctest
    doctest.testmod()