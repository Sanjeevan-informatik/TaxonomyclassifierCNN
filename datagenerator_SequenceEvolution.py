# Use this cell for your code
from numpy import array, diag
import numpy as np
from scipy.linalg import expm, eig, inv
from math import exp, sqrt
from taxonomy_assignment import lablel_taxonomy_data
def GetStationaryDistribution(Q):
  # Compute pi by solving piQ = [0 0 0 0]
  # Note that pi = [0, 0, 0, 0] is not a feasible solution to 
  # piQ = [0 0 0 0] because pi is a probability distribution and its elements must sum to 1
  # How do we ensure this? 
  # We construct an augmented matrix Q_aug by adding a 
  # column to Q (first column) that contains 1 in each row
  # We do this because multiplying pi = [pi(a) pi(t) pi(g) pi(c)] with [1 1 1 1]^{T} will give the scalar 1
  # To ensure that pi is a probability distribution We solve for pi using ordinary 
  # least-squares regression such that piQ_aug = [1 0 0 0 0]  
    b = np.array([1,0,0,0,0])
    b.shape=(5,1)
    Q_aug = np.c_[np.ones(4),Q]
    Q_aug = Q_aug.transpose()
    pi_stationary = np.linalg.lstsq(Q_aug,b,rcond=None)[0]
    pi_stationary = pi_stationary.transpose()[0]
    return (pi_stationary)

def ComputeProbabilityMatrix_using_eigenvaluedecomposition(Q,t):
    # Compute stationary distribution
    P = np.zeros((4,4))        
    pi = GetStationaryDistribution(Q)
    pi_sqrt = list(map(sqrt,pi))
    PI_sqrt = diag(pi_sqrt)
    PI_sqrt_inv = inv(PI_sqrt)  
    PI_sqrt_Q_PI_sqrt_inv = PI_sqrt.dot(Q).dot(PI_sqrt_inv)    
    D, B = eig(PI_sqrt_Q_PI_sqrt_inv)    
    A = PI_sqrt_inv.dot(B)
    A_inv = inv(A)
    # Q = ADA^-1
    # Qt = ADtA^-1
    D_scaled = D*t # We are interesting in matrix exponentiation at time t
    # exp(D)[i,i] = exp(D[i,i]); exp(D)[i,j] = 0 (i not equal to j)
    expD_scaled = diag(list(map(exp,D_scaled))) 
    # P = exp(Qt) = Aexp(Dt)A^-1
    P = A.dot(expD_scaled).dot(A_inv)    
    return (D, B, P)


from scipy.linalg import expm


def GenerateUNRESTRateMatrix():
  Q = np.zeros((4,4))  
  for row in range(4):
    for col in range(4):
      if row!=col:
        Q[row][col] = np.random.uniform(0, 1, 1) 
  # ensure that each row of Q sums to zero  
  for row in range(4):
    row_sum = sum(Q[row])
    Q[row][row] = -1 * (row_sum)
  pi = GetStationaryDistribution(Q)
  mu = -1 * sum([pi[i]*Q[i][i] for i in range(4)])
  Q/=mu
  return(Q)

# Use this cell for your code
from copy import deepcopy
import re
import numpy as np
from math import exp, log

class Vertex:
    def __init__(self,name):
        self.name = name
        self.parent = self
        self.children = []
        self.neighbors = []
        self.timesVisited = 0
        self.degree = 0
        self.inDegree = 0
        self.outDegree = 0
        self.newickLabel = ""
        self.sequence = ""        

class Tree:
    def __init__(self): 
        # here we collect the attributes a tree class requires
        self.vertices = []
        self.leaves = []
        self.newickLabel = ""
        # vertice map
        self.vmap = {}
        # edge lenghts
        self.edgeLengthMapForDirectedEdges = {}
        self.edgeLengthMapForUndirectedEdges = {}
        # vertice list
        self.verticesForPostOrderTraversal = []
        self.verticesForPreOrderTraversal = []
        self.newickLabelForLeavesSet_flag = False
        self.root = ""
        self.sequenceLength = 0
        self.pi_rho = [0]*4
        self.parsimonyScore = 0
        self.logLikelihoodScore = 0
        self.DNA_map = {"A":0,"C":1,"G":2,"T":3}
        self.DNA = "ACGT"
        self.transitionMatrices = {}
        self.rateMatrices = {}
        self.conditionalLikelihoodVectors = {}
    def GetEdgeLength(self,parent, child):
      # this is self explanatory: we basically have the edge lenght as a distance mesure for instance
        return (self.edgeLengthMapForDirectedEdges[(parent.name,child.name)])
    def ContainsVertex(self,v_name):
      # check if a given vertex is alreay in the vertex dict
        is_vertex_in_tree = v_name in self.vmap.keys()
        return (is_vertex_in_tree)
    def AddVertex(self,v_name):
      # add vertex to the vertex dict
      # in needs to be a class of Vertex
        v = Vertex(v_name)
        self.vmap[v_name] = v
        self.vertices.append(v)
    def GetVertex(self,v_name):
      return (self.vmap[v_name])
    def AddEdge(self,parent_name, child_name, edge_length):
      # adds an edge between parent and child with a given edge lenght
      # add parent to the vertex map
        p = self.vmap[parent_name]
      # add children to the vertex map
        c = self.vmap[child_name]
      # assign the parent to the child 
        c.parent = p
      # assing children to the parent 
        p.children.append(c)
      # increase the indegree of a child by one
        c.inDegree += 1
      # increase the outdegree of a parent by one
        p.outDegree += 1
      # increase the degree of a child by one (for undirected tree)
        c.degree += 1
      # increase the degree of a parent by one (for undirected tree)
        p.degree += 1
      # add the parent, child and the edge length to the edgeLengthMapForDirectedEdges dict       
        self.edgeLengthMapForDirectedEdges[(parent_name,child_name)] = edge_length      
    def ReadEdgeList(self,edgeListFileName):
      # this function basically reads the csv file and populates the data structure
        file = open(edgeListFileName,"r")
        for line in file:
            if len(line.split(",")) != 3:
                print (line)
            u_name, v_name, length = line.split(",")
            u_name = u_name.strip()
            v_name = v_name.strip()
            length = float(length.strip())
            # add the vertexes to the vertex map
            if not self.ContainsVertex(u_name):
                self.AddVertex(u_name)
            if not self.ContainsVertex(v_name):
                self.AddVertex(v_name)
            # adds edge between the vertexes
            self.AddEdge(u_name, v_name, length)
        file.close()
        self.SetLeaves()        
        self.SetRoot()
        self.SetVerticesForPostOrderTraversal()
        self.SetVerticesForPreOrderTraversal()
    def PrintEdgeList(self):
        for edge in self.edgeLengthMapForDirectedEdges.keys():
            length = self.edgeLengthMapForDirectedEdges[edge]
            print (edge[0],edge[1],length)    
    def SetLeaves(self):
      # the vertexes with the vertex degree 1 are the leves
        for v in self.vertices:
            if v.degree == 1:
                self.leaves.append(v)
    def SetRoot(self):
      # the vertex with the indegree 0 is root
        for v in self.vertices:
            if v.inDegree == 0:
                self.root = v              
    def SetNewickLabelForLeaves(self):
      # sets the vertex name as newick label
        if (self.newickLabelForLeavesSet_flag):
            pass
        else:
            self.newickLabelForLeavesSet_flag = True
            for v in self.leaves:
                v.newickLabel = v.name
    def ResetTimesVisited(self):
        for v in self.vertices:
            v.timesVisited = 0        
    def ComputeNewickFormat(self):
        print ("Computing newick format")
        self.SetNewickLabelForLeaves()
        if len(self.verticesForPostOrderTraversal) == 0:
          # get the hidden vertices
            self.SetVerticesForPostOrderTraversal()        
        for v in self.verticesForPostOrderTraversal:
          # get both the children
            child_left, child_right = v.children
          # get the distance/edge lenght between the left child and the parent  
            length_left = self.GetEdgeLength(v,child_left)
          # get the distance/edge lenght between the right child and the parent
            length_right = self.GetEdgeLength(v,child_right)
          # write it in a newick format
            v.newickLabel = "(" + child_left.newickLabel + ":" + str(length_left)
            v.newickLabel += "," + child_right.newickLabel + ":" + str(length_right) + ")"
        self.newickLabel = v.newickLabel + ";"         
    def SetVerticesForPostOrderTraversal(self):
      # start from leaves and visit all the vertices such that children are visited before parents
        for v in self.leaves:
            self.verticesForPostOrderTraversal.append(v)
        # create a deep copy of the leaves list so that the original remains unchanged 
        # : any changes made to a copy of object do not reflect in the original object.
        verticesToVisit = []
        for v in self.leaves:
          verticesToVisit.append(v)
        while len(verticesToVisit) > 0:
          # goes over the vertices and removes them iteratively from the "verticesToVisit" list
          # always remove a vertex from the end of the list
            c = verticesToVisit.pop()
          # p is the parent of the leave
            p = c.parent
          # increase the timesVisited variable by one           
            p.timesVisited += 1
          # if both the children of a parent are traversed, append the parent to "verticesForPostOrderTraversal" list
            if p.timesVisited == 2:
                self.verticesForPostOrderTraversal.append(p)
                # if both the children are traversed and if the vertex is a hidden node (not the root),
                # add the hidden node to the of verticesToVisit list
                if p.inDegree == 1:
                    verticesToVisit.append(p)
    def SetVerticesForPreOrderTraversal(self):
      if len(self.verticesForPostOrderTraversal) == 0:
        self.SetVerticesForPostOrderTraversal()
      self.verticesForPreOrderTraversal = []
      num_vertices = len(self.vertices)
      for i in range(num_vertices):
        self.verticesForPreOrderTraversal.append(self.verticesForPostOrderTraversal[num_vertices-1-i])      
    def AddSequences(self,sequences_dic):
      for v_name in sequences_dic.keys():
        v = self.GetVertex(v_name)
        v.sequence = sequences_dic[v.name]              
        self.SetSequenceLength(len(sequences_dic[v.name]))
    def SetSequenceLength(self,sequence_length):
      self.sequenceLength = sequence_length
    def ReadNewickFile(self,newickFileName):
        file = open(newickFileName,"r")
        newick_string = file.readline()
        file.close()
        hidden_vertex_ind = 1        
        # use escape character for matching parenthesis 
        rx = r'\([^()]+\)'
        numTries = 0        
        while "(" in newick_string:            
            numTries += 1
            # search for the paranthesis
            m = re.search(rx,newick_string)
            # returns a tuple containing all the subgroups of the match "()"
            string_match = m.group()            
            # remove ( and )
            siblings_string = string_match[1:-1]
            c_left_name_and_length, c_right_name_and_length = siblings_string.split(",")
            c_left_name, c_left_length = c_left_name_and_length.split(":")
            c_right_name, c_right_length = c_right_name_and_length.split(":")
            if not self.ContainsVertex(c_left_name):
                self.AddVertex(c_left_name)
            if not self.ContainsVertex(c_right_name):
                self.AddVertex(c_right_name)
            hidden_vertex_name = "h" + str(hidden_vertex_ind)
            self.AddVertex(hidden_vertex_name)            
            self.AddEdge(hidden_vertex_name, c_left_name, float(c_left_length))
            self.AddEdge(hidden_vertex_name, c_right_name, float(c_right_length))
            newick_string = newick_string.replace(string_match,hidden_vertex_name)
            hidden_vertex_ind += 1                                
        self.SetLeaves()
        self.SetRoot()
  
  
    def GenerateHMMonTree(self,model="UNREST"):
      self.rateMatrices = {}
      if model == "UNREST":
        Q = GenerateUNRESTRateMatrix()        
        for v in self.vertices:
          if v != self.root:
            self.rateMatrices[v.name] = Q
        pi = GetStationaryDistribution(Q)        
        
      self.SetPiRho(pi)
      self.ComputeAndSetTransitionMatrices()
      return(None)
    def SetPiRho(self,pi_rho):
        self.pi_rho = pi_rho[:]
        return (None)
    def ComputeAndSetTransitionMatrices(self):
      print("Computing and setting transition matrices")
      self.transitionMatrices = {}      
      for v in self.vertices:
        if v.name!= self.root.name:
          p = v.parent
          t = self.GetEdgeLength(p,v)
          Q = self.rateMatrices[v.name]
          P = expm(Q*t)          
          self.transitionMatrices[v.name] = P                
      return (None)
    def SimulateSequenceEvolution(self):
      # first, simulate the root sequence
      root_sequence = ""
      # for each base in the seqeunce
      for _ in range(self.sequenceLength):
        # draw from a uniform distribution
        u = np.random.uniform(0,1,1)
        # initialize the cumulative sum
        cumSum=0
        for base in range(4):
          # get the stationary distribution of the bases
          # sum the stationary distribution proportions of the bases
          # until it exceeds your drawn dist u
            cumSum += self.pi_rho[base]
            if cumSum > u:
                break              
        root_sequence+=self.DNA[base]
      # initialize the root distribution
      self.root.sequence = root_sequence
      # get the vertices from the PRE-order traversal             
      for c in self.verticesForPreOrderTraversal:
        # for each vertex        
        if c.name != self.root.name:
          # get the parent          
          p = c.parent
          # get transition matrix (key: child name)          
          P = self.transitionMatrices[c.name]
          # obtain the sequence of the parent
          parent_sequence = p.sequence
          child_sequence = ""
          # evolve the sequence
          for parent_char in parent_sequence:
            # index of the parent character
            # reminder: self.DNA_map = {"A":0,"C":1,"G":2,"T":3}
            parent_char_ind = self.DNA_map[parent_char]
            # draw from uniform dist
            u = np.random.uniform(0,1,1)
            cumSum=0
            for child_char_ind in range(4):
              # sum the transition probablilites until it reaches to the
              # drawn value u is reached
              cumSum += P[parent_char_ind][child_char_ind]
              if cumSum > u:
                break
            # once it exceeds, evolved child seq is determined             
            child_sequence+=self.DNA[child_char_ind]
            c.sequence = child_sequence 
      return child_sequence


def taxonomy_datagenerator(taxonomylength,numberoftaxonomy):
    T = Tree()
    edge_list_file_name_T = 'edge_list_E4_problem3'

    T.ReadEdgeList(edge_list_file_name_T)
    # Construct a stationary homogeneous non-reversible CT-HMM on trees
    # Assign same UNREST rate matrix to each edge of the tree
    T.GenerateHMMonTree(model="UNREST")
    T.SetSequenceLength(taxonomylength)

    with open("sequence_data.txt", 'w') as f:
        for x in range(numberoftaxonomy):
            seq = T.SimulateSequenceEvolution()
            seq = seq+"\n"
            f.write(seq)
    f.close()
    
    lablel_taxonomy_data()   
    print(str(numberoftaxonomy)+' taxonomy data with '+str(numberoftaxonomy)+' length with label was created successfully')
    

taxonomylength=50
numberoftaxonomy=1000
taxonomy_datagenerator(taxonomylength,numberoftaxonomy)