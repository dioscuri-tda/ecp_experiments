import numpy as np

def create_local_graph(points, i, threshold, dbg=False):
    center_vertex = points[i]

    # enumeration needs to start from i+1 because the center is at position i
    id_neigs_of_center_vectex = [j for j,point in enumerate(points[i+1:], i+1) 
                                 if np.linalg.norm(center_vertex - point) <= threshold]
    
    if dbg: print(id_neigs_of_center_vectex)
       
    # create the center graph as a list of lists
    # each list corespond to a node and contains its neighbours with distance
    # note, edges are always of the type
    # (i, j) with i<j in the ordering. 
    # This is to save space, we do not need the edge (j, i) 
    # In practice, we are building onlty the upper triangular part
    # of the adjecency matrix
    
    considered_graph = []
    
    # add central vertex
    mapped_center_vertex = []
    # enumeration needs to start from 1 because 0 is the center
    for j, neigh in enumerate(id_neigs_of_center_vectex, 1):
        mapped_center_vertex.append( (j, np.linalg.norm(center_vertex - points[neigh])) )
        
    considered_graph.append(mapped_center_vertex)
    
    # add the rest
    for j, neigh in enumerate(id_neigs_of_center_vectex, 1):
        
        if dbg: print(j, neigh)
        
        neighbours_of_j = []
        
        # add the others
        # note that the index k starts from 1, be careful with the indexing
        for z, other_neigh in enumerate(id_neigs_of_center_vectex[j:], j+1):
            if dbg: print('    ', z, other_neigh)
            dist = np.linalg.norm(points[neigh] - points[other_neigh])
            if (dist <= threshold):
                neighbours_of_j.append( (z, dist) )
        
        considered_graph.append(neighbours_of_j)
              
    return considered_graph

def compute_ECC_single_vertex(considered_graph, dbg=False):
    # this is the list of simplices in the current dimension. 
    # We will overwrite it everytime we move to a higher dimension.
    simplices_in_current_dimension = []
    
    # this is a list which tells us what are the filtration values of 
    # the simplices in the simplices_in_current_dimension vector. 
    # We keep it not in a structure to make the code faster.
    filtration_of_those_simplices = []
    
    # the value of Euler characteristic will only change at some number 
    # of real numbers, equal to the lenth of edges. We will keep in in the structure of a map
    ECC = dict()
    ECC[0] = 1
    
    # first we will fill in simplices_in_current_dimension vector 
    # with the edges from central element to its neighbors with higher id:
    for edge in considered_graph[0]:
        simplices_in_current_dimension.append([0, edge[0]])
        filtration_of_those_simplices.append(edge[1])
        
        # changes to ECC due to those edges:
        if dbg: print('######Change of ECC at the level : {} by: -1'.format(edge[1]))
        ECC[edge[1]] = ECC.get(edge[1], 0) - 1
        
    if dbg:
        print('simplices_in_current_dimension :')
        for i,s in enumerate(simplices_in_current_dimension):
            print('[{}] --> {}'.format(s, filtration_of_those_simplices[i]))
            
    # We do not need to list the neighbors of the certal vertex, 
    # as they are vertices with numbers between 1 and len(considered_graph[0])
    last_neigh_of_current_vertex = len(considered_graph[0])
    if dbg: print('last_neigh_of_current_vertex : {}'.format(last_neigh_of_current_vertex))
    
    # now we can compute all the neighbors of the created 1-dimensional simplices:
    common_neighs = []
    
    # keep track of the number of simplices discovered
    number_of_simplices = 1 + len(simplices_in_current_dimension)
    
    for simplex in simplices_in_current_dimension:
        # we need to check which vertices among neighs_of_neighs_of_centre 
        # are common neighs of the vertices in this simplex.
        the_other_vertex = simplex[1]
        if dbg: print('We will search for the common neighbors of the edge [0,{}]'.format(the_other_vertex))
        neighs = []
        if dbg: print('The common neigh is : ')
        for neigh_of_other in considered_graph[the_other_vertex]:
            neighs.append(neigh_of_other[0])
            if dbg: print('\t{}'.format(neigh_of_other[0]))

        common_neighs.append(neighs)
    
    
    if dbg:
        print('Here is final common_neighs list')
        for i in range(len(common_neighs)):
            print('For the edge : [{},{}] we have the common neighs : '.format(simplices_in_current_dimension[i][0],
                                                                               simplices_in_current_dimension[i][1]))
            print(common_neighs[i])
            print('Moreover its filtration is : {}'.format(filtration_of_those_simplices[i]))
            
    
    # We will use this datastructure for quick computations of intersections of neighbor lists
    neighs_of_vertices = []
    for vertex_list in considered_graph:
        neighs_of_vertices.append(set([v[0] for v in vertex_list]))

    # Now we have created the list of edges, and each of the edge is equipped 
    # with common_neighs and all filtration values. Now, we can now create all higher dimensional simplices:
    dimension = 2
    dimm = 1
    while ( len(simplices_in_current_dimension) > 0 ):
        # first we declare all cointainters that we need.
        new_simplices_in_current_dimension = []
        new_filtration_of_those_simplices = []
        new_common_neighs = []

        # the real computations begins here:
        for i in range(len(simplices_in_current_dimension)):
            if dbg:
                print('Consider simplex : {}'.format(simplices_in_current_dimension[i]))
                print('common_neighs[i].size() : {}'.format(len(common_neighs[i])))
                
            # let us check if we can extend simplices_in_current_dimension[i]
            for j in range(len(common_neighs[i])):                
                if dbg: print('\n{} can be extended by adding a vertex : {}'.format(simplices_in_current_dimension[i],
                                                                                    common_neighs[i][j]))

                #we can extend simplices_in_current_dimension[i] by adding vertex common_neighs[i][j]
                new_simplex = simplices_in_current_dimension[i].copy()
                new_simplex.append(common_neighs[i][j])
                new_simplices_in_current_dimension.append( new_simplex )
                number_of_simplices += 1
                if dbg: print('Adding new simplex : {}'.format(new_simplex))

                # now once we have the new simplex, we need to compute its filtration and common neighs
                # let us start with the filtration. We will set it up initially to the 
                # filtration of simplices_in_current_dimension[i]:
                filtration_of_this_simplex = filtration_of_those_simplices[i]
                for k in range(len(simplices_in_current_dimension[i])):
                    # check what is the weight of an edge from simplices_in_current_dimension[i][k] 
                    # to common_neighs[i][j]
                    length_of_this_edge = 0
                    for l in range(len(considered_graph[ simplices_in_current_dimension[i][k]])):
                        if ( considered_graph[ simplices_in_current_dimension[i][k] ][l][0] == common_neighs[i][j] ):
                            length_of_this_edge = considered_graph[ simplices_in_current_dimension[i][k] ][l][1]
                            break
                    if ( length_of_this_edge > filtration_of_this_simplex ):
                        filtration_of_this_simplex = length_of_this_edge
                        
                new_filtration_of_those_simplices.append( filtration_of_this_simplex )
                
                if dbg: print('#####Change of ECC at the level : {} by: {}'.format(filtration_of_this_simplex,
                                                                                   dimm))
                ECC[filtration_of_this_simplex] = ECC.get(filtration_of_this_simplex, 0) + dimm


                if dbg: print('The filtration of this simplex is : {}'.format(filtration_of_this_simplex))
                    
                # now we still need to deal with the common neighbors.
                neighs_of_new_simplex = []
                new_vertex = common_neighs[i][j]
                if dbg: print('The common neigh of the new vertex are  : {}'.format(neighs_of_vertices[new_vertex]))
                if dbg: print('The common neigh of the old simplex are : {}'.format(common_neighs[i]))
                for k in range(len(common_neighs[i])):
                    if common_neighs[i][k] in neighs_of_vertices[new_vertex]: 
                        neighs_of_new_simplex.append( common_neighs[i][k] )
                
                if dbg: print('the common neight of the simplex {} are {}'.format(new_simplex, neighs_of_new_simplex))
    
                new_common_neighs.append( neighs_of_new_simplex )
        
        if dbg : print('Moving to the next dimension...\n')
        
        simplices_in_current_dimension = new_simplices_in_current_dimension
        filtration_of_those_simplices = new_filtration_of_those_simplices
        common_neighs = new_common_neighs
        dimension += 1
        dimm = dimm*(-1)

    if dbg : print('Out of the loop, return result. \n')     
                               
    return ECC, number_of_simplices



def compute_local_contributions(point_cloud, epsilon):
    # for each point, create its local graph and find all the
    # simplices in its star
    
    ECC_list = []
    total_number_of_simplices = 0

    for i in range(len(point_cloud)):
        graph_i = create_local_graph(point_cloud, i, epsilon)
        
        local_ECC, number_of_simplices = compute_ECC_single_vertex(graph_i)
        ECC_list.append(local_ECC)
        total_number_of_simplices += number_of_simplices

    total_ECC = dict()

    for single_ECC in ECC_list:
        for key in single_ECC:
            total_ECC[key] = total_ECC.get(key, 0) + single_ECC[key]

    # remove the contributions that are 0
    to_del = []
    for key in total_ECC:
        if total_ECC[key] == 0:
            to_del.append(key)
    for key in to_del:
        del total_ECC[key]
        
    return sorted(list(total_ECC.items()), key = lambda x: x[0]), total_number_of_simplices