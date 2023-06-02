import numpy as np
import gmshparser

def read_msh(name):
    mesh = gmshparser.parse(name)

    print(mesh)
    node_coordinates=np.zeros((mesh.get_number_of_nodes(),2))
    for entity in mesh.get_node_entities():
        for node in entity.get_nodes():
            nid = node.get_tag()
            ncoords = node.get_coordinates()
            node_coordinates[node.get_tag()-1]=np.array(node.get_coordinates())[0:2]
            #print("Node id = %s, node coordinates = %s" % (nid, ncoords))


    ind_sorted=np.lexsort((node_coordinates[:,1],node_coordinates[:,0]))
    node_coordinates=node_coordinates[ind_sorted]
    pre_Sorted=np.argsort(ind_sorted)

    connection_table=[]
    dirichlet_nodes=[]

    for entity in mesh.get_element_entities():
        eltype = entity.get_element_type()
        
        #print("Element type: %s" % eltype)
        if eltype==8:
            for element in entity.get_elements():
                elcon=element.get_connectivity()
                con=[pre_Sorted[i-1] for i in elcon]
                dirichlet_nodes+=con
        if eltype==9:
            for element in entity.get_elements():
                elid = element.get_tag()
                elcon = element.get_connectivity()
                con=[pre_Sorted[i-1] for i in elcon]
                connection_table.append(con)
                #print("Element id = %s, connectivity = %s" % (elid, elcon))
                #print(con)
    return node_coordinates,connection_table,list(set(dirichlet_nodes))
