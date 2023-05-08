# -*- coding: utf-8 -*-
"""
Created on Sat May  6 11:15:03 2023

@author: 43802
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import sys
def string_to_array(input_string):
    input_list = input_string.split(';')
    input_array = np.zeros((len(input_list)-1, 3), dtype='U3')
    temp = np.zeros((1, 3),  dtype='U3')
    temp[0][0] = input_list[0]
    for i in range(len(input_list)-1):
        string = input_list[i+1].split()
        input_array[i, :len(string)] = string
    input_array = np.vstack((temp, input_array))
    return input_array

def depth(input_array):
    """
    Check the max number in a 1D array
    the 1D array has the same size as the max num of qubit
    We track the gate apply on which qubit and add one to the 
    corresponded element of the 1D array

    Parameters
    ----------
    input_array : string 2D array
        The text describe the circuit

    Returns
    -------
    depth : int
        The depth of the circuit

    """
    depth = 0
    num_qubit = int(input_array[0][0])
    temp_two = np.ones(num_qubit)
    for i in range(1, len(input_array)):
        temp_one = np.zeros(num_qubit)
        if input_array[i][0] == "h":
            temp_one[int(input_array[i][1])] = 1
        elif input_array[i][0] == "rz":
            temp_one[int(input_array[i][2])] = 1
        elif input_array[i][0] == "cx":
            temp_one[int(input_array[i][1])] = 1
            temp_one[int(input_array[i][2])] = 1
        temp = temp_one+temp_two
        valve = False
        for j in range(len(temp)):
            temp_two[j] = temp[j]
            if temp[j] > 1:
                valve = True
        if valve:
            temp_two = temp_one
            depth += 1
    return depth

def divide_block(input_array, num_sub_circuit):
    """
    put the same depth gates in the same block
    if the block has more than two rz gates
    start a new block

    Parameters
    ----------
    input_array : string 2D array
        The text describe the circuit

    Returns
    -------
    sub_blocks : array list
    Each element in the list represent a circuit block that is the integration 
    of the gates in the same circuit depth

    """
    sub_blocks = []
    count = 1
    count_rz = 0
    last_depth = 1
    temp_array = np.zeros((0, len(input_array[0])))
    while(count!=len(input_array)):
        if(input_array[count][0]=="rz"):
            count_rz+=1
        new_depth = depth(input_array[:count+1])
        if(new_depth==last_depth):
            if count_rz<=1:
                temp_array = np.vstack((temp_array,input_array[count]))
            elif count_rz>1:
                sub_blocks.append(temp_array)
                temp_array = np.zeros((0,len(input_array[0])))
                temp_array = np.vstack((temp_array, input_array[count]))
                count_rz = 0
        elif(new_depth!=last_depth):
            last_depth = new_depth
            sub_blocks.append(temp_array)
            temp_array = np.zeros((0,len(input_array[0])))
            temp_array = np.vstack((temp_array, input_array[count]))
            count_rz = 0
        count+=1
    sub_blocks.append(temp_array)
    
    sub_circuit = []
    num_sub_block = int(len(sub_blocks)/num_sub_circuit)
    remainder = len(sub_blocks)%num_sub_circuit
    temp_circuit = input_array[0]
    while(len(sub_blocks)!=0):
        if(remainder!=0):
            temp_blocks = sub_blocks[0:num_sub_block+1]
            remainder-=1
            del sub_blocks[0:num_sub_block+1]
        elif(remainder==0):
            temp_blocks = sub_blocks[0:num_sub_block]
            del sub_blocks[0:num_sub_block]
        for i in range(len(temp_blocks)):
            temp_circuit = np.vstack((temp_circuit, temp_blocks[i]))
            bar = ['','','']
            temp_circuit = np.vstack((temp_circuit,np.array(bar)))
        sub_circuit.append(temp_circuit)
        temp_circuit = input_array[0]
        
    return sub_circuit
        
        
        
    
    
def nodes(input_array):
    """
    Similar to the depth calculation process
    We need a 1D array and a temp array
    All elements in 1D array is zero except the elements that has the same index
    with the gate
    For example, if h 3, The third element(index 2) of the 1D array will raise 1.
    We let temp array loop over the input_array and add with 1D array
    if any elements larger than 2. We create a edge.
    
    Parameters
    ----------
    input_array : array

    Returns
    -------
    node_array : array
        [[name of the gate],[n_th_row],[link_to_m_th_row],[link_to_j_th_row]]

    """
    num_qubit = int(input_array[0][0])
    node_list = []
    temp_two = np.ones(num_qubit,dtype=object)
    #loop over the input array
    for i in range(1, input_array.shape[0]):
        new_depth = depth(input_array[:i+1])
        temp_one = np.zeros(num_qubit, dtype=object)
        temp_node_array = []
        temp_node_array.extend([str(new_depth)])#track depth
        #current gates
        if input_array[i][0] == "h":
            temp_one[int(input_array[i][1])] = 1
            temp_node_array.extend([input_array[i][0], str(i)])
        elif input_array[i][0] == "rz":
            temp_one[int(input_array[i][2])] = 1
            temp_node_array.extend([input_array[i][0], str(i)])
        elif input_array[i][0] == "cx":
            temp_one[int(input_array[i][1])] = 1
            temp_one[int(input_array[i][2])] = 1
            temp_node_array.extend([input_array[i][0], str(i)])
        #loop over the rest gates
        for j in range(i + 1, input_array.shape[0]):
            temp_two = np.zeros(num_qubit)
            if input_array[j][0] == "h":
                temp_two[int(input_array[j][1])] = 1
            elif input_array[j][0] == "rz":
                temp_two[int(input_array[j][2])] = 1
            elif input_array[j][0] == "cx":
                temp_two[int(input_array[j][1])] = 1
                temp_two[int(input_array[j][2])] = 1
            #now we combine temp one and temp two
            #if any elements greater than one
            #we record the index this will be, for h and rz, out 1, and both
            #out1 and out2 for cx
            if temp_node_array[1] == "h" or temp_node_array[1] == "rz":
                temp = temp_one + temp_two
                valve = False
                for k in range(temp.shape[0]):
                    if temp[k] > 1:
                        valve = True
                if valve:
                    temp_node_array.append(str(j))
                    break
            elif temp_node_array[1] == "cx":
                temp = temp_one + temp_two
                valve = False
                count = 0
                
                for k in range(temp.shape[0]):
                    if temp[k] > 1:
                        valve = True
                        count += 1
                        temp_one[k] = 0
                if valve and count == 1:
                    temp_node_array.append(str(j))
                    if len(temp_node_array) == 5:
                        break
                elif valve and count == 2:
                    temp_node_array.extend([str(j), str(j)])
                    break
        
        node_list.append(temp_node_array)
    num_rows = len(node_list)
    node_array = np.empty((num_rows, 5), dtype=object)
    
    # Iterate over the list and fill the array row by row
    for i, row in enumerate(node_list):
        for j, element in enumerate(row):
            node_array[i, j] = element
    for i in range(len(node_array)):
        for j in range(len(node_array[i])):
            if node_array[i][j] == None:
                node_array[i][j] = ''
    return node_array


def plot_dag(data):

    dag = nx.DiGraph()
    
    # Create nodes and edge
    for node in data:
        num_depth, gate_name, current, out1, out2 = node
        name = gate_name + current  # name is op + index
        dag.add_node(current, op=name, name=name)
        if out1:
            dag.add_edge(current, out1)
        if out2:
            dag.add_edge(current, out2)

    # Position nodes based on their depth
    pos = {}
    drift = 1
    for node in dag.nodes:
        depth = int(data[int(node)-1][0])
        if depth not in pos:
            pos[depth] = 0
        pos[depth] += 1
        if drift == 1:
            drift = 0
        elif drift == 0:
            drift = 0.5
        elif drift == 0.5:
            drift = 1
            
        y = -pos[depth] - drift*0.5-0.5*int(current)*drift
        pos[node] = (depth,y)
    #pos = nx.kamada_kawai_layout(dag)
    #plot
    nx.draw_networkx_nodes(dag, pos, node_color="lightblue", node_size=800)
    node_labels = nx.get_node_attributes(dag, 'name')
    nx.draw_networkx_labels(dag, pos, labels=node_labels, font_size=14, font_family="Arial")
    nx.draw_networkx_edges(dag, pos, edge_color="gray", arrows=True)
    plt.axis("off")
    plt.show()
    

def main():
    
    input_string_one = "6;h 0;h 1;h 4;rz 0 0;cx 1 2;cx 4 5;cx 0 1;cx 2 3;h 2;h 3;cx 1 2;cx 3 5;rz 0 3;cx 4 3;cx 3 0"
    input_string_two = "3;h 2;cx 2 1;rz 0 1;cx 0 1;rz 1 1;cx 2 1;rz 1 1;cx 0 1;rz 1 1;cx 0 2;rz 0 1;cx 0 2;rz 0 0;rz 0 2;h 2"
    
    #convert to string array
    input_array_one = string_to_array(input_string_one)
    input_array_two = string_to_array(input_string_two)
    print(input_array_one)
    print(input_array_two)
    #calculate depth
    depth_one = depth(input_array_one)
    depth_two = depth(input_array_two)
    print(depth_one)
    print(depth_two)
    #sub circuit
    divide_block_one = divide_block(input_array_one, 4)# devide to 4 subcircuit
    divide_block_two = divide_block(input_array_two, 2)# devide to 2 subcircuit
    print(divide_block_one)
    print(divide_block_two)
    #calculate nodes
    nodes_one = nodes(input_array_one)
    nodes_two = nodes(input_array_two)
    print(nodes_one, nodes_two)
    #plot
    #plot_dag(nodes_one)
    plot_dag(nodes_two)
main()