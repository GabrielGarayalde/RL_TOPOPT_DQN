# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 11:07:48 2023

@author: gabri
"""

import math
import matplotlib.pyplot as plt
import numpy as np
from Classes_RL import Action
        
def line(node1_index, node2_index, Nodes):
    
    node1 = Nodes[node1_index]
    node2 = Nodes[node2_index]
    
    x1 = node1.x
    y1 = node1.y
    x2 = node2.x
    y2 = node2.y
    
    if x1 == x2:
        m = None
        b = None
    else:
        m = (y2 - y1) / (x2 - x1)
        b = y2 - m*x2
            
    return m, b

def computeDistance(node1_index,node2_index, Nodes):
    
    node1 = Nodes[node1_index]
    node2 = Nodes[node2_index]

    x1 = node1.x
    y1 = node1.y
    x2 = node2.x
    y2 = node2.y
    
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def containsNodes(node1_index, node2_index, Nodes):
    
    node1 = Nodes[node1_index]
    node2 = Nodes[node2_index]

    min_x = min(node1.x, node2.x)
    max_x = max(node1.x, node2.x)
    min_y = min(node1.y, node2.y)
    max_y = max(node1.y, node2.y)

    nodes = []

    for node in Nodes:
        if min_x == node.x == max_x and min_y < node.y < max_y:
            nodes.append(node.ID)
        elif min_x < node.x < max_x and min_y == node.y == max_y:
            nodes.append(node.ID)
        elif min_x < node.x < max_x and min_y < node.y < max_y:
            nodes.append(node.ID)

    return nodes
            
    
def Mesh(m,n,y,x):
    nodes = []
    for j in range(0,m*y,y):
        for k in range(0,n*x,x):
            nodes.append([k,j])
    return nodes

def PlotElem(elements,Nodes, grid):
    x,y = [],[]
    for index, element in enumerate(elements):
        node1 = Nodes[element[0]]
        node2 = Nodes[element[1]]
        
        x.append(node1.x)
        y.append(node1.y)
        x.append(node2.x)
        y.append(node2.y)
        #print(x,y)
        plt.plot(x,y)
        plt.text((x[0]+x[1])/2,(y[0]+y[1])/2,index)
        x,y = [],[]
    plt.scatter(*zip(*grid),s=2)
    plt.show()

def ActiveNodes(Elements, Nodes):
    #important to copy or else the deletion would occur to both the objects
    active_nodes = []
    Nodes = list(range(len(Nodes)))
    for element in Elements:
        active_nodes.append(element[0])
        active_nodes.append(element[1])
        
    active_nodes = sorted(set(active_nodes))
    
    inactive_nodes = [x for x in Nodes if x not in active_nodes]
    
    return active_nodes, inactive_nodes

def ThirdPointNodes(element, active_nodes):
    
    third_point_nodes = active_nodes.copy()
    values_to_delete = [element[0], element[1]]

    for value in values_to_delete:
        while value in third_point_nodes:
            third_point_nodes.remove(value)
            
    return third_point_nodes

def AllActionsList(Elements, Nodes):
    active_nodes, inactive_nodes = ActiveNodes(Elements, Nodes)
    AllActions = []
    ID_counter = 0
    for inactive_node in inactive_nodes:
            for element in range(len(Elements)):
                for operator in range(2):
                    action = Action(inactive_node, element, operator, ID_counter)
                    AllActions.append(action)
                    ID_counter = ID_counter + 1
    return AllActions

def AllowableActionsList(AllActions):
    
    AllowedActions = []
    UnallowedActions = []

    for action in AllActions:
        if action.allowed:
            AllowedActions.append(action)
        else:
            UnallowedActions.append(action)

    return AllowedActions, UnallowedActions


def ComputeVolume(Elements, Area, Nodes):
    volume = 0
    for element in Elements:
        node1 = element[0]
        node2 = element[1]
        element_volume = computeDistance(node1, node2, Nodes) * Area
        volume = volume + element_volume

    return volume
        
        
# is this the determinant??? in case should it be equal or greater than??
def ccw(A,B,C,grid):
	return (grid[C][1]-grid[A][1])*(grid[B][0]-grid[A][0]) > (grid[B][1]-grid[A][1])*(grid[C][0]-grid[A][0])

# these input variables are node IDs
def LineIntersectCheck(A, B, C, D, grid):
    
    #the intersection algorithm cant handle 2 lines that share a common endpoint
    #this is the case i'm excluding here
    if A == C or A == D or B == C or B == D:
        return False
    result = ccw(A,C,D,grid) != ccw(B,C,D,grid) and ccw(A,B,C,grid) != ccw(A,B,D,grid)
    # if result:
        # print('line intersect check failed')
    return result


#returns False if the check is passed
def PointIntersectCheck(node_index, element, Nodes):
    
    node = Nodes[node_index]
    containedNodes = containsNodes(element[0], element[1], Nodes)
    
    #The general case is false unless proven true
    PointIntersect = False
    for containedNode in containedNodes:
    
        #check if the chosen node is one of the "contained" nodes
        if node_index == containedNode:
            m, b = line(element[0], element[1], Nodes)
            
            # means the line is vertical
            if m == None:
                PointIntersect = True
                # print('line is vertical and node lies on the line')
                break
            # this checks our chosen node lies on the line
            elif abs(node.y - (m*node.x + b)) <= 1e-10:
                PointIntersect = True
                # print('node lies on the line')
                break
            
    return PointIntersect

#checks both the case where op = D and op = T
def ActionCheck(takenAction,Elements,grid,active_nodes, Nodes):
    
    elements_new = Elements.copy()
    # instances of the proposed action
    trial_n = takenAction.n
    trial_e_index = takenAction.e
    trial_e = elements_new[trial_e_index]
    trial_op = takenAction.op
    # print('trial_n = ', trial_n, ', trial_e = ', trial_e, ', trial_op = ', trial_op)

    ActionFail = False
    
    while not ActionFail:
        
        # 1. CHECK THE PROPOSED LINE AND EXISTING LINES DONT INTERSECT
        for element in elements_new:
            #node 1 of the element
            ActionFail = ActionFail + LineIntersectCheck(trial_n, trial_e[0], element[0], element[1], grid)
            #node 2 of the element
            ActionFail = ActionFail + LineIntersectCheck(trial_n, trial_e[1], element[0], element[1], grid)
        # 2. CHECK THE CHOSEN NODE DOESNT LIE ON ANY LINES
        for element in elements_new:
            ActionFail = ActionFail + PointIntersectCheck(trial_n, element, Nodes)
        # 3. CHECK PROPOSED LINE DOESNT PASS OVER ACTIVE NODES
        for node in active_nodes:
            ActionFail = ActionFail + PointIntersectCheck(node, [trial_n, trial_e[0]], Nodes)
            ActionFail = ActionFail + PointIntersectCheck(node, [trial_n, trial_e[1]], Nodes)
        # 4. Check trial node isnt already one of the active nodes
        if trial_n in active_nodes:
            ActionFail += 1
        break
    
    
    # ---- PERFORM EITHER T OR D OPERATION ---- #
    # T operation - same as D operation but we remove an element
    if trial_op == 0:
        
        del elements_new[trial_e_index]
    
    # CREATE A LIST CONTAINING ALL THE CLOSEST ACTIVE NODES (WITHOUT THE ELEMENT NODES)
    third_point_nodes = ThirdPointNodes(trial_e, active_nodes)
    distanceArray = []
    distanceArrayID = []
    for third_point_node in third_point_nodes:
        distance = computeDistance(trial_n, third_point_node, Nodes)
        distanceArray.append(distance)
        distanceArrayID.append(third_point_node)

    distanceArraySorted = np.argsort(distanceArray) 
    
    for index in distanceArraySorted:
        # distanceindex = distanceArraySorted[i]     
        trial_third_point = distanceArrayID[index]
        # print('third point selected is ', trial_third_point)
        
        break_flag = False
        for element in elements_new:
            if LineIntersectCheck(trial_n, trial_third_point, element[0], element[1], grid):
                # print('line check failed for element', element.ID)
                break_flag = True
                break

        if not break_flag:        
            elements_new.append([trial_n, trial_third_point])
            
    
            
    # if ActionFail == False  and thirdpointfound == True:
    if ActionFail == False:
        elements_new.append([trial_n, trial_e[0]])
        elements_new.append([trial_n, trial_e[1]])
        
        # print('ActionCheck passed')
        return True, elements_new
    else:
        # print('ActionCheck failed')
        return False, elements_new



def stiffnessMatrixAssemble(state, Nodes, grid,Area,Emod,density):
    
    active_nodes, _ = ActiveNodes(state, Nodes)

    NNODE = len(active_nodes)

    K = np.zeros((2*NNODE,2*NNODE))
    F = np.zeros((2*NNODE, 1))
    
    dofs = []
    
    for i in range(NNODE):
        for j in range(2):
            dofs.append(int(2*active_nodes[i]+j))
            
    for element in state:
        node1 = Nodes[element[0]]
        node2 = Nodes[element[1]]
        
        length = computeDistance(element[0], element[1], Nodes)
        
        nodal_load = -length*Area*density*9.81/2
        
        active_nodes_index_node1 = active_nodes.index(element[0])
        active_nodes_index_node2 = active_nodes.index(element[1])

        #populate the Force F vector
        F[2*active_nodes_index_node1+1, 0] += nodal_load
        F[2*active_nodes_index_node2+1, 0] += nodal_load

        #populate the local stiffness matrix
        klocal = np.array([[1, -1], [-1, 1 ]])*Area*Emod/length
        ca = (node2.x - node1.x)/length
        sa = (node2.y - node1.y)/length
        rot = np.array([[ca, -sa, 0, 0], [0, 0, ca, -sa]])
        klocal = np.matmul(np.matmul(rot.T,klocal),rot)
        
        
        #populate the GLobal stiffness matrix
        temp = np.zeros((4),dtype=int)

        for idof in range(2):
            temp[idof] = 2*active_nodes.index(element[0])+idof
            temp[2+idof] = 2*active_nodes.index(element[1])+idof

        for i1 in range(4):
            for i2 in range(4):        
                K[temp[i1],temp[i2]] = K[temp[i1],temp[i2]] + klocal[i1,i2]
    
    
    for index, value in enumerate(active_nodes):
    
        if Nodes[value].force_x:
            F[2*index, 0] += Nodes[value].force_x
        
        if Nodes[value].force_y:
            F[2*index+1, 0] += Nodes[value].force_y
    
    
    
    #ID's of the restrained DOFS
    dofs_bf = []
    for index in active_nodes:
    
        if Nodes[index].freeDOF_x == False:
            dofs_bf.append(index*2)
        
        if Nodes[index].freeDOF_y == False:
            dofs_bf.append(index*2 + 1)    
    
    # The indices of the restrained DOFS wrt to the active DOFS
    dofs_delete = []
    for i in dofs_bf:
        index = dofs.index(i)
        dofs_delete.append(index)
    
    K = np.delete(K, dofs_delete, 0)  
    K = np.delete(K, dofs_delete, 1)
    F = np.delete(F, dofs_delete, 0)
    
    dofs_remain = [item for index, item in enumerate(dofs) if index not in dofs_delete]
    
    return K, F, dofs_remain

def displacementSolve(K, F):
    #CONDITIONING CHECK
    eigenvalues = np.linalg.eigvals(K)
    if abs(np.min(eigenvalues)) < 10e-10:
        # print('eigenvalue is too small')
        return False
    if abs(np.max(eigenvalues) / np.min(eigenvalues)) > 10000:
        #ill conditioned stiffness matrix
        # print('Global Stiffness matrix K is ill-conditioned')
        return False
    
    U = np.linalg.solve(K, F)
    
    return U

    
def displacementAtNode(displacement_vector, node, direction, dofs_remain):
    
    dof  = node * 2 + direction
    u_index = dofs_remain.index(dof)

    return float(displacement_vector[u_index])

import matplotlib.pyplot as plt


def plotRewards(rewards_all_episodes):
# Calculate the average of every 10 values
    averages = [np.mean(rewards_all_episodes[i:i+10]) for i in range(0, len(rewards_all_episodes), 10)]
    
    # Generate the indices corresponding to the average values
    indices = range(0, len(rewards_all_episodes), 10)
    
    # Plot average values against indices
    plt.plot(indices, averages)
    
    # Set labels and title
    plt.xlabel('episodes')
    plt.ylabel('Accumulated episodic reward')
    plt.title('Plot of rewards (average of batches of 10) against episodes')
    
    # Display the plot
    plt.show()