# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 14:39:27 2023

@author: gabri
"""

from Classes_RL import Action

def allActionsList(maxElements, Nodes):
    AllActions = []
    ID_counter = 0
    for nodes in range(len(Nodes)):
            for element in range(len(maxElements)):
                for operator in range(2):
                    action = Action(nodes, element, operator, ID_counter)
                    AllActions.append(action)
                    ID_counter = ID_counter + 1
    return AllActions