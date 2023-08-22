
"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
import math
from typing import Optional, Union

import numpy as np

import gym
from Methods_RL import ComputeVolume, ActiveNodes, ActionCheck, stiffnessMatrixAssemble, displacementSolve
from Methods_RL import displacementAtNode, PlotElem
from Methods_DQN import allActionsList

class TrussEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, initialState, Nodes, grid, material, spring, render_mode: Optional[str] = None):
        self.initialState = initialState
        self.Nodes = Nodes
        self.grid = grid
        self.material = material
        
        Area = material["Area"]
        Emod = material["E"]
        density = material["Density"]
        
        self.maxStates = 4
        self.maxVolume = 240
        self.maxElements = 14
        
        #this allActionsList is a non-mutable list that contains all the possible actions that could occur
        self.allActionsList = allActionsList(self.maxElements, self.Nodes)
        
        # we need to have a list of all the previous states and displacements
        # allows us to compare results of different states to calculate rewards
        # will be useful in the "variable" implementation
        self.statesList = []
        self.displacementsList = []

        self.currentStateID = 0
        self.nextStateID = None
        self.stateCounter = None
        self.terminated = False
        self.reward = 0
       
        self.statesList[self.currentStateID] = self.initialState = initialState
    
        #Add the initial displacement to the displacementsList
        K, F, dofs_remain           = stiffnessMatrixAssemble(self.statesList[self.currentStateID], Nodes, grid, Area, Emod,  density)
        U                           = displacementSolve(K, F)
        displacementCurrent        = displacementAtNode(U, 14, 0, dofs_remain)

        self.displacementsList.append(displacementCurrent)

        self.Q = []
        self.Q.append([0] * len(self.AllActionsList))
        
        
        self.render_mode = True
        

    def step(self, actionIndex):
        
        err_msg = f"{actionIndex!r} ({type(actionIndex)}) invalid"
        assert self.action_space.contains(actionIndex), err_msg
        assert self.state is not None, "Call reset before using step method."
        
        currentState = self.state

        Area = self.material["Area"]
        Emod = self.material["E"]
        density = self.material["Density"]
        
        takenAction                     = allActionsList[actionIndex]
        active_nodes, inactive_nodes    = ActiveNodes(currentState, self.Nodes)
        result, nextState               = ActionCheck(takenAction, currentState, self.grid, active_nodes, self.Nodes)
        K, F, dofs_remain               = stiffnessMatrixAssemble(nextState, self.Nodes, self.grid, Area, Emod, density)
        U                               = displacementSolve(K, F)
        displacementCurrent             = displacementAtNode(U, 14, 0, dofs_remain)
        
        Volume                          = ComputeVolume(nextState, self.Area, self.Nodes)

        # IF THE FOLLOWING THREE CONDITIONS ARE SATISFIED WE MOVE ONTO THE NEXT STATE
        # - Volume constraints satisfied
        # - active nodes are left
        # - we havent reached the max state
        if (Volume <= self.maxVolume and len(inactive_nodes) > 0) or self.stateCounter < self.maxStates:
            
            # if this configuration already exists
            if nextState in self.statesList:
                self.newStateID         = self.statesList.index(nextState)
                displacementCurrent     = self.displacementsList[self.newStateID]
            else:
                
                self.displacementsList.append(displacementCurrent)
                self.statesList.append(nextState)
                self.Q.append([0] * len(self.AllActionsList))
            
                #Since we are appending the new state - the index corresponds to the final value in the list
                self.newStateID              = len(self.statesList) - 1
            
            
            if displacementCurrent>=0 and abs(displacementCurrent) <= abs(self.displacementsList[0]):                    
                    
                state_displacement_change   = self.displacementsList[self.newStateID] - self.displacementsList[self.currentStateID]
                global_displacement_change  = self.displacementsList[self.newStateID] - self.displacementsList[0]
                
                self.reward                 = min(-state_displacement_change, -global_displacement_change)
                                    
                # self.Q[self.currentStateID][takenAction.ID] = (1 - learning_rate) * Q[currentStateID][takenAction.ID] + learning_rate * (reward + discount_rate * max(Q[newStateID]))
                
                self.currentStateID = self.newStateID
    
                self.stateCounter += 1

            # else:
            #     Q[currentStateID][takenAction.ID] = -10
        
        # terminal state reached
        else:
            self.terminated = True

        if self.render_mode:
            self.render()
            
        
        return self.state, self.reward, self.terminated, False, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        
        #INITIAL STATE ADDED TO THE RESET FUNCTION
        self.state = self.initialState
        
        self.currentStateID = 0
        self.stateCounter = 0
        self.terminated = False
        self.reward = 0

        self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()
        return self.state, {}

    def render(self):
        
        print('------ STATE: ', self.currentStateID, '--------')
        PlotElem(self.statesList[self.currentStateID], self.Nodes, self.grid)
        print('Displacement for state ', self.stateCounter, '(', self.currentStateID, ') = ', self.displacementsList[self.currentStateID])
        # print('reward = ', reward)
        return None

    def close(self):
        return None