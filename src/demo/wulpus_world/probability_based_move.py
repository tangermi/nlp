
#----- IFN680 Assignment 1 -----------------------------------------------#
#  The Wumpus World: a probability based agent
#
#  Implementation of two functions
#   1. PitWumpus_probability_distribution()
#   2. next_room_prob()
#
#    Student 1 no: n10296255
#    Student 1 name: Yongrui Pan
#s
#    Student 2 no: n10057862
#    Student 2 name: Jianwei Tang
#
#-------------------------------------------------------------------------#
from random import *
from AIMA.logic import *
from AIMA.utils import *
from AIMA.probability import *
from tkinter import messagebox
from logic_based_move import next_room

#--------------------------------------------------------------------------------------------------------------
#
#  The following two functions are to be developed by you. They are functions in class Robot. If you need,
#  you can add more functions in this file. In this case, you need to link these functions at the beginning
#  of class Robot in the main program file the_wumpus_world.py.
#
#--------------------------------------------------------------------------------------------------------------
#   Function 1. PitWumpus_probability_distribution(self, width, height)
#
# For this assignment, we treat a pit and the wumpus equally. Each room has two states: 'empty' or 'containing a pit or the wumpus'.
# A Boolean variable to represent each room: 'True' means the room contains a pit/wumpus, 'False' means the room is empty.
#
# For a cave with n columns and m rows, there are totally n*m rooms, i.e., we have n*m Boolean variables to represent the rooms.
# A configuration of pits/wumpus in the cave is an event of these variables.
#
# The function PitWumpus_probability_distribution() below is to construct the joint probability distribution of all possible
# pits/wumpus configurations in a given cave, two parameters
#
# width : the number of columns in the cave
# height: the number of rows in the cave
#
# In this function, you need to create an object of JointProbDist to store the joint probability distribution and  
# return the object. The object will be used by your function next_room_prob() to calculate the required probabilities.
#
# This function will be called in the constructor of class Robot in the main program the_wumpus_world.py to construct the
# joint probability distribution object. Your function next_room_prob() will need to use the joint probability distribution
# to calculate the required conditional probabilities.
#
def PitWumpus_probability_distribution(self, width, height): 
    # Create a list of variable names to represent the rooms. 
    # A string '(i,j)' is used as a variable name to represent a room at (i, j)
    self.PW_variables = [] 
    for column in range(1, width + 1):
        for row in range(1, height + 1):
            self.PW_variables  = self.PW_variables  + ['(%d,%d)'%(column,row)]

    #--------- Add your code here -------------------------------------------------------------------
    
    T, F = True, False
    # assume each room contains a wumpus/pit with probability 0.2
    p_true = 0.2
    p_false = 1 - p_true

    var_values = {each: [T, F] for each in self.PW_variables}
    #print(var_values)
    JPD =JointProbDist(self.PW_variables,var_values)
    events = all_events_jpd(self.PW_variables, JPD, {})
    
    for each_event in events:
        # Calculate the probability for this event
        # if the value of a variable is false, motiply by p_false which is 0.12, otherwise motiply by p_true which is 1-0.12 
        prob = 1 # initial value of the probability
        for (var, val) in each_event.items(): # for each (variable, value) pair in the dictionary
            prob = prob * p_false if val == F else prob * p_true
        # Assign the probability to this event
        JPD[each_event]= prob
    
    return JPD
                
        
#---------------------------------------------------------------------------------------------------
#   Function 2. next_room_prob(self, x, y)
#
#  The parameters, (x, y), are the robot's current position in the cave environment.
#  x: column
#  y: row
#
#  This function returns a room location (column,row) for the robot to go.
#  There are three cases:
#
#    1. Firstly, you can call the function next_room() of the logic-based agent to find a
#       safe room. If there is a safe room, return the location (column,row) of the safe room.
#    2. If there is no safe room, this function needs to choose a room whose probability of containing
#       a pit/wumpus is lower than the pre-specified probability threshold, then return the location of
#       that room.
#    3. If the probabilities of all the surrounding rooms are not lower than the pre-specified probability
#       threshold, return (0,0).
#
def next_room_prob(self, x, y):
    #messagebox.showinfo("Not yet complete", "You need to complete the function next_room_prob.")
    
    #--------- Add your code here -------------------------------------------------------------------
    logic_based_check = next_room(self,x,y)
    
    # check if there is a safe room by logic-based agent
    if  len(self.path)!=0:
        return logic_based_check
    else:
        # get all rooms in the cave
        all_rooms = []
        for column in range(1, self.cave.WIDTH+1):
            for row in range(1, self.cave.HEIGHT+1):
                all_rooms.append((column,row))
        
        # get all unknown rooms by remove all visited rooms
        for each in self.visited_rooms:
            all_rooms.remove(each)
        unknowns = all_rooms
        
        
        query = []
        # get all query rooms which the agent can reach
        # these query rooms are unknown and adjacent to unknown rooms
        for each_visited in self.visited_rooms:
            for each_unknown in unknowns:
                if each_visited[0] == each_unknown[0]:
                    if each_visited[1] == each_unknown[1]+1 or each_visited[1] == each_unknown[1]-1:
                        query.append(each_unknown)
                if each_visited[1] == each_unknown[1]:
                    if each_visited[0] == each_unknown[0]+1 or each_visited[0] == each_unknown[0]-1:
                        query.append(each_unknown)
        
        # remove duplicates 
        query = list(set(query))
        
        # get BS_known
        BS_known = self.observation_breeze_stench(self.visited_rooms)

        # PW variables with known truth values in PW known
        evidence = self.observation_pits(self.visited_rooms)
        
        # each room probability of wumpus/pit
        probs = []
        
        for room in query:
            # initialize probability summation of all events that the query room contains a wumpus
            P_sum = 0
            # initialize probability summation of all events that the query room does NOT contain a wumpus
            P_sum1 = 0
            # get all events with probability in the cave
            events = all_events_jpd(self.PW_variables, self.jdP_PWs, evidence)
            
            for each_event in events:
                # the room contains a wumpus/pit
                if each_event[str(room).replace(' ', '')] == True:
                    P_sum += self.consistent(BS_known,each_event)*self.jdP_PWs[each_event]
                # the room does NOT contains a wumpus/pit
                else:
                    P_sum1 += self.consistent(BS_known,each_event)*self.jdP_PWs[each_event]
                    
            # calculate the probability of pit/wumpus the query room 
            probs.append(P_sum/(P_sum1+P_sum))
        
        # check if the minimum probability of query rooms is lower than the max pit probability
        # if not, game over
        if min(probs) > self.max_pit_probability:
            # if not, game over
            return logic_based_check
        
        # the next rooms will be the one which has lowest probability contains a wumpus/pit in the query rooms
        return query[probs.index(min(probs))]

#---------------------------------------------------------------------------------------------------
 
####################################################################################################
