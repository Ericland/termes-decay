from numpy.random import default_rng
import numpy as np
import copy

from CRC_Analysis import get_wrongPlacementLoc, model_CAT_idealParallelNxN, read_action_prob


# In[]
class World:
    def __init__(self, robot_num = None,
                 cur_struct = None,
                 goal_struct = None,
                 start = None,
                 docking = None,
                 exit_list = [],
                 robot_map = None,
                 parents_map = None,
                 children_map = None, 
                 cur_step = 0):
        
        '''
        Robot parameters
        '''
        self.robot_num = robot_num
        self.robot_list = []
        self.robot_map = robot_map
        self.robotConfig_dict = {}
        
        '''
        Structure parameters
        
        parents_map tells parents of each location:
            e.g. parents_map[2,2] ---> [{'position': array([1, 2])}, {'position': array([2, 1])}]
            
        children_map tells children and the uniform transition probability of each location:
            e.g. children_map[1,2] ---> [{'position': array([2, 2]), 'prob': 0.5}, 
                                         {'position': array([1, 3]), 'prob': 0.5}]
            
        pathProbMap tells the optimized transition probabilities of each location:
            e.g. pathProbMap[0,0] ---> [[0, 1, 0.5], [1, 0, 0.5]]
            
        pathProbMapType tells the simulator whether "Normal" or "Optimized" transition probabilities will be used.
        
        robot_map records robot status in the structure: 
            e.g. robot_map[1,1] ---> "0" means there is no robot.
            e.g. robot_map[1,1] ---> {'name': xxx, 'height': xxx, 'heading': xxx}
            
        robotConfig_dict tells the position and configuration of each robot:
            e.g. robotConfig_dict[Robot#1] = {'position': xxx, 'heading': xxx, 'height': xxx}
        '''
        self.cur_struct = cur_struct
        self.goal_struct = goal_struct
        self.start_struct = None
        self.start = start
        self.docking = docking
        self.exit_list = exit_list
        self.parents_map = parents_map
        self.children_map = children_map
        self.pathProbMap = None
        self.pathProbMapType = None
        
        '''
        Simulation parameters
        '''
        self.brickPickupNum = 0
        self.brickPlacementNum = 0
        self.cur_step = cur_step
        self.direction_list = {"N": np.array([-1, 0]), 
                               "E": np.array([0, 1]), 
                               "S": np.array([1, 0]), 
                               "W": np.array([0, -1])}
        self.criticalSimInfo_dict = {}
        
        
    # =============================================================================
    #
    # Define construction initialization processes
    # 
    # =============================================================================


    def create_robot(self, robot_num = 1, robot_type = "ideal"):
        '''
        Create robots from 'Robot' class.

        Parameters
        ----------
        robot_num : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        None.

        '''
        self.robot_num = robot_num
        self.robot_list = [None] * robot_num
        robot_name_list = []
        for i in range(len(self.robot_list)):
            robot_name = "Robot#" + str(i+1)
            
            # Create robots based on the given type
            if robot_type == "real":
                self.robot_list[i] = Robot_Real(robot_name)
            elif robot_type == "ideal":
                self.robot_list[i] = Robot(robot_name)
            else:
                self.robot_list[i] = Robot(robot_name)
            
            self.robotConfig_dict[robot_name] = {}
            robot_name_list.append(robot_name)
        
        return robot_name_list


    def update_nav_path(self, parents_map, children_map, pathProbMap, pathProbMapType):
        '''
        Define the parents, the children, and transition probabilities of each location.

        Parameters
        ----------
        parents_map : TYPE
            parents of each location
        children_map : TYPE
            children of each location
        pathProbMap : TYPE
            list of transition probabilities of each location to its children
        pathProbMapType : TYPE
            whether using optimized transition probabilities

        Returns
        -------
        None.

        '''
        self.parents_map = parents_map
        self.children_map = children_map
        self.pathProbMap = pathProbMap
        self.pathProbMapType = pathProbMapType


    def update_goal_struct(self, goal_struct, cur_struct, start, exit_list, docking): 
        '''
        Initialize the structure details.

        Parameters
        ----------
        goal_struct : TYPE
            padded goal structure in array of float16
            e.g. 3x3v1 "D" is the docking location
              ------- y ------>
             | 0 0 0 0 0
             | D 1 1 1 0
             | 0 2 1 1 0  
             x 0 3 2 1 0
             | 0 0 0 0 0
             V
        cur_struct : TYPE
            padded current structure in array of float16
            e.g. 3x3v1 start
              ------- y ------>
             | 0 0 0 0 0
             | D 1 0 0 0
             | 0 0 0 0 0  
             x 0 0 0 0 0
             | 0 0 0 0 0
             V
        start : TYPE
            e.g. [1 1]
        exit_list : list
            e.g. two exits for s3x3v1: [[3 3], [3 2]]
        docking : TYPE
            brick pickup location set at [1 0]

        Returns
        -------
        None.

        '''
        self.goal_struct = goal_struct
        rowNum, colNum = goal_struct.shape

        # Initiate current structure
        self.cur_struct = cur_struct # with padding
        self.start = start# with padding
        self.exit_list = exit_list
        self.docking = docking # Assume docking is always on the west of the start

        # Initiate robot_map which tells the location of each robot
        self.robot_map = np.zeros((rowNum, colNum), dtype = object) # with padding
        
        
    def update_timeStep(self):
        '''
        Update the time step counter.

        Returns
        -------
        None.

        '''
        self.cur_step += 1
        
        
    # =============================================================================
    #
    # Define functions for checking structure or location status
    # 
    # =============================================================================
    
    
    def check_consProgress(self): 
        '''
        Check the construction progress. 

        Returns
        -------
        None.

        '''
        # Find number of bricks in docking location and remove it in calculation
        xd, yd = self.docking
        brickNumDocking = self.start_struct[xd, yd]
        
        # Compute the effective construction progress
        goalPLB_loc_list = np.transpose(np.nonzero(self.goal_struct)).tolist()
        goalTotalBrickNum = np.sum(self.goal_struct)
        correctBrickNum = 0
        for loc in goalPLB_loc_list:
            goalBrickNum = self.goal_struct[tuple(loc)]
            curBrickNum = self.cur_struct[tuple(loc)]
            if curBrickNum <= goalBrickNum:
                correctBrickNum += curBrickNum
            else:
                correctBrickNum += goalBrickNum
        consProgressEffective = (correctBrickNum - brickNumDocking) / (goalTotalBrickNum - brickNumDocking)
        
        # Compute the construction progress
        curTotalBrickNum = np.sum(self.cur_struct)
        consProgress = (curTotalBrickNum - brickNumDocking) / (goalTotalBrickNum - brickNumDocking)
        
        # return cons. prog. info
        consProgressInfo = {'consProgressEffective': consProgressEffective, 
                            'consProgress': consProgress}
        return(consProgressInfo)
    
    
    def check_start_safety(self):
        '''
        Check if the start location is safe for robot initialization.
        Checked locations are ("S" is start, "C" is checked location):
            S C C
            C C
            C

        Returns
        -------
            True: Safe
            False: Unsafe

        '''        
        check_loc_list = []
        check_loc_list.append(np.copy(self.start))
        check_loc_list.append(self.start + self.direction_list["E"])
        check_loc_list.append(self.start + 2 * self.direction_list["E"])
        check_loc_list.append(self.start + self.direction_list["S"])
        check_loc_list.append(self.start + 2 * self.direction_list["S"])
        check_loc_list.append(self.start + self.direction_list["E"] + self.direction_list["S"])
        
        start_safety = True
        for check_loc in check_loc_list:
            xc, yc = check_loc
            if self.robot_map[xc, yc] != 0:
                start_safety = False
                break
        
        return(start_safety)
        
        
    # =============================================================================
    #
    # Define functions for accessing structure information
    # 
    # =============================================================================
        
        
    def _get_heightInfo(self, loc):
        '''
        Get height information of a given location

        Parameters
        ----------
        loc : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        rowNum, colNum = self.goal_struct.shape # Get padded map dimension
        x, y = loc # Get location coordinates
        cur_height = 0 # current height of the given location
        goal_height = 0 # desired height of the given location
        
        # Check if location is within the map
        if (x >= 0 and x <= (rowNum - 1) and y >= 0 and y <= (colNum - 1)):
            loc_in_map = True
        else:
            loc_in_map = False
        
        # If location is within the map, get the height information from cur_struct and goal_struct
        if loc_in_map: 
            cur_height = self.cur_struct[x, y]
            goal_height = self.goal_struct[x, y]
        else: 
            pass
        
        # Store height info
        heightInfo = {}
        heightInfo['cur_height'] = cur_height
        heightInfo['goal_height'] = goal_height
            
        return(heightInfo)
    
    
    def _get_locInfo(self, loc):
        '''
        Get information of a given location including: 
            Whether the given location is within the map 
            Whether the given location is part of the structure
            Whether the given location is on the accessible margin
            Whether there is a robot at the given location

        Parameters
        ----------
        loc : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        rowNum, colNum = self.goal_struct.shape # Get padded map dimension
        x, y = loc # Get location coordinates
        heightInfo = self._get_heightInfo(loc) # Get height info
        
        # Check if location is within the map
        if (x >= 0 and x <= (rowNum - 1) and y >= 0 and y <= (colNum - 1)):
            loc_in_map = True
        else:
            loc_in_map = False
            
        # Check if location is part of the structure
        if heightInfo['goal_height'] != 0: 
            loc_of_struct = True
        else:
            loc_of_struct = False
            
        # Check if location is on the accessible margin
        # The accessible margin of the structure is defined as any location that: 
        #     1) has current height of 1
        #     2) has at least one neighbor location that has goal height of 0 and current height of 0
        loc_in_accessMargin = False
        if heightInfo['cur_height'] == 1: 
            cond2 = False # whether the 2nd condition is satisfied
            for direction_vector in self.direction_list.values(): 
                loc_sur = loc + direction_vector
                heightInfo_sur = self._get_heightInfo(loc_sur)
                if heightInfo_sur['cur_height'] == 0 and heightInfo_sur['goal_height'] == 0: 
                    cond2 = True
                    break
            if cond2: 
                loc_in_accessMargin = True
                
        # Check if the location is occupied by a robot
        # By default, no robot can exists outside the map
        loc_has_robot = False
        loc_robotName = None
        if loc_in_map: 
            if self.robot_map[x, y] != 0: 
                loc_has_robot = True
                loc_robotName = self.robot_map[x, y]['name']
                
        # Store location info
        locInfo = {}
        locInfo['loc_in_map'] = loc_in_map # Whether the given location is within the map 
        locInfo['loc_of_struct'] = loc_of_struct # Whether the given location is part of the structure
        locInfo['loc_in_accessMargin'] = loc_in_accessMargin # Whether the given location is on the accessible margin
        locInfo['loc_has_robot'] = loc_has_robot # Whether there is a robot at the given location
        locInfo['loc_robotName'] = loc_robotName # Name of the robot that occupies the given location
        
        return(locInfo)
    
    
    # =============================================================================
    #
    # Define functions for structure modification
    # 
    # =============================================================================
    
    
    def add_bricks(self, loc, brickNum = 1): 
        '''
        Virtually add brick(s) to a given location. 
        By default, 1 brick will be added.

        Parameters
        ----------
        loc : TYPE
            DESCRIPTION.
        brickNum : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        None.

        '''
        x, y = loc # Get location coordinates
        locInfo = self._get_locInfo(loc) # Get location information
        
        # Only add the brick when the given location is in the map
        if locInfo['loc_in_map']: 
            self.cur_struct[x, y] += brickNum
            
            # log the structure modification
            senderName = None
            eventMsg = 'A brick was added!'
            eventInfo = {'eventLoc': {'loc': np.copy(loc), 'height': self.cur_struct[x, y] - 1}}
            self.log_criticalSimInfo(senderName, eventMsg, eventInfo)
            
        else: 
            raise Exception("Cannot add the brick!")
            
            
    def remove_bricks(self, loc, brickNum = 1): 
        '''
        Virtually remove brick(s) from a given location. 
        By default, 1 brick will be removed.

        Parameters
        ----------
        loc : TYPE
            DESCRIPTION.
        brickNum : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        None.

        '''
        x, y = loc # Get location coordinates
        locInfo = self._get_locInfo(loc) # Get location information
        heightInfo = self._get_heightInfo(loc) # Get height information
        
        # Only remove the brick when the given location is in the map and its height > 0
        if locInfo['loc_in_map'] and heightInfo['cur_height'] > 0: 
            self.cur_struct[x, y] -= brickNum
            
            # log the structure modification
            senderName = None
            eventMsg = 'A brick was removed!'
            eventInfo = {'eventLoc': {'loc': np.copy(loc), 'height': self.cur_struct[x, y] + 1}}
            self.log_criticalSimInfo(senderName, eventMsg, eventInfo)
            
        else:
            raise Exception("Cannot remove the brick!")
            
            
    # =============================================================================
    #
    # Define functions for logging simulation information
    # 
    # =============================================================================
    
    
    def log_criticalSimInfo(self, senderName, msg, msgInfo = None): 
        '''
        This function extracts and logs critical information from a message sent from the robot.
        
        CAUTION: 
            1) If structure modification is involved in the event, the registered cood should be pre-modification cood.

        Parameters
        ----------
        senderName : TYPE
            DESCRIPTION.
        msg : TYPE
            DESCRIPTION.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        def add_criticalSimInfo(criticalSimInfo): 
            '''
            Add critical simulation information to self.criticalSimInfo_dict.
            Events are organized by the event type.
            '''
            if criticalSimInfo['eventType'] not in self.criticalSimInfo_dict: 
                self.criticalSimInfo_dict[criticalSimInfo['eventType']] = [criticalSimInfo]
            else:
                self.criticalSimInfo_dict[criticalSimInfo['eventType']].append(criticalSimInfo)
                
                
        def add_criticalSimInfo_lastingEvent(criticalSimInfo): 
            '''
            Add critical simulation information of lasting events to self.criticalSimInfo_dict.
            Events are organized by the event type. 
            Inside each lasting event class, events are further organized by senderName. 
            E.g.: criticalSimInfo['eventType']['senderName'] = [event1, event2, ...]
            '''
            eventType = criticalSimInfo['eventType']
            senderName = criticalSimInfo['senderName']
            eventTime = criticalSimInfo['eventTime']
            if eventType not in self.criticalSimInfo_dict: 
                self.criticalSimInfo_dict[eventType] = {}
                self.criticalSimInfo_dict[eventType][senderName] = [criticalSimInfo]
            else: 
                if senderName not in self.criticalSimInfo_dict[eventType]: 
                    self.criticalSimInfo_dict[eventType][senderName] = [criticalSimInfo]
                else: 
                    eventEndTime = self.criticalSimInfo_dict[eventType][senderName][-1]['eventInfo']['eventEndTime']
                    if eventTime == eventEndTime + 1: 
                        self.criticalSimInfo_dict[eventType][senderName][-1]['eventInfo']['eventEndTime'] = eventTime  
                    else: 
                        self.criticalSimInfo_dict[eventType][senderName].append(criticalSimInfo)
            
            
        ''' #################### Start logging #################### '''
        
        # Check if the input is not string
        if type(msg) != str:
            raise Exception("Invalid input type! (log_criticalSimInfo)")
            
        # Store the event information
        criticalSimInfo = {'senderName': senderName, 
                           'eventType': None, 
                           'eventTime': None, 
                           'eventLoc': {'loc': None, 'height': None}, 
                           'eventInfo': None}
        
        
        ''' #################### Log momentary events #################### '''
        
        ''' log robot deployment '''
        if 'is deployed to the structure' in msg: 
            criticalSimInfo['eventType'] = 'robot deployment'
            criticalSimInfo['eventTime'] = self.cur_step
            add_criticalSimInfo(criticalSimInfo)
        
        ''' log brick addition '''
        if 'A brick was added' in msg: 
            criticalSimInfo['eventType'] = 'brick addition'
            criticalSimInfo['eventTime'] = self.cur_step
            criticalSimInfo['eventLoc']['loc'] = msgInfo['eventLoc']['loc']
            criticalSimInfo['eventLoc']['height'] = msgInfo['eventLoc']['height']
            add_criticalSimInfo(criticalSimInfo)
            
        ''' log brick removal '''
        if 'A brick was removed' in msg: 
            criticalSimInfo['eventType'] = 'brick removal'
            criticalSimInfo['eventTime'] = self.cur_step
            criticalSimInfo['eventLoc']['loc'] = msgInfo['eventLoc']['loc']
            criticalSimInfo['eventLoc']['height'] = msgInfo['eventLoc']['height']
            add_criticalSimInfo(criticalSimInfo)
            
        ''' log wrong placement errors '''
        if 'Wrong placement' in msg: 
            wrongPlacementLoc, wrongPlacementHeight = get_wrongPlacementLoc(msg)
            criticalSimInfo['eventType'] = 'wrong placement'
            criticalSimInfo['eventTime'] = self.cur_step
            criticalSimInfo['eventLoc']['loc'] = wrongPlacementLoc
            criticalSimInfo['eventLoc']['height'] = wrongPlacementHeight
            criticalSimInfo['eventInfo'] = {'error was removed by decay': False}
            if 'Wrong placement on structure' in msg: 
                criticalSimInfo['eventInfo']['error is critical'] = True
            else: 
                criticalSimInfo['eventInfo']['error is critical'] = False
            add_criticalSimInfo(criticalSimInfo)
            
        ''' log action error detection '''
        if 'Error is detected with action' in msg: 
            criticalSimInfo['eventType'] = 'action error detection'
            criticalSimInfo['eventTime'] = self.cur_step
            eventInfo = copy.deepcopy(msgInfo)
            eventInfo.pop('eventLoc')
            criticalSimInfo['eventInfo'] = eventInfo
            criticalSimInfo['eventLoc']['loc'] = msgInfo['eventLoc']['loc']
            criticalSimInfo['eventLoc']['height'] = msgInfo['eventLoc']['height']
            add_criticalSimInfo(criticalSimInfo)
            
        ''' log action errors '''
        if 'Error occurs with action' in msg: 
            criticalSimInfo['eventType'] = 'action error'
            criticalSimInfo['eventTime'] = self.cur_step
            eventInfo = copy.deepcopy(msgInfo)
            eventInfo.pop('eventLoc')
            criticalSimInfo['eventInfo'] = eventInfo
            criticalSimInfo['eventLoc']['loc'] = msgInfo['eventLoc']['loc']
            criticalSimInfo['eventLoc']['height'] = msgInfo['eventLoc']['height']
            add_criticalSimInfo(criticalSimInfo)
            
        ''' log belief correction '''
        if 'Heading correction occurs' in msg: 
            criticalSimInfo['eventType'] = 'belief correction'
            criticalSimInfo['eventTime'] = self.cur_step
            criticalSimInfo['eventInfo'] = {'msg': msg, 'beliefCorrectionSuccess': msgInfo['beliefCorrectionSuccess']}
            criticalSimInfo['eventLoc']['loc'] = msgInfo['eventLoc']['loc']
            criticalSimInfo['eventLoc']['height'] = msgInfo['eventLoc']['height']
            add_criticalSimInfo(criticalSimInfo)
            
        ''' log decay events '''
        if 'A decay event occurred' in msg: 
            criticalSimInfo['eventType'] = 'decay'
            criticalSimInfo['eventTime'] = self.cur_step
            criticalSimInfo['eventLoc']['loc'] = msgInfo['eventLoc']['loc']
            criticalSimInfo['eventLoc']['height'] = msgInfo['eventLoc']['height']
            eventInfo = copy.deepcopy(msgInfo)
            eventInfo.pop('eventLoc')
            criticalSimInfo['eventInfo'] = eventInfo
            add_criticalSimInfo(criticalSimInfo)
            
        ''' log brick placement abandonment'''
        if 'Brick placement is abandoned' in msg: 
            criticalSimInfo['eventType'] = 'brick placement abandonment'
            criticalSimInfo['eventTime'] = self.cur_step
            criticalSimInfo['eventLoc']['loc'] = msgInfo['eventLoc']['loc']
            criticalSimInfo['eventLoc']['height'] = msgInfo['eventLoc']['height']
            criticalSimInfo['eventInfo'] = msg
            add_criticalSimInfo(criticalSimInfo)
            
            
        ''' #################### Log lasting events #################### '''
        
        ''' log deadlock '''
        if 'Deadlock' in msg: 
            criticalSimInfo['eventType'] = 'deadlock'
            criticalSimInfo['eventTime'] = self.cur_step
            criticalSimInfo['eventInfo'] = {'eventEndTime': self.cur_step, 
                                            'error was removed by decay': False}
            criticalSimInfo['eventLoc']['loc'] = msgInfo['eventLoc']['loc']
            criticalSimInfo['eventLoc']['height'] = msgInfo['eventLoc']['height']
            add_criticalSimInfo_lastingEvent(criticalSimInfo)
        
        ''' log failure to leave structure '''
        if 'Robot tries to leave but leaving conditions are not met' in msg: 
            criticalSimInfo['eventType'] = 'failure to leave structure'
            criticalSimInfo['eventTime'] = self.cur_step
            criticalSimInfo['eventInfo'] = {'eventEndTime': self.cur_step}
            criticalSimInfo['eventLoc']['loc'] = msgInfo['eventLoc']['loc']
            criticalSimInfo['eventLoc']['height'] = msgInfo['eventLoc']['height']
            add_criticalSimInfo_lastingEvent(criticalSimInfo)
            
        ''' log waiting due to unreachable front location '''
        if 'Robot waits since front location is not reachable' in msg: 
            criticalSimInfo['eventType'] = 'unreachable front location'
            criticalSimInfo['eventTime'] = self.cur_step
            criticalSimInfo['eventInfo'] = {'eventEndTime': self.cur_step}
            criticalSimInfo['eventLoc']['loc'] = msgInfo['eventLoc']['loc']
            criticalSimInfo['eventLoc']['height'] = msgInfo['eventLoc']['height']
            add_criticalSimInfo_lastingEvent(criticalSimInfo)
            
        return(criticalSimInfo)
        
        
        
# In[]

class Eroder_Probabilistic: 
    '''
    An agent who erodes bricks or robots away with certain probability
    '''
    
    def __init__(self, TERMES_World, eroder_name): 
        '''
        Eroder feature parameters
        '''
        self.name = eroder_name
        self.type = "probabilistic"
        
        '''
        Erosion parameters
        '''
        # Use uniform erosion bias map by default
        self.erosionBiasProbMap = np.ones(TERMES_World.goal_struct.shape) / TERMES_World.goal_struct.size
        
        # Construction rate. Use an NxN structure to get an initial guess
        structDim = max(TERMES_World.goal_struct.shape) - 2
        self.consRate = (structDim ** 2 - 1) / model_CAT_idealParallelNxN(structDim)[0]
        
        # Erosion speed factor controls the ratio of erosion rate to construction rate. Default is 0.1.
        self.erosionSpeedFactor = 0.1
        
        # Success probability at each location, use default erosionBiasProbMap and consRate to get an initial value
        self.successProbMap = None
        self._update_successProbMap()
        
        '''
        Variables for logging erosion actions
        '''
        # Erosion history
        self.erosionTrialNum = 0
        
        '''
        Other simulation variables.
        '''
        # initialize random number generator for each robot
        self.rng = default_rng()
        
        
    def _update_successProbMap(self): 
        '''
        Update the success probability at each location.

        Returns
        -------
        None.

        '''
        ''' Set up erosion parameters '''
        erosionBiasProbMap = np.copy(self.erosionBiasProbMap)
        locNum = erosionBiasProbMap.size
        ratioEC = self.erosionSpeedFactor # ratio of erosion rate to construction rate
        consRate = self.consRate
        
        ''' Find the maximum success probability '''
        piMax1 = 1 - (1 - ratioEC * consRate) ** (1 / locNum) # upperbound 1
        piMax2 = ratioEC * consRate / locNum # upperbound 2
        piMax = min(piMax1, piMax2) # choose the min. of upperbounds
        
        ''' Compute the success probability at each location '''
        self.successProbMap = erosionBiasProbMap / np.mean(erosionBiasProbMap) * piMax
        
        
    def update_erosionParameter(self, erosionParameter_dict): 
        '''
        This is a general function for updating the erosion parameters.

        Parameters
        ----------
        erosionProbMap : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        # Update the erosionBiasProbMap
        if 'erosionBiasProbMap' in erosionParameter_dict: 
            self.erosionBiasProbMap = np.copy(erosionParameter_dict['erosionBiasProbMap'])
            
        # Update the construction rate
        if 'consRate' in erosionParameter_dict: 
            self.consRate = erosionParameter_dict['consRate']
            
        # Update the erosion speed factor
        if 'erosionSpeedFactor' in erosionParameter_dict: 
            self.erosionSpeedFactor = erosionParameter_dict['erosionSpeedFactor']
            
        # Update the success probability map
        self._update_successProbMap()
        
        
    def _choose_erosionLoc(self): 
        '''
        Choose the erosion location based on the erosion probability map.

        Returns
        -------
        None.

        '''        
        ''' Set up the success probability and location list '''
        successProbMap = np.copy(self.successProbMap)
        pi_list = successProbMap[np.nonzero(successProbMap)].tolist()
        loc_list = []
        locList_list = np.transpose(np.nonzero(successProbMap)).tolist()
        for locList in locList_list:
            loc_list.append(np.array([locList[0], locList[1]]))
        
        ''' Perform Bernoulli trial at each location '''
        erosionLoc_list = []
        for ii, pi in enumerate(pi_list): 
            tried_loc = loc_list[ii]
            rngOutcome = self.rng.choice(2, p=[1 - pi, pi])
            if rngOutcome == 1:
                erosionLoc_list.append(tried_loc)
        
        return(erosionLoc_list)
    
    
    def show_erosionProperty(self): 
        '''
        Show the probalistic properties of the eroder

        Returns
        -------
        None.

        '''
        ''' Show the probability of erosion '''
        successProbMap = np.copy(self.successProbMap)
        pi_list = successProbMap[np.nonzero(successProbMap)].tolist()
        q_ = 1
        for pi in pi_list: 
            q_ *= (1 - pi)
        q = 1 - q_
        
        ''' Find the conditional expectation '''
        # Given that at least one erosion happens, what is the expected number of erosions? 
        condExpect = np.sum(successProbMap) / q
        
        ''' Find the average erosion rate '''
        trialNum = 1000
        erosionNum_list = []
        for tt in range(trialNum): 
            erosionNum = 0
            for pi in pi_list: 
                rngOutcome = self.rng.choice(2, p=[1 - pi, pi])
                if rngOutcome == 1:
                    erosionNum += 1
            erosionNum_list.append(erosionNum)
        erosionNumAvg = np.mean(erosionNum_list)
        erosionNumStd = np.std(erosionNum_list)
        
        ''' Collect property information '''
        erosionProperty_dict = {'erosionProb': q, 
                                'condExpect': condExpect, 
                                'erosionNumAvg': erosionNumAvg, 
                                'erosionNumStd': erosionNumStd}
        
        return(erosionProperty_dict)
    
    
    def _erode_brick(self, TERMES_World, erosionLoc): 
        '''
        Erode one brick at the given location.

        Parameters
        ----------
        TERMES_World : TYPE
            DESCRIPTION.
        erosionLoc : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        # Get erosion location coordinate
        xer, yer = erosionLoc
        erosion_heightInfo = TERMES_World._get_heightInfo(erosionLoc)
        erosion_locInfo = TERMES_World._get_locInfo(erosionLoc)
        
        # If it is not physically possible to erode the location, raise error
        if erosion_locInfo['loc_has_robot']: 
            raise Exception("Cannot erode given location since there is a robot there!")
        if erosion_heightInfo['cur_height'] == 0: 
            raise Exception("Cannot erode given location since there is no brick there!")
        if np.array_equal(erosionLoc, TERMES_World.docking): 
            raise Exception("Cannot erode the docking location!")
            
        # Check if the brick decay is effective
        # There are two effectiveness in terms of brick decay: 
        #   1) critically effective: decay removes a critical WP error
        #   2) noncritically effective: decay removes a noncritical error
        effectiveness = 'not effective'
        cood_decay = np.array([xer, yer, erosion_heightInfo['cur_height'] - 1]).astype(int) # cood of decay location after decay
        if 'wrong placement' in TERMES_World.criticalSimInfo_dict: 
            for WPErrorInfo in TERMES_World.criticalSimInfo_dict['wrong placement']: 
                WP_is_removed = WPErrorInfo['eventInfo']['error was removed by decay']
                if not WP_is_removed: 
                    xWP, yWP = WPErrorInfo['eventLoc']['loc']
                    zWP = WPErrorInfo['eventLoc']['height']
                    cood_WP = np.array([xWP, yWP, zWP]).astype(int)
                    if np.array_equal(cood_decay, cood_WP): 
                        WPErrorInfo['eventInfo']['error was removed by decay'] = True
                        WPErrorInfo['eventInfo']['time when error was removed'] = TERMES_World.cur_step
                        if WPErrorInfo['eventInfo']['error is critical']: 
                            effectiveness = 'critically effective'
                        else: 
                            effectiveness = 'noncritically effective'
        
        # Log erosion information
        senderName = self.name
        eventMsg = 'A decay event occurred!'
        eventInfo = {'eventLoc': {'loc': erosionLoc, 'height': erosion_heightInfo['cur_height']}, 
                     'effectiveness': effectiveness, 
                     'objectClass': 'brick', 
                     'objectInfo': None}
        TERMES_World.log_criticalSimInfo(senderName, eventMsg, eventInfo)
        
        # Erode the brick away
        TERMES_World.remove_bricks(erosionLoc)
        
        
    def _erode_robot(self, TERMES_World, erosionLoc): 
        '''
        Erode the robot at the given location

        Parameters
        ----------
        TERMES_World : TYPE
            DESCRIPTION.
        erosionLoc : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        # Get erosion location coordinate
        xer, yer = erosionLoc
        erosion_heightInfo = TERMES_World._get_heightInfo(erosionLoc)
        erosion_locInfo = TERMES_World._get_locInfo(erosionLoc)
        
        # If it is not physically possible to erode the location, raise error
        if not erosion_locInfo['loc_has_robot']: 
            raise Exception("Cannot erode given location since there is no robot there!")
        
        # Find the robot to be eroded away
        robotName = erosion_locInfo['loc_robotName']
        robotInd = int(robotName[robotName.find("#")+1 : ]) - 1
        erodedRobot = TERMES_World.robot_list[robotInd]
        
        # Check if the decay is effective
        # A robot decay is effective only if the robot removed is in deadlock
        effectiveness = 'not effective'
        if 'deadlock' in TERMES_World.criticalSimInfo_dict: 
            if robotName in TERMES_World.criticalSimInfo_dict['deadlock']: 
                RSErrorInfo = TERMES_World.criticalSimInfo_dict['deadlock'][robotName][-1]
                RS_is_removed = RSErrorInfo['eventInfo']['error was removed by decay']
                if not RS_is_removed: 
                    cur_time = TERMES_World.cur_step
                    eventEndTime = RSErrorInfo['eventInfo']['eventEndTime']
                    if eventEndTime == cur_time: 
                        RSErrorInfo['eventInfo']['error was removed by decay'] = True
                        RSErrorInfo['eventInfo']['time when error was removed'] = cur_time
                        effectiveness = 'effective'
        
        # Log erosion information
        senderName = self.name
        eventMsg = 'A decay event occurred!'
        eventInfo = {'eventLoc': {'loc': erosionLoc, 'height': erosion_heightInfo['cur_height']}, 
                     'effectiveness': effectiveness, 
                     'objectClass': 'robot', 
                     'objectInfo': {'name': robotName}}
        TERMES_World.log_criticalSimInfo(senderName, eventMsg, eventInfo)
        
        # Erode the robot by removing it
        erodedRobot._reset_robotInWorld(TERMES_World)
        
        
    def _erode_nothing(self, TERMES_World, erosionLoc): 
        '''
        This function only records the erosion history. No erosion is performed.

        Parameters
        ----------
        TERMES_World : TYPE
            DESCRIPTION.
        erosionLoc : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        # Get erosion location coordinate
        xer, yer = erosionLoc
        erosion_heightInfo = TERMES_World._get_heightInfo(erosionLoc)
        
        # Log erosion information
        senderName = self.name
        eventMsg = 'A decay event occurred!'
        eventInfo = {'eventLoc': {'loc': erosionLoc, 'height': erosion_heightInfo['cur_height']}, 
                     'effectiveness': None, 
                     'objectClass': None, 
                     'objectInfo': None}
        TERMES_World.log_criticalSimInfo(senderName, eventMsg, eventInfo)
        
        
    def _check_erosionCondition(self, TERMES_World, erosionLoc): 
        '''
        Check whether the current location can be eroded.

        Parameters
        ----------
        TERMES_World : TYPE
            DESCRIPTION.
        erosionLoc : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        # Get erosion locatin coordinates
        xer, yer = erosionLoc
        erosion_heightInfo = TERMES_World._get_heightInfo(erosionLoc)
        erosion_locInfo = TERMES_World._get_locInfo(erosionLoc)
        canErodeBrick = False
        canErodeRobot = False
        
        '''
        If there is a robot at the erosion location, one can erode the robot away
        '''
        if erosion_locInfo['loc_has_robot']: 
            canErodeRobot = True
            
        '''
        If there is no robot, the location has height > 0, and the erosion location is not the docking location
        further check the brick-erosion rules.
        '''
        if (not erosion_locInfo['loc_has_robot'] and erosion_heightInfo['cur_height'] > 0 
            and not np.array_equal(erosionLoc, TERMES_World.docking)): 
            canErodeBrick = True
            
        return(canErodeBrick, canErodeRobot)
    
        
        
    def erode(self, TERMES_World, **kwargs): 
        '''
        Perform probabilistic erosion

        Parameters
        ----------
        TERMES_World : TYPE
            DESCRIPTION.
        erosionObj : TYPE, optional
            DESCRIPTION. Options for choosing specific types of erosion object:
                both: bricks and robots
                brick: bricks only
                robot: robots only

        Returns
        -------
        None.

        '''
        # Get erosion setting parameters
        if 'erosionObj' in kwargs: 
            erosionObj = kwargs['erosionObj']
        else:
            erosionObj = 'both'
        
        # Generate list of locations to be eroded
        erosionLoc_list = self._choose_erosionLoc()
        
        # If the erosion location list is not empty, further check each location to decide erosion
        if len(erosionLoc_list) != 0: 
            self.erosionTrialNum += 1
            for erosionLoc in erosionLoc_list: 
                # Check erosion conditions
                canErodeBrick, canErodeRobot = self._check_erosionCondition(TERMES_World, erosionLoc)
                
                if erosionObj == 'both': 
                    if canErodeRobot: 
                        self._erode_robot(TERMES_World, erosionLoc)
                    elif canErodeBrick: 
                        self._erode_brick(TERMES_World, erosionLoc)
                    else: 
                        self._erode_nothing(TERMES_World, erosionLoc)
                        
                elif erosionObj == 'brick': 
                    if canErodeBrick: 
                        self._erode_brick(TERMES_World, erosionLoc)
                    else: 
                        self._erode_nothing(TERMES_World, erosionLoc)
                        
                elif erosionObj == 'robot': 
                    if canErodeRobot: 
                        self._erode_robot(TERMES_World, erosionLoc)
                    else: 
                        self._erode_nothing(TERMES_World, erosionLoc)
                
        return(erosionLoc_list)
    
    
    
class Eroder_Restorative(Eroder_Probabilistic): 
    '''
    Restorative eroder which eliminates an error immediately after its occurence.
    A subclass of Eroder_Probabilistic
    '''

    def __init__(self, TERMES_World, eroder_name): 
        '''
        Call super class constructors.
        '''
        Eroder_Probabilistic.__init__(self, TERMES_World, eroder_name)
        
        '''
        Change robot feature parameters
        '''
        self.type = "restorative"
        
        '''
        Variables for logging erosion actions
        '''
        # Record the errors that have been eliminated
        self.correctionHistory = []
        
        
    def erode(self, TERMES_World, **kwargs): 
        '''
        Perform ideal erosion. 
        When an erosion location is chosen, this function randomly eliminates a critical error (if there is any).
        This is a conceptual function and should not be used for modeling real erosion.

        Parameters
        ----------
        TERMES_World : TYPE
            DESCRIPTION.
        erosionObj : TYPE, optional
            DESCRIPTION. Options for choosing specific types of erosion object:
                both: bricks and robots
                brick: bricks only
                robot: robots only

        Returns
        -------
        None.

        '''
        ''' Get erosion setting parameters '''
        if 'erosionObj' in kwargs: 
            erosionObj = kwargs['erosionObj']
        else:
            erosionObj = 'both'
        
        ''' Collect error information '''
        # Collect wrong placement error information
        WPErrorInfo_list = []
        if 'wrong placement' in TERMES_World.criticalSimInfo_dict: 
            WPEventInfo_list = TERMES_World.criticalSimInfo_dict['wrong placement']
            for WPEventInfo in WPEventInfo_list: 
                WPErrorInfo_list.append({'errorType': 'WP', 
                                         'errorTime': WPEventInfo['eventTime'], 
                                         'errorLoc': WPEventInfo['eventLoc']['loc'], 
                                         'robotName': WPEventInfo['senderName']})
        # Remove wrong placement errors that have already been eliminated
        WPErrorInfo_list_filtered = []
        if len(WPErrorInfo_list) != 0: 
            for WPErrorInfo in WPErrorInfo_list: 
                repetition = False
                if len(self.correctionHistory) == 0: 
                    pass
                else: 
                    for correctionInfo in self.correctionHistory: 
                        if (WPErrorInfo['errorType'] == correctionInfo['errorType']
                            and WPErrorInfo['errorTime'] == correctionInfo['errorTime'] 
                            and WPErrorInfo['robotName'] == correctionInfo['robotName'] 
                            and np.array_equal(WPErrorInfo['errorLoc'], correctionInfo['errorLoc'])): 
                            repetition = True
                            break
                if not repetition: 
                    WPErrorInfo_list_filtered.append(copy.deepcopy(WPErrorInfo))
                        
        # Collect robot stalling error informatin
        RSErrorInfo_list = []
        deadlockPoint = 36 + 1
        for robot in TERMES_World.robot_list: 
            if robot.waitTimer >= deadlockPoint: 
                errorTime = TERMES_World.cur_step
                errorLoc = np.copy(TERMES_World.robotConfig_dict[robot.name]['position'])
                robotName = robot.name
                RSErrorInfo_list.append({'errorType': 'RS', 
                                         'errorTime': errorTime, 
                                         'errorLoc': errorLoc, 
                                         'robotName': robotName})
                
        # Make a list of critical error information
        criticalErrorInfo_list = WPErrorInfo_list_filtered + RSErrorInfo_list
        
        ''' Eliminate errors '''
        errorNum = len(criticalErrorInfo_list)
        correctedError_list = []
        if errorNum > 0: 
            self.erosionTrialNum += 1
                
            # Eliminate chosen errors
            for chosenError in criticalErrorInfo_list: 
                errorType = chosenError['errorType']
                errorLoc = chosenError['errorLoc']
                # Check erosion conditions
                canErodeBrick, canErodeRobot = self._check_erosionCondition(TERMES_World, errorLoc)
                
                # If chosen error is wrong placement, remove the top brick
                if errorType == 'WP' and (erosionObj == 'both' or erosionObj == 'brick'): 
                    if canErodeBrick: 
                        self._erode_brick(TERMES_World, errorLoc)
                        correctedError_list.append(chosenError)
                    else: 
                        self._erode_nothing(TERMES_World, errorLoc)
                        
                # If chosen error is robot stalling, remove the robot
                if errorType == 'RS' and (erosionObj == 'both' or erosionObj == 'robot'): 
                    if canErodeRobot: 
                        self._erode_robot(TERMES_World, errorLoc)
                        correctedError_list.append(chosenError)
                    else:
                        self._erode_nothing(TERMES_World, errorLoc)
                    
        # Record the corrected errors
        self.correctionHistory += correctedError_list
        
        return(correctedError_list)
    
    
    
class Eroder_Smart(Eroder_Probabilistic): 
    '''
    A probabilistic eroder with erosion rules based on local height difference. 
    The eroder only removes the brick when the removal will not cause any gap or cliff. 
    '''
    
    def __init__(self, TERMES_World, eroder_name): 
        '''
        Call super class constructors.
        '''
        Eroder_Probabilistic.__init__(self, TERMES_World, eroder_name)
        
        '''
        Change robot feature parameters
        '''
        self.type = "smart"
        
        '''
        Initialize a Robot class (ideal robot) so that we can use its methods for rule checking
        '''
        self.toolRobot = Robot("toolRobot")
        
        
    def _compute_heightInfo(self, TERMES_World, erosionLoc): 
        '''
        Compute the surrounding height and height difference of the erosion location

        Parameters
        ----------
        TERMES_World : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        # Get erosion locatin coordinates
        x, y = erosionLoc
        
        # Get current height of erosion location
        heightInfo = TERMES_World._get_heightInfo(erosionLoc)
        h0 = heightInfo['cur_height']

        # Get surrounding height of erosion location
        surHeight_list = []
        surHeightDiff_list = []
        for heading_local in self.toolRobot.heading_map_local.values(): 
            # Get current height of neighbor locations
            loc_sur = erosionLoc + heading_local
            heightInfo_sur = TERMES_World._get_heightInfo(loc_sur)
            hn = heightInfo_sur['cur_height']
            # Compute height difference and record all information
            surHeightDiff = hn - h0
            surHeight_list.append([loc_sur, hn])
            surHeightDiff_list.append([loc_sur, surHeightDiff])
            
        return(surHeight_list, surHeightDiff_list)
        
        
    def _check_erosionCondition(self, TERMES_World, erosionLoc): 
        '''
        Check whether the robot/brick at the current location can be eroded 
        based on the erosion rules.

        Parameters
        ----------
        TERMES_World : TYPE
            DESCRIPTION.
        erosionLoc : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        # Get erosion locatin coordinates
        xer, yer = erosionLoc
        erosion_heightInfo = TERMES_World._get_heightInfo(erosionLoc)
        erosion_locInfo = TERMES_World._get_locInfo(erosionLoc)
        canErodeBrick = False
        canErodeRobot = False
        
        '''
        If there is a robot at the erosion location, one can erode the robot away
        '''
        if erosion_locInfo['loc_has_robot']: 
            canErodeRobot = True
            
        '''
        If there is no robot, the location has height > 0, and the erosion location is not the docking location
        further check the brick-erosion rules.
        '''
        if (not erosion_locInfo['loc_has_robot'] and erosion_heightInfo['cur_height'] > 0 
            and not np.array_equal(erosionLoc, TERMES_World.docking)): 
            # Compute surrounding height difference if a brick is removed
            surHeight_list, surHeightDiff_list = self._compute_heightInfo(TERMES_World, erosionLoc)
            surHeightDiff_afterRemoval_list = copy.deepcopy(surHeightDiff_list)
            for surHeightDiff_afterRemoval in surHeightDiff_afterRemoval_list: 
                surHeightDiff_afterRemoval[1] += 1
                
            # Check cliff
            cliffStatus = False
            for surHeightDiff_afterRemoval in surHeightDiff_afterRemoval_list: 
                if abs(surHeightDiff_afterRemoval[1]) > 1: 
                    cliffStatus = True
                    break
            
            # Check gap
            gapStatus = False
            if ((surHeightDiff_afterRemoval_list[0][1] >= 1 and surHeightDiff_afterRemoval_list[1][1] >= 1) 
                or (surHeightDiff_afterRemoval_list[2][1] >= 1 and surHeightDiff_afterRemoval_list[3][1] >= 1)): 
                gapStatus = True
                
            # If no cliff nor gap is created, a brick can be removed
            if not cliffStatus and not gapStatus: 
                canErodeBrick = True
            
        return(canErodeBrick, canErodeRobot)
    
    
    
class Eroder_Stigmergic(Eroder_Probabilistic): 
    '''
    A probabilistic eroder with erosion rules based on stigmergic rules of brick placement.
    To check if the brick at the current location is removeable, the eroder performs following tasks: 
        1) Check if removing the brick will cause a gap
        2) Check if removing the brick will cause a cliff between the erosion location and any of its children and parents
        3) Check if there is at least one child that has the same height after removing the brick
    '''
    
    def __init__(self, TERMES_World, eroder_name): 
        '''
        Call super class constructors.
        '''
        Eroder_Probabilistic.__init__(self, TERMES_World, eroder_name)
        
        '''
        Change robot feature parameters
        '''
        self.type = "stigmergic"
        
        '''
        Initialize a Robot class (ideal robot) so that we can use its methods for rule checking
        '''
        self.toolRobot = Robot("toolRobot")
        
        
    def _compute_surInfo(self, TERMES_World, erosionLoc): 
        '''
        Get informatin of surrounding locations.

        Parameters
        ----------
        TERMES_World : TYPE
            DESCRIPTION.
        erosionLoc : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        # Get erosion locatin coordinates
        xer, yer = erosionLoc
        
        # Record information of 4 surrounding locations
        surInfo_list = [{}, {}, {}, {}]
        
        # Get current height of erosion location. 
        heightInfo = TERMES_World._get_heightInfo(erosionLoc)
        h0 = heightInfo['cur_height']
            
        # Get desired height of erosion location.
        H0 = heightInfo['goal_height']

        # Get surrounding height of erosion location
        for ii, heading_local in enumerate(self.toolRobot.heading_map_local.values()): 
            # Get coordinate of the surrounding location
            sur_loc = erosionLoc + heading_local
            surInfo_list[ii]['loc'] = sur_loc
            
            # Get height of the surrounding location
            heightInfo_sur = TERMES_World._get_heightInfo(sur_loc)
            hn = heightInfo_sur['cur_height']
            surInfo_list[ii]['height'] = hn
                
            # Compute height difference of surrounding location
            surHeightDiff = hn - h0
            surInfo_list[ii]['heightDiff'] = surHeightDiff
            
            # Find if the surrounding location is child/parent of erosion location
            # If the erosion location is not part of the structure or the erosion location is the docking location, 
            # label all locations as 'NA'
            if H0 == 0 or np.array_equal(erosionLoc, TERMES_World.docking): 
                surInfo_list[ii]['childOrParent'] = 'NA'
            # Otherwise, check children and parents map
            else: 
                surInfo_list[ii]['childOrParent'] = 'NA'
                # Check parents map
                parentInfo_list = TERMES_World.parents_map[xer, yer]
                if len(parentInfo_list) != 0: 
                    for parentInfo in parentInfo_list: 
                        if np.array_equal(sur_loc, parentInfo['position']): 
                            surInfo_list[ii]['childOrParent'] = 'parent'
                # Check children map
                childInfo_list = TERMES_World.children_map[xer, yer]
                if len(childInfo_list) != 0: 
                    for childInfo in childInfo_list: 
                        if np.array_equal(sur_loc, childInfo['position']): 
                            surInfo_list[ii]['childOrParent'] = 'child'
            
        return(surInfo_list)
        
        
    def _check_erosionCondition(self, TERMES_World, erosionLoc): 
        '''
        Check whether the robot/brick at the current location can be eroded 
        based on the erosion rules.

        Parameters
        ----------
        TERMES_World : TYPE
            DESCRIPTION.
        erosionLoc : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        # Get erosion locatin coordinates
        xer, yer = erosionLoc
        erosion_heightInfo = TERMES_World._get_heightInfo(erosionLoc)
        erosion_locInfo = TERMES_World._get_locInfo(erosionLoc)
        canErodeBrick = False
        canErodeRobot = False
        
        '''
        If there is a robot at the erosion location, one can erode the robot away
        '''
        if erosion_locInfo['loc_has_robot']: 
            canErodeRobot = True
            
        '''
        If following condistions are met, further check the brick-erosion rules: 
            1) there is no robot 
            2) the location has height > 0
            3) the erosion location is not the docking location
            4) the brick to be removed is not the seed brick at the start location
        '''
        if (not erosion_locInfo['loc_has_robot'] 
            and erosion_heightInfo['cur_height'] > 0 
            and not np.array_equal(erosionLoc, TERMES_World.docking) 
            and not (np.array_equal(erosionLoc, TERMES_World.start) and erosion_heightInfo['cur_height'] == 1)): 
            # Get information of surrounding locations
            surInfo_list = self._compute_surInfo(TERMES_World, erosionLoc)
            surHeightDiff_list = []
            childParentLabel_list = []
            for surInfo in surInfo_list: 
                surHeightDiff_list.append(surInfo['heightDiff'])
                childParentLabel_list.append(surInfo['childOrParent'])
            
            # Get information of surrounding locations after erosion
            surInfo_afterRemoval_list = copy.deepcopy(surInfo_list)
            surHeightDiff_afterRemoval_list = []
            for surInfo_afterRemoval in surInfo_afterRemoval_list: 
                surInfo_afterRemoval['heightDiff'] += 1 # after removal, height difference increases by 1
                surHeightDiff_afterRemoval_list.append(surInfo_afterRemoval['heightDiff'])
            
            # Check if gap will be created
            createGap = True
            if not ((surHeightDiff_afterRemoval_list[0] >= 1 and surHeightDiff_afterRemoval_list[1] >= 1) 
                    or (surHeightDiff_afterRemoval_list[2] >= 1 and surHeightDiff_afterRemoval_list[3] >= 1)): 
                createGap = False
                
            # Check if any cliff will be created along the travel direction
            createCliffATD = False
            # If erosion location is not part of the structure, ignore this test.
            if not erosion_locInfo['loc_of_struct']: 
                pass
            # Otherwise, check if any parent or child will have height difference > 1 after removal
            else:
                for ii, childParentLabel in enumerate(childParentLabel_list): 
                    if childParentLabel == 'parent' or childParentLabel == 'child': 
                        if abs(surHeightDiff_afterRemoval_list[ii]) > 1: 
                            createCliffATD = True
                            break
                
            # Check if all child locations have the same height after removal
            childHasSameHeight = False
            # If erosion location is not part of the structure, ignore this test and set childHasSameHeight to True
            if not erosion_locInfo['loc_of_struct']: 
                childHasSameHeight = True
            # Otherwise, check each location
            else:
                # If there is no child location, the erosion location is the exit.
                # In this case, check locations labeled with 'NA'
                if not 'child' in childParentLabel_list: 
                    for ii, childParentLabel in enumerate(childParentLabel_list): 
                        if childParentLabel == 'NA' and surHeightDiff_afterRemoval_list[ii] == 0: 
                            childHasSameHeight = True
                            break
                # Otherwise, check locations labeled with 'child'
                else: 
                    childHasSameHeight = True 
                    for ii, childParentLabel in enumerate(childParentLabel_list): 
                        if childParentLabel == 'child' and surHeightDiff_afterRemoval_list[ii] != 0: 
                            childHasSameHeight = False
                            break
                
            # If no cliff nor gap is created and at least one child location has the same height after removal, 
            # a brick can be removed
            if not createGap and not createCliffATD and childHasSameHeight: 
                canErodeBrick = True
                
            # print(createGap, createCliffATD, childHasSameHeight)
            
        return(canErodeBrick, canErodeRobot)
        


# In[]

class Robot:
    '''
    Ideal robot which does not make mistakes. 
    This is also a super class for defining more realistic robots.
    '''
    
    def __init__(self, robot_name):
        '''
        Robot feature parameters.
        '''
        self.name = robot_name
        self.type = "ideal"
        
        '''
        Robot state parameters. These parameters are absolutely correct.
        '''
        self.robot_OnOff = "OFF"
        self.workMode = None
        self.workStage = 0
        self.carry_brick = False
        self.wrongLocalization = False
        
        '''
        Robot position parameters. These parameters are derived from local sensing and could be wrong.
        These parameters are what the robot believes.
        Notice that robot does not have any global knowledge about the structure.
        '''
        self.height = 0
        self.position = np.array([0, 0])
        self.heading = "E"
        self.parents = []
        self.children = []
        self.next_loc = np.copy(self.position)
        self.next_heading = None
        self.place_loc_list = []
        self.localExam_dict = {}
        
        '''
        These parameters are used for avoiding collisions and solving deadlock
        '''
        self.actionMemory = [] # memory of current and last action
        self.positionMemory = [] # memory of current and last position
        self.localExamMemory = [] # memory of current and last local examination results
        self.waitTimer = 0
        self.rechoose = False
        self.breakDeadlock = False
        self.robotNearby = [] # index is the same as senseRobot_map_local 
        
        '''
        These parameters are used for motion planning and localization.
        Define directions and orientations:
          (0, 0)--------- y -------->
             |        
             |            N(0)
             |             ^
             |             |
             x   W(3) <----|----> E(1)
             |             |
             |             V
             |            S(2)
             V        
        '''
        # unit vector corresponding to each heading
        self.heading_map = {"N": np.array([-1, 0]), 
                            "E": np.array([0, 1]), 
                            "S": np.array([1, 0]), 
                            "W": np.array([0, -1])}
        # heading after turning 90deg CW
        self.turning_map_CW = {"N":"E", "E":"S", "S":"W", "W":"N"}
        # heading after turning 90deg CCW
        self.turning_map_CCW = {"N":"W", "E":"N", "S":"E", "W":"S"}
        # heading after turning 180deg
        self.opposite_heading = {"N":"S", "E":"W", "W":"E", "S":"N"} 
        # sequence of actions needed to turn from heading 1 to heading 2
        self.turnActionSeq_dict = {"NN":[[]], "NE":[["turn 90deg CW"]], "NW":[["turn 90deg CCW"]], 
                                   "NS":[["turn 90deg CW","turn 90deg CW"],["turn 90deg CCW","turn 90deg CCW"]], 
                                   "EE":[[]], "ES":[["turn 90deg CW"]], "EN":[["turn 90deg CCW"]], 
                                   "EW":[["turn 90deg CW","turn 90deg CW"],["turn 90deg CCW","turn 90deg CCW"]], 
                                   "SS":[[]], "SW":[["turn 90deg CW"]], "SE":[["turn 90deg CCW"]], 
                                   "SN":[["turn 90deg CW","turn 90deg CW"],["turn 90deg CCW","turn 90deg CCW"]], 
                                   "WW":[[]], "WN":[["turn 90deg CW"]], "WS":[["turn 90deg CCW"]], 
                                   "WE":[["turn 90deg CW","turn 90deg CW"],["turn 90deg CCW","turn 90deg CCW"]]}
        
        '''
        These parameters are used for self-localization.
        '''
        # Rotation matrix that maps the position in robot coordinate to gloabl coordinate.
        # The x-axis in robot coordinate points along the heading direction.
        # When the robot is pointing south (S), its own frame is aligned with the global frame.
        # When the robot is pointing e.g. east (E), its own frame rotates by 90deg w.r.t. the global frame.
        self.rotationMatrix_dict = {"S": np.array([[ 1,  0], [ 0,  1]]), # 0 degree
                                    "E": np.array([[ 0, -1], [ 1,  0]]), # 90 degree
                                    "N": np.array([[-1,  0], [ 0, -1]]), # 180 degree
                                    "W": np.array([[ 0,  1], [-1,  0]])} # 270 degree
        # local heading map
        self.heading_map_local = {"front": np.array([1, 0]), 
                                  "back": np.array([-1, 0]), 
                                  "left": np.array([0, 1]), 
                                  "right": np.array([0, -1])}
        # local sensing directions for sensing robots
        self.senseRobot_map_local = [np.array([1, 0]), # 0: 1 brick ahead
                                     np.array([2, 0]), # 1: 2 brick ahead
                                     np.array([1, 1]), # 2: 45 degree left
                                     np.array([1, -1])] # 3: 45 degree right
        
        '''
        These parameters are used for defining the state of each action
        '''
        # Name of all 1st level actions
        self.actionName_list = ["move", # 0
                                "move up", # 1
                                "move down", # 2
                                "turn 90deg CW", # 3 
                                "turn 90deg CCW", # 4
                                "pick up", # 5
                                "place down", # 6
                                "wait", # 7
                                "leave"] # 8
        # Name of all outcomes of the 2nd level action
        self.actionStateL2_list = ["success", # 0
                                   "in progress", # 1
                                   "failure"] # 2
        
        '''
        These parameters are used for error checking and belief correction
        '''
        # This permutation matrix can be used to transform the view of the robot 
        # after the robot makes a turn with corresponding degree CCW
        self.permMatrix_dict = {0: np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]), 
                                90: np.array([[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,0,0,0]]), 
                                180: np.array([[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]]), 
                                270: np.array([[0,0,0,1],[1,0,0,0],[0,1,0,0],[0,0,1,0]])}
        self.unresolvedActionError = False
        
        '''
        Other simulation variables.
        '''
        # initialize random number generator for each robot
        self.rng = default_rng()
        # error and warning message
        self.errorMsg = "None"
        self.warningMsg_dict = {}
        
        
    # =============================================================================
    #
    # Define error logging functions
    # 
    # =============================================================================
    
    
    def _report_errorMsg(self, TERMES_World, errorMsg):
        '''
        This function can be used to log error messages.
        It can also be used to stop the program when error occurs.

        Parameters
        ----------
        errorMsg : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        # Check if the input is not string
        if type(errorMsg) != str:
            self.errorMsg = "Step#" + str(TERMES_World.cur_step) + ": " + self.name + ": " + "Invalid errorMsg type!"
            raise Exception(self.errorMsg)
            
        # Log error message
        self.errorMsg = "Step#" + str(TERMES_World.cur_step) + ": " + self.name + ": " + errorMsg
        raise Exception(self.errorMsg)
        
        
    def _report_warningMsg(self, TERMES_World, warningMsg, warningInfo = None):
        '''
        This function is used to log warning messages.

        Parameters
        ----------
        TERMES_World : TYPE
            DESCRIPTION.
        warningMsg : TYPE
            DESCRIPTION.
        warningInfo : TYPE, optional
            DESCRIPTION. This variable can be used to store more details of the warning

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        # Check if the input is not string
        if type(warningMsg) != str:
            self.errorMsg = "Step#" + str(TERMES_World.cur_step) + ": " + self.name + ": " + "Invalid warningMsg type!"
            raise Exception(self.errorMsg)
            
        # Log warning message
        msgKey = TERMES_World.cur_step
        if msgKey not in self.warningMsg_dict: 
            self.warningMsg_dict[msgKey] = [warningMsg]
        else:
            self.warningMsg_dict[msgKey].append(warningMsg)
            
        # Log critical simulation information to TERMES_World
        TERMES_World.log_criticalSimInfo(self.name, warningMsg, warningInfo)
        
        
    # =============================================================================
    #
    # Define robot initialization processes.
    # 
    # =============================================================================
    
    
    def _init_heading(self, TERMES_World):
        '''
        Initialize the heading direction based on the docking position

        Parameters
        ----------
        start : TYPE
            DESCRIPTION.
        docking : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        heading_vector = TERMES_World.docking - TERMES_World.start
        heading = self._find_heading(TERMES_World, heading_vector)
            
        return(heading)


    def robot_on(self, TERMES_World):
        '''
        Turn on the robot and initialize all state parameters

        Parameters
        ----------
        TERMES_World : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        # Initialize robot configuration parameters
        self.robot_OnOff = "ON"
        self.workStage = 0
        self.carry_brick = False
        self.wrongLocalization = False
        self.height = 1
        self.position = np.copy(TERMES_World.start)
        self.heading = self._init_heading(TERMES_World)
        self.parents = []
        self.children = []
        self.next_loc = np.copy(self.position)
        self.next_heading = None
        self.place_loc_list = []
        self.localExam_dict = {}
        self.actionMemory = []
        self.positionMemory = []
        self.localExamMemory = []
        self.waitTimer = 0
        self.rechoose = False
        self.breakDeadlock = False
        self.robotNearby = []
        self.unresolvedActionError = False
        
        # Update robot information in the world
        self._update_robotInWorld(TERMES_World, np.copy(self.position), self.heading)
        
        # Enter the next work mode
        self.workMode = "pick brick"
        
        # Log critical simulation information to TERMES_World
        warningMsg = self.name + " is deployed to the structure."
        warningInfo = None
        TERMES_World.log_criticalSimInfo(self.name, warningMsg, warningInfo)
        
        
    # =============================================================================
    #
    # Define the interaction between the robot and the structure (TERMES_World)
    # 
    # =============================================================================
    
    
    def _update_robotInWorld(self, TERMES_World, new_loc = np.array([0, 0]), new_heading = None):
        '''
        This function updates the robot configuration in TERMES_World.
        The robot configuration includes:
            position
            height
            heading
        
        robot_map records robot status in the structure: 
            e.g. robot_map[1,1] ---> "0" means there is no robot.
            e.g. robot_map[1,1] ---> {'name': xxx, 'height': xxx, 'heading': xxx}
            
        robotConfig_dict tells the position and configuration of each robot:
            e.g. robotConfig_dict[Robot#1] = {'position': xxx, 'heading': xxx, 'height': xxx}
                                  
        ### Notice that in this function, direct access to TERMES_World.robot_map is allowed. ###

        Parameters
        ----------
        TERMES_World : TYPE
            DESCRIPTION.
        new_loc : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        ''' Get old position information if new position is not provided. '''
        if not new_loc.any(): 
            if 'position' in TERMES_World.robotConfig_dict[self.name]: 
                new_loc = TERMES_World.robotConfig_dict[self.name]['position']
            else: 
                self._report_errorMsg(TERMES_World, "New position must be provided!")
        if not new_heading:
            if 'heading' in TERMES_World.robotConfig_dict[self.name]: 
                new_heading = TERMES_World.robotConfig_dict[self.name]['heading']
            else:
                self._report_errorMsg(TERMES_World, "New position must be provided!")
        
        ''' Remove robot from old location '''
        if 'position' in TERMES_World.robotConfig_dict[self.name]: 
            
            # If the old location is in the map, remove the robot
            # If the old location is outside the map, do nothing
            old_loc = TERMES_World.robotConfig_dict[self.name]['position']
            old_locInfo = TERMES_World._get_locInfo(old_loc)
            if old_locInfo['loc_in_map']: 
                TERMES_World.robot_map[old_loc[0], old_loc[1]] = 0
        
        ''' Update robot in the world '''
        # If the new location is in the map, update the map accordingly
        new_locInfo = TERMES_World._get_locInfo(new_loc)
        new_heightInfo = TERMES_World._get_heightInfo(new_loc)
        if new_locInfo['loc_in_map']: 
            # Get new height
            new_height = new_heightInfo['cur_height']
        
            # Add robot to the new location if the new location is not occupied. Otherwise, raise simulation error.
            if not new_locInfo['loc_has_robot']: 
                TERMES_World.robot_map[new_loc[0], new_loc[1]] = {'name': self.name, 
                                                                  'heading': new_heading, 
                                                                  'height': new_height}
            else:
                robot_other = new_locInfo['loc_robotName']
                self._report_errorMsg(TERMES_World, "Robot overlaps with " + robot_other + "!")
                
            # Update robotConfig_dict.
            TERMES_World.robotConfig_dict[self.name] = {'position': new_loc, 
                                                        'heading': new_heading, 
                                                        'height': new_height}
            
        # If the new location is not in the map, just update robot configuration.
        # No overlapping checking if the robot is outside the map since _sense_robot does not work.
        # A warning message will be added.
        else: 
            
            # height outside the map is ground since no brick is allowed to be placed outside the map
            new_height = 0
            
            # Update robotConfig_dict.
            TERMES_World.robotConfig_dict[self.name] = {'position': new_loc, 
                                                        'heading': new_heading, 
                                                        'height': new_height}
            
            # Add warning message
            self._report_warningMsg(TERMES_World, "Robot is outside map.")
        
        
    def _reset_robotInWorld(self, TERMES_World): 
        '''
        Remove the robot from the structure and reset all robot parameters
        
        ### Notice that in this function, direct access to TERMES_World.robot_map is allowed. ###

        Parameters
        ----------
        TERMES_World : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        ''' Remove robot from the structure and turn it off '''
        # Get and check real location
        real_loc = TERMES_World.robotConfig_dict[self.name]['position']
        # If the real location is in the map, updat robot_map
        # If the real location is outside the map, do nothing
        real_locInfo = TERMES_World._get_locInfo(real_loc)
        if real_locInfo['loc_in_map']: 
            TERMES_World.robot_map[real_loc[0], real_loc[1]] = 0
        
        # Update robotConfig_dict.
        TERMES_World.robotConfig_dict[self.name] = {}
        
        ''' Reset robot configuration parameters '''
        self.robot_OnOff = "OFF"
        self.workMode = None
        self.workStage = 0
        self.carry_brick = False
        self.wrongLocalization = False
        self.height = 0
        self.position = np.array([0, 0])
        self.heading = "E"
        self.parents = []
        self.children = []
        self.next_loc = np.copy(self.position)
        self.next_heading = None
        self.place_loc_list = []
        self.localExam_dict = {}
        self.actionMemory = []
        self.positionMemory = []
        self.localExamMemory = []
        self.waitTimer = 0
        self.rechoose = False
        self.breakDeadlock = False
        self.robotNearby = []
        self.unresolvedActionError = False
        
        
    # =============================================================================
    #
    # Define robot perception functions.
    # 
    # =============================================================================
    
    
    def _sense_heightDiff(self, cur_height, next_height):
        '''
        Sense the height difference between the height of current location (cur_height)
        and the height of a nearby location (next_height).
        Currently the sensed result is always correct.

        Parameters
        ----------
        cur_height : TYPE
            DESCRIPTION.
        next_height : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        heightDiff = next_height - cur_height
        
        return (heightDiff)
    
    
    def _sense_robot(self, TERMES_World):
        '''
        Sense robots at any height in the sensing range.
        Currently the function always returns correct results. 
        In current simulation, robot cannot exist outside the map
        
        Sensing range: "U" is unknown and "S" is detectable location.
            U S U
            S S S         ^ x' 
            U R U         |
                   y' <----

        Parameters
        ----------
        TERMES_World : TYPE
            DESCRIPTION.

        Returns
        -------
        bool
            True: There are robots nearby
            False: There are no robot nearby

        '''
        # Get real location and orientation
        real_loc = np.copy(TERMES_World.robotConfig_dict[self.name]['position'])
        real_heading = TERMES_World.robotConfig_dict[self.name]['heading']
        real_rotationMatrix = np.copy(self.rotationMatrix_dict[real_heading])
        
        # Sense robots in the sensing range.
        self.robotNearby = []
        for senseDirection_local in self.senseRobot_map_local: 
            senseResult = False
            senseDirection_global = np.matmul(real_rotationMatrix, senseDirection_local)
            sense_loc = real_loc + senseDirection_global
            # check if there is any robot
            sense_locInfo = TERMES_World._get_locInfo(sense_loc)
            if sense_locInfo['loc_has_robot']: 
                senseResult = True
            self.robotNearby.append(senseResult)
            
            
    def _sense_ground(self, TERMES_World): 
        '''
        Ground can be used as a reference since there is no marks on ground.
        For ideal robots, this function is not needed and thereby it does nothing here.

        Parameters
        ----------
        TERMES_World : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        correction = False # whether the robot has corrected its belief based on the sensing results
        
        return(correction)


    def examine_local(self, TERMES_World, dataMemorization = False):
        '''
        Robot senses the height difference between the current site and surrounding sites. 
        
        This function will update following variables: 
            self.parents
            self.children
            self.localExam_dict
            
        This function will be executed automatically in following functions: 
            move_to_next (for checking placeability)
            place_brick (for sensing the front height difference)
            check_actionError (for PLC)

        Parameters
        ----------
        TERMES_World : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''        
        ''' Robot corrects its height by using ground as a reference '''
        self._sense_ground(TERMES_World)
        
        ''' Get location and orientation from robot's belief '''
        robot_loc = np.copy(self.position)
        robot_heading = self.heading
        robot_height = self.height
        robot_rotationMatrix = np.copy(self.rotationMatrix_dict[robot_heading])
        
        ''' Get real location and orientation '''
        real_loc = np.copy(TERMES_World.robotConfig_dict[self.name]['position'])
        real_heading = TERMES_World.robotConfig_dict[self.name]['heading']
        real_height = TERMES_World.robotConfig_dict[self.name]['height']
        real_rotationMatrix = np.copy(self.rotationMatrix_dict[real_heading])        
        
        ''' Get height difference of surroundings '''
        heightPair_list = []
        for heading_local in self.heading_map_local.values():
            # Get real coordinate of surrounding location
            real_heading_global = np.matmul(real_rotationMatrix, heading_local)
            real_sur_loc = real_loc + real_heading_global
            real_sur_heightInfo = TERMES_World._get_heightInfo(real_sur_loc)
            real_sur_cur_height = real_sur_heightInfo['cur_height']
            
            # Get believed coordinate of surrounding location
            robot_heading_global = np.matmul(robot_rotationMatrix, heading_local)
            robot_sur_loc = robot_loc + robot_heading_global
            
            # Sense the height difference
            heightDiff = self._sense_heightDiff(real_height, real_sur_cur_height)
            # Infer the height of surrounding location from the sensed height difference
            heightPair_list.append((robot_sur_loc, heightDiff + robot_height))
            
        ''' Update information of parents and children of the current location '''
        self.parents = []
        self.children = []
        # Check if robot_loc is within the structure
        robot_locInfo = TERMES_World._get_locInfo(robot_loc)
        if robot_locInfo['loc_of_struct']: 
            
            # Check if robot_loc is start
            if not np.array_equal(robot_loc, TERMES_World.start):
                for parent in TERMES_World.parents_map[robot_loc[0], robot_loc[1]]:
                    for heightPair in heightPair_list: 
                        if np.array_equal(heightPair[0], parent['position']): 
                            heightInfo = TERMES_World._get_heightInfo(heightPair[0])
                            this_parent_dict = {"parent_position": heightPair[0],
                                                "parent_cur_height": heightPair[1],
                                                "parent_goal_height": heightInfo['goal_height']}
                            self.parents.append(this_parent_dict)
                            
            # Check if robot_loc is an exit
            if not any(np.array_equal(robot_loc, exit_loc) for exit_loc in TERMES_World.exit_list):
                for child in TERMES_World.children_map[robot_loc[0], robot_loc[1]]:
                    for heightPair in heightPair_list: 
                        if np.array_equal(heightPair[0], child['position']): 
                            heightInfo = TERMES_World._get_heightInfo(heightPair[0])
                            this_child_dict = {"child_position": heightPair[0], 
                                               "child_cur_height": heightPair[1], 
                                               "child_goal_height": heightInfo['goal_height'], 
                                               "child_prob": child["prob"]}
                            self.children.append(this_child_dict)
                    
        ''' 
        Update local examination results 
        self.localExam_dict = {'front': (loc, height), 
                               'back': (loc, height), 
                               'left': (loc, height), 
                               'right': (loc, height), 
                               'cur': (loc, height)} # current location of the robot
        '''
        # Generate local examination results
        self.localExam_dict = {}
        for ii, heading in enumerate(self.heading_map_local.keys()): 
            self.localExam_dict[heading] = heightPair_list[ii]
        self.localExam_dict['cur'] = (np.copy(self.position), self.height)
            
        # Memorize local examination results if required
        if dataMemorization: 
            self.memorize_localExam(TERMES_World)
            
            
    # =============================================================================
    #
    # Define functions for simulation analysis. These functions are not available for real robots
    # 
    # =============================================================================
    
    
    def _check_placeability_real(self, TERMES_World, place_loc):
        '''
        Check whether the given location is a valid placement location based on reality.
        The function assumes that the robot has a brick (checkState1 = True).

        Parameters
        ----------
        TERMES_World : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        xp, yp = place_loc
        place_heightInfo = TERMES_World._get_heightInfo(place_loc)
        h0 = place_heightInfo['cur_height'] # current height of current location
        H0 = place_heightInfo['goal_height'] # desired height of current location
        parentInfo_list = []
        childInfo_list = []
        
        checkState1 = True
        checkState2 = False
        checkState3 = False
        checkState4 = False
        placeability = False
        
        ''' Get information of parents and children of the given location '''
        # Check if the given location is within the structure
        place_locInfo = TERMES_World._get_locInfo(place_loc)
        if place_locInfo['loc_of_struct']: 
            
            # Check if the given location is start
            if not np.array_equal(place_loc, TERMES_World.start):
                for parent in TERMES_World.parents_map[xp, yp]:
                    parent_heightInfo = TERMES_World._get_heightInfo(parent['position'])
                    parentInfo_list.append({'position': parent['position'], 
                                            'cur_height': parent_heightInfo['cur_height'],  
                                            'goal_height': parent_heightInfo['goal_height']})
                            
            # Check if the given location is an exit
            if not any(np.array_equal(place_loc, exit_loc) for exit_loc in TERMES_World.exit_list):
                for child in TERMES_World.children_map[xp, yp]:
                    child_heightInfo = TERMES_World._get_heightInfo(child['position'])
                    childInfo_list.append({'position': child['position'], 
                                           'cur_height': child_heightInfo['cur_height'],  
                                           'goal_height': child_heightInfo['goal_height']})
        
        ''' Check placement conditions '''
        # Check if the current height is less than the desired height: h0 < H0
        if h0 < H0:
            checkState2 = True
            
            # Check if for all parent sites i: hi > h0 OR hi = Hi
            if len(parentInfo_list) != 0: 
                for parentInfo in parentInfo_list: 
                    hi = parentInfo['cur_height']
                    Hi = parentInfo['goal_height']
                    if hi > h0 or hi == Hi:
                        checkState3 = True
                    else:
                        checkState3 = False
                        break
                if checkState3:
                    
                    # Check if for all child sites i: hi = h0 OR |Hi - H0| > 1
                    if len(childInfo_list) != 0: 
                        for childInfo in childInfo_list: 
                            hi = childInfo['cur_height']
                            Hi = childInfo['goal_height']
                            if hi == h0 or abs(Hi - H0) > 1:
                                checkState4 = True
                            else:
                                checkState4 = False
                                break
                    else:
                        checkState4 = True
                    if checkState4:
                        
                        # Current location is placeable only if all conditions are satisfied
                        placeability = True
            
        return(placeability, [checkState1, checkState2, checkState3, checkState4])
    
    
    def check_localization(self, TERMES_World): 
        '''
        This function checks if the current localization is correct.

        Parameters
        ----------
        TERMES_World : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        ''' Get location and orientation from robot's belief '''
        robot_loc = np.copy(self.position)
        robot_heading = self.heading
        robot_height = self.height
        
        ''' Get real location and orientation '''
        real_loc = np.copy(TERMES_World.robotConfig_dict[self.name]['position'])
        real_heading = TERMES_World.robotConfig_dict[self.name]['heading']
        real_height = TERMES_World.robotConfig_dict[self.name]['height']
        
        ''' Check if there is any mismatch '''
        if (not np.array_equal(robot_loc, real_loc) 
            or robot_heading != real_heading 
            or robot_height != real_height):
            self.wrongLocalization = True
        else:
            self.wrongLocalization = False
        
        
    # =============================================================================
    #
    # Define robot computation functions.
    # 
    # =============================================================================
    
    
    def _find_heading(self, TERMES_World, heading_vector):
        '''
        Find the heading that matches the heading vector.
        heading_vector = next loc - cur loc
        next loc must be an adjacent location of cur_loc, 
        in other words, heading_vector must have norm of 1.

        Parameters
        ----------
        heading_vector : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        heading = None
        for kk, vv in self.heading_map.items():
            if np.array_equal(heading_vector, vv):
                heading = kk
                break
        
        # Report error if heading_vector is unexpected
        if not heading: 
            self._report_errorMsg(TERMES_World, "Unexpected heading_vector!")
            
        return(heading)
    
    
    def _check_placeability(self, TERMES_World):
        '''
        Check whether the robot can place a brick at its current location based on the robot's belief.

        Parameters
        ----------
        TERMES_World : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        real_selfPosition_heightInfo = TERMES_World._get_heightInfo(self.position)
        h0 = self.height # believed current height of current location
        H0 = real_selfPosition_heightInfo['goal_height'] # desired height of believed current location
        
        checkState1 = False
        checkState2 = False
        checkState3 = False
        checkState4 = False
        placeability = False
        
        # Check if the robot has a brick
        if self.carry_brick: 
            checkState1 = True
            
            # Check if the current height is less than the desired height: h0 < H0
            if h0 < H0:
                checkState2 = True
                
                # Check if for all parent sites i: hi > h0 OR hi = Hi
                for parent in self.parents:
                    hi = parent["parent_cur_height"]
                    Hi = parent["parent_goal_height"]
                    if hi > h0 or hi == Hi:
                        checkState3 = True
                    else:
                        checkState3 = False
                        break
                if checkState3:
                    
                    # Check if for all child sites i: hi = h0 OR |Hi - H0| > 1
                    # Check if the location is an exit. Exit does not have children.
                    if not any(np.array_equal(self.position, exit_loc) for exit_loc in TERMES_World.exit_list):
                        for child in self.children:
                            hi = child["child_cur_height"]
                            Hi = child["child_goal_height"]
                            if hi == h0 or abs(Hi - H0) > 1:
                                checkState4 = True
                            else:
                                checkState4 = False
                                break
                    else:
                        checkState4 = True
                    if checkState4:
                        
                        # Current location is placeable only if all conditions are satisfied
                        placeability = True
            
        return(placeability, [checkState1, checkState2, checkState3, checkState4])
    
    
    def _check_robotInBoundary(self, TERMES_World): 
        '''
        Check if the robot is within the structure/map based on the reality and its belief

        Parameters
        ----------
        TERMES_World : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        # Get real and believed location
        real_loc = np.copy(TERMES_World.robotConfig_dict[self.name]['position'])
        robot_loc = np.copy(self.position)
        
        # Check if the location is within the structure        
        real_locInfo = TERMES_World._get_locInfo(real_loc)
        robot_locInfo = TERMES_World._get_locInfo(robot_loc)
        
        # Generate a dictionary that includes all the information
        CLIB_dict = {'robotInStructure': robot_locInfo['loc_of_struct'], 
                     'robotInMap': robot_locInfo['loc_in_map'], 
                     'robotInMargin': robot_locInfo['loc_in_accessMargin'], 
                     'realInStructure': real_locInfo['loc_of_struct'], 
                     'realInMap': real_locInfo['loc_in_map'], 
                     'realInMargin': real_locInfo['loc_in_accessMargin']}
        
        return(CLIB_dict)
    
    
    def _check_leaveConditions(self, TERMES_World):
        '''
        Check if the robot can leave the structure.
        This function returns two results based on robot's belief and reality respectively.

        Parameters
        ----------
        TERMES_World : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        # Get real and believed location and height
        real_loc = np.copy(TERMES_World.robotConfig_dict[self.name]['position'])
        real_height = TERMES_World.robotConfig_dict[self.name]['height']
        robot_loc = np.copy(self.position)
        robot_height = self.height
        
        # Check if the location is within the structure
        real_locInfo = TERMES_World._get_locInfo(real_loc)
        robot_locInfo = TERMES_World._get_locInfo(robot_loc)
        
        # Examine the leaving conditions based on robot's belief
        # If robot is at any exit or the current height is 0 or the robot is outside the structure, robot can leave.
        if (any(np.array_equal(robot_loc, exit_loc) for exit_loc in TERMES_World.exit_list) 
            or robot_height == 0 or not robot_locInfo['loc_of_struct']):
            robot_canLeave = True
        else:
            robot_canLeave = False
            
        # Examne the leaving conditions based on reality
        # If robot is at any exit or the current height is 0 or the robot is outside the structure, robot can leave.
        if (any(np.array_equal(real_loc, exit_loc) for exit_loc in TERMES_World.exit_list) 
            or real_height == 0 or not real_locInfo['loc_of_struct']):
            real_canLeave = True
        else:
            real_canLeave = False
            
        return(robot_canLeave, real_canLeave)
    
    
    # =========================== CURRENTLY UNUSED ===========================
    def memorize_action(self, TERMES_World, new_action, memory_length = 2):
        '''
        Robot memorizes its actions.
        For memory with length n, and assuming action_i is the current action
        actionMemory = [action_(i-(n-1)), action_(i-(n-2)), ..., action_i]

        Parameters
        ----------
        action : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        # If length is smaller than the setting, add new item to the list
        if len(self.actionMemory) < memory_length:
            self.actionMemory.append(new_action)
            
        # If length reaches the setting, add new item and remove the oldest item
        elif len(self.actionMemory) == memory_length:
            self.actionMemory.append(new_action)
            self.actionMemory.pop(0)
        
        # Report exceptions
        else:
            self._report_errorMsg(TERMES_World, "Unexpected memory length (memorize_action)!")
            
            
    def memorize_position(self, TERMES_World, memory_length = 2):
        '''
        Robot memorizes its positions.
        For memory with length n, and assuming position_i is the current position
        positionMemory = [position_(i-(n-1)), position_(i-(n-2)), ..., position_i]
        
        This function will be executed automatically in following functions: 
            check_deadlock

        Parameters
        ----------
        TERMES_World : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        # If length is smaller than the setting, add new item to the list
        if len(self.positionMemory) < memory_length:
            self.positionMemory.append(np.copy(self.position))
            
        # If length reaches the setting, add new item and remove the oldest item
        elif len(self.positionMemory) == memory_length:
            self.positionMemory.append(np.copy(self.position))
            self.positionMemory.pop(0)
        
        # Report exceptions
        else:
            self._report_errorMsg(TERMES_World, "Unexpected memory length (memorize_position)!") 
            
            
    def memorize_localExam(self, TERMES_World, memory_length = 2): 
        '''
        Robot memorizes its local examination results.
        For memory with length n, and assuming localExam_i is the current local examination result
        localExamMemory = [localExam_(i-(n-1)), localExam_(i-(n-2)), ..., localExam_i]
        
        This function will be executed automatically in following functions: 
            examine_local

        Parameters
        ----------
        TERMES_World : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        # If length is smaller than the setting, add new item to the list
        if len(self.localExamMemory) < memory_length:
            self.localExamMemory.append(copy.deepcopy(self.localExam_dict))
            
        # If length reaches the setting, add new item and remove the oldest item
        elif len(self.localExamMemory) == memory_length:
            self.localExamMemory.append(copy.deepcopy(self.localExam_dict))
            self.localExamMemory.pop(0)
        
        # Report exceptions
        else:
            self._report_errorMsg(TERMES_World, "Unexpected memory length (memorize_localExam)!") 
    
    
    def check_deadlock(self, TERMES_World):
        '''
        Solve deadlock amomg multiple agents.
        This function works closely with following functions:
            _avoid_collisions
            move_to_next
            move_to_ground_or_exit

        Parameters
        ----------
        TERMES_World : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        ''' Memorize its current position and count the time of stationariness '''
        self.memorize_position(TERMES_World)
        if all([np.array_equal(self.positionMemory[0], loc) for loc in self.positionMemory]): 
            self.waitTimer += 1
        else:
            self.waitTimer = 0
            
        ''' Find and solve the deadlock '''
        # Robot does rechoose, rechoose, rechoose and breakDeadlock every certain amount of wait actions
        timeInterval = 3
        timePoints = np.array([1, 2, 3, 4]) * timeInterval
        
        # When waitTimer is reset to 0, deadlock is solved
        if self.waitTimer == 0:
            self.rechoose = False
            self.breakDeadlock = False
            
        # When waitTimer is larger than 0, choose actions to solve deadlock
        elif self.waitTimer > 0:
            
            # Robot rechooses its next position every timeInterval wait actions
            if any([self.waitTimer % timePoints[-1] == ii for ii in timePoints[0:-1]]): 
                self.rechoose = True
                
            # Robot breaks the deadlock every (4 * timeInterval) wait actions
            elif self.waitTimer % timePoints[-1] == 0:
                self.breakDeadlock = True
                
            # Otherwise, robot does not do anything
            else:
                self.rechoose = False
                self.breakDeadlock = False
           
        ''' Fail to solve the deadlock. Report errors. '''
        # If waitTimer is larger than (12 * timeInterval), raise simulation error.
        # Notice that very rarely (~1/1000) this is not an error. It just takes very long for robots to break tie.
        # If this becomes an issue, increase the waitTimer threshold of reporting errors.
        if self.waitTimer > (timePoints[-1] * 3):
            self._report_errorMsg(TERMES_World, "Deadlock!")
            
            
    def check_actionError(self, TERMES_World, actionName, **kwargs): 
        '''
        Check if there is any action error after the robot takes the given action.
        For ideal robots, this function is not needed and thereby it does nothing here.

        Parameters
        ----------
        TERMES_World : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        # Record checking result
        checkResult = {'actionName': actionName, 
                       'errorDetected': False,
                       'preAction_data': None, 
                       'postAction_data': None, 
                       'inferredAction': None}
            
        return(checkResult)
    
    
    def plan_motion(self, TERMES_World, start_config = None, goal_config = None): 
        '''
        Plan a sequence of motions to travel to a given adjacent location. 
        configuration = {'loc': np.array, 'heading': str}
        Basic motion planning flow: 
            turn to goal_loc
            move to goal_loc
            turn to goal_heading

        Parameters
        ----------
        TERMES_World : TYPE
            DESCRIPTION.
        start_config : dictionary, optional
            DESCRIPTION. The default is None.
        goal_config : dictionary, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        '''
        # Get initial configuration. If not given, use the current configuration.
        if not start_config: 
            start_loc = np.copy(self.position)
            start_heading = self.heading
            start_config = {'loc': start_loc, 'heading': start_heading}
        else:
            start_loc = start_config['loc']
            start_heading = start_config['heading']
        
        # Get goal configuration. If not given, use the next location
        if not goal_config: 
            goal_loc = np.copy(self.next_loc)
            goal_heading = self.next_heading
            goal_config = {'loc': goal_loc, 'heading': goal_heading}
        else:
            goal_loc = goal_config['loc']
            goal_heading = goal_config['heading']
            
        # Check if goal_loc is an adjacent location
        displacement = goal_loc - start_loc
        distance = np.linalg.norm(displacement)
        if distance > 1: 
            self._report_errorMsg(TERMES_World, "Goal location is not an adjacent location!")
            
        # Start motion planning
        planActionSeq = [] # sequence of actions planned
        cur_config = {'loc': start_loc, 'heading': start_heading}
        
        ''' turn to goal_loc '''
        if distance > 0: 
            s2g_heading = self._find_heading(TERMES_World, displacement)
            turnActionSeq_list = self.turnActionSeq_dict[start_heading + s2g_heading]
            if len(turnActionSeq_list) > 1: 
                turnActionSeqInd = self.rng.choice(len(turnActionSeq_list))
                turnActionSeq = turnActionSeq_list[turnActionSeqInd]
            else:
                turnActionSeq = turnActionSeq_list[0]
            planActionSeq += turnActionSeq
            cur_config['heading'] = s2g_heading
            
        ''' move to goal_loc '''
        if distance > 0: 
            planActionSeq.append("move")
            cur_config['loc'] = goal_loc
            
        ''' turn to goal_heading '''
        if goal_heading: 
            cur_heading = cur_config['heading']
            turnActionSeq_list = self.turnActionSeq_dict[cur_heading + goal_heading]
            if len(turnActionSeq_list) > 1: 
                turnActionSeqInd = self.rng.choice(len(turnActionSeq_list))
                turnActionSeq = turnActionSeq_list[turnActionSeqInd]
            else:
                turnActionSeq = turnActionSeq_list[0]
            planActionSeq += turnActionSeq
            cur_config['heading'] = goal_heading 
            
        ''' Check if goal_config is achieved '''
        if goal_config['heading']: 
            if cur_config['heading'] != goal_config['heading']: 
                self._report_errorMsg(TERMES_World, "Motion planning fails!")
        
        if not np.array_equal(cur_config['loc'], goal_config['loc']): 
            self._report_errorMsg(TERMES_World, "Motion planning fails!")
            
        ''' Return planned action sequence'''
        return(planActionSeq)


    # =============================================================================
    #
    # Define the 1st level robot actions.
    # Inside each function, after the robot takes an action: 
    #   1) the function will report any change to TERMES_World.
    #   2) the robot updates its belief assuming that the action is successful
    # 
    # =============================================================================
    
    
    def _move_forward(self, TERMES_World):
        '''
        Robot moves forward in the heading direction.
        There are three possible actions:
            a. move on a flat surface
            b. move up for one brick
            c. move down for one brick
        For each action, an actionState which can be generated randomly is assigned
        to represent different outcome after taking such action.

        Parameters
        ----------
        TERMES_World : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        # Get real location and orientation
        real_heading = TERMES_World.robotConfig_dict[self.name]['heading']
        real_cur_loc = np.copy(TERMES_World.robotConfig_dict[self.name]['position'])
        real_cur_height = TERMES_World.robotConfig_dict[self.name]['height']
        real_next_loc = real_cur_loc + self.heading_map[real_heading]
        real_next_heightInfo = TERMES_World._get_heightInfo(real_next_loc)
        real_next_height = real_next_heightInfo['cur_height']
        real_heightDiff = real_next_height - real_cur_height
        
        # Check the height difference
        action = None
        if real_heightDiff == 0: # move
            action = self.actionName_list[0]
        elif real_heightDiff == 1: # move up
            action = self.actionName_list[1]
        elif real_heightDiff == -1: # move down
            action = self.actionName_list[2]
        else:
            self._report_errorMsg(TERMES_World, "Unreachable location ahead!")
            
        # Generate the outcome of the action. Currently we assume that every action is successful.
        actionState = "success"
        
        ''' Update TERMES_World based on different outcomes. '''
        if actionState == "success":
            self._update_robotInWorld(TERMES_World, real_next_loc, None)
            
        ''' Robot updates its belief '''
        self.position += self.heading_map[self.heading]
        self.height += real_heightDiff
        
        return(action, actionState)


    def _turn_90deg(self, TERMES_World, turnDirection):
        '''
        Robot turns by 90 degree clockwise (CW) or counterclockwise (CCW).

        Parameters
        ----------
        turnDirection : TYPE
            DESCRIPTION.
        TERMES_World : TYPE
            DESCRIPTION.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        # Get real location and orientation
        real_cur_heading = TERMES_World.robotConfig_dict[self.name]['heading']
        
        # Check turning direction
        action = None
        if turnDirection == "clockwise": # turn 90deg CW
            action = self.actionName_list[3]
            real_next_heading = self.turning_map_CW[real_cur_heading]
        elif turnDirection == "counterclockwise": # turn 90deg CCW
            action = self.actionName_list[4]
            real_next_heading = self.turning_map_CCW[real_cur_heading]
        else:
            self._report_errorMsg(TERMES_World, "Invalid turnDirection!")
        
        # Generate the outcome of the action. Currently we assume that every action is successful.
        actionState = "success"
        
        ''' Update TERMES_World based on different outcomes. '''
        if actionState == "success":
            self._update_robotInWorld(TERMES_World, np.array([0, 0]), real_next_heading)
            
        ''' Robot updates its belief '''
        robot_cur_heading = self.heading
        if turnDirection == "clockwise": # turn 90deg CW
            self.heading = self.turning_map_CW[robot_cur_heading]
        elif turnDirection == "counterclockwise": # turn 90deg CCW
            self.heading = self.turning_map_CCW[robot_cur_heading]
        
        return(action, actionState)
    
    
    def _pick_up(self, TERMES_World):
        '''
        Robot picks up a brick from the location in front of itself.

        Parameters
        ----------
        TERMES_World : TYPE
            DESCRIPTION.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        # Get real location and orientation
        real_heading = TERMES_World.robotConfig_dict[self.name]['heading']
        real_cur_loc = np.copy(TERMES_World.robotConfig_dict[self.name]['position'])
        real_next_loc = real_cur_loc + self.heading_map[real_heading]
        
        # Check if the robot is facing the docking location
        if not np.array_equal(real_next_loc, TERMES_World.docking):
            self._report_errorMsg(TERMES_World, "Not facing docking!")
        
        # Check if the robot does not have a brick yet
        if self.carry_brick:
            self._report_errorMsg(TERMES_World, "Already has brick!")
        
        # Get the action
        action = self.actionName_list[5]
        
        # Generate the outcome of the action. Currently we assume that every action is successful.
        actionState = "success"
        
        ''' Update TERMES_World based on different outcomes. '''
        if actionState == "success":
            TERMES_World.brickPickupNum += 1
        
        ''' Robot updates its belief '''
        self.carry_brick = True
        
        return(action, actionState)
    
    
    def _place_down(self, TERMES_World):
        '''
        Robot places down a brick to the location in front of itself.

        Parameters
        ----------
        TERMES_World : TYPE
            DESCRIPTION.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        # Get real location and orientation.
        real_heading = TERMES_World.robotConfig_dict[self.name]['heading']
        real_cur_loc = np.copy(TERMES_World.robotConfig_dict[self.name]['position'])
        real_cur_height = TERMES_World.robotConfig_dict[self.name]['height']
        real_next_loc = real_cur_loc + self.heading_map[real_heading]
        real_next_locInfo = TERMES_World._get_locInfo(real_next_loc)
        real_next_heightInfo = TERMES_World._get_heightInfo(real_next_loc)
        real_next_height = real_next_heightInfo['cur_height']
        real_heightDiff = real_next_height - real_cur_height
        
        # Check if the placement location has the same height.
        if real_heightDiff != 0:
            self._report_errorMsg(TERMES_World, "Location is unplaceable (different height)!")
            
        # Check if the placement location is robot-free.
        if real_next_locInfo['loc_has_robot']: 
            self._report_errorMsg(TERMES_World, "Location is unplaceable (occupied by robot)!")
        
        # Check if the robot has a brick.
        if not self.carry_brick:
            self._report_errorMsg(TERMES_World, "Location is unplaceable (no brick)!")
            
        # Get the action.
        action = self.actionName_list[6]
            
        # Generate the outcome of the action. Currently we assume that every action is successful.
        actionState = "success"
        
        ''' Update TERMES_World based on different outcomes. '''
        if actionState == "success":
            TERMES_World.add_bricks(real_next_loc)
            TERMES_World.brickPlacementNum += 1 
            
        ''' Robot updates its belief '''
        self.carry_brick = False
        
        return(action, actionState)
    
    
    def _wait(self, TERMES_World):
        '''
        Robot waits for one time step.

        Parameters
        ----------
        TERMES_World : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        # Get the action.
        action = self.actionName_list[7]
            
        # Generate the outcome of the action. Currently we assume that every action is successful.
        actionState = "success"
        
        return(action, actionState)
    
    
    def _leave(self, TERMES_World):
        '''
        Robot leaves the structure.
        This is a virtual 1st level action, which can be considered as 
        perimeter following in real TERMES system.
        Currently this action has 100% success rate and takes no time.

        Parameters
        ----------
        TERMES_World : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        # Get the action.
        action = self.actionName_list[8]
            
        # Generate the outcome of the action. Currently we assume that every action is successful.
        actionState = "success"
        
        return(action, actionState)
    
    
    def _avoid_collisions(self, TERMES_World):
        '''
        Robot avoids collisions and solves deadlock.
        This function worls closely with following functions:
            check_deadlock
            move_to_next
            move_to_ground_or_exit
        Local robot sensing map:
            0: 1 brick ahead
            1: 2 brick ahead
            2: 45 degree left
            3: 45 degree right

        Parameters
        ----------
        TERMES_World : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''        
        # Sense robot nearby
        self._sense_robot(TERMES_World)
        
        # If there is any robot nearby, robot waits
        collisionAlert = any(self.robotNearby)
        if collisionAlert: 
            action, actionState = self._wait(TERMES_World)
        else:
            action = None
            actionState = None
            
        return(action, actionState)

    
    # =============================================================================
    #
    # Define the 2nd level robot actions.
    # 
    # =============================================================================
    
    
    def pick_brick(self, TERMES_World):
        '''
        Robot picks up a brick.

        Parameters
        ----------
        TERMES_World : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        # Check if the work mode matches the action
        if self.workMode != "pick brick":
            self._report_errorMsg(TERMES_World, "Wrong workMode (pick_brick)!")
            
        # Take 1st level action
        action, actionStateL1 = self._pick_up(TERMES_World)
        
        # Generate the outcome of the 2nd level action.
        actionStateL2 = self.actionStateL2_list[0]
        
        # Move to the next work mode when the current work mode is over
        self.workMode = "move to next"
        
        return(action, [actionStateL1, actionStateL2])
    
    
    def move_to_next(self, TERMES_World):
        '''
        Robot randomly moves to one of the child locations of current location.

        Parameters
        ----------
        TERMES_World : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        ''' Check if the work mode matches the action '''
        if self.workMode != "move to next":
            self._report_errorMsg(TERMES_World, "Wrong workMode (move_to_next)!")
            
        ''' Check if it is needed to stop the placement '''
        # If there is an unresolved action error, 
        # robot abandons this placement process and goes back to "move to ground or exit" or "leave structure"
        if self.unresolvedActionError: 
            action, actionStateL1 = self._wait(TERMES_World)
            actionStateL2 = self.actionStateL2_list[2] # outcome is failure
            self.workStage = 0
            
            # If the robot reaches one of the exits or ground, robot leaves structure
            robot_canLeave, real_canLeave = self._check_leaveConditions(TERMES_World)
            if robot_canLeave: 
                self.workMode = "leave structure"
            # If not, robot moves to ground or exit
            else:
                self.workMode = "move to ground or exit"
            
            return(action, [actionStateL1, actionStateL2])
    
        ''' Choose a new next position if the robot reaches last next position '''
        if self.workStage == 0:
            # Check if the current location is outside boundary
            CLIB_dict = self._check_robotInBoundary(TERMES_World)
            
            # If robot believes that it is outside the structure, stop the navigation
            if not CLIB_dict['robotInStructure']: 
                # Check leaving condition
                robot_canLeave, real_canLeave = self._check_leaveConditions(TERMES_World)
                # If real leaving condition is NOT satisfied, raise error
                if not real_canLeave: 
                    self._report_errorMsg(TERMES_World, "Robot lost its location on the structure (move_to_next)!")
                # If real leaving condition is satisfied, leave the structure
                else: 
                    action, actionStateL1 = self._wait(TERMES_World)
                    actionStateL2 = self.actionStateL2_list[2] # outcome is failure
                    self.workMode = "leave structure"
                    self.workStage = 0
                    self._report_warningMsg(TERMES_World, "Robot left the structure due to localization error (move_to_next).")
                    
                    # Stop execution of the function and return action information
                    return(action, [actionStateL1, actionStateL2])

            # If robot believes that it is on the structure, start nevigation
            else: 
                # Choose a child location
                xr, yr = self.position
                prob_list = []
                child_loc_list = []
                if TERMES_World.pathProbMapType == "Optimized":
                    for xp, yp, prob in TERMES_World.pathProbMap[xr, yr]:
                        prob_list.append(prob)
                        child_loc_list.append(np.array([xp + 1, yp + 1]))
                elif TERMES_World.pathProbMapType == "Normal":
                    for child in self.children:
                        prob_list.append(child["child_prob"])
                        child_loc_list.append(child["child_position"])
                # Set up goal configuration
                self.next_loc = self.rng.choice(child_loc_list, 1, p = prob_list)[0]
                self.next_heading = None
                
                # Enter the next work stage
                self.workStage = 1
            
        ''' Take 1st level action '''
        if self.workStage == 1: 
            # Perform motion planning to travel to next loc
            planActionSeq = self.plan_motion(TERMES_World)
            # Take the 1st action
            if len(planActionSeq) > 0: 
                planAction = planActionSeq[0]
                if planAction == 'turn 90deg CW': 
                    action, actionStateL1 = self._turn_90deg(TERMES_World, 'clockwise')
                elif planAction == 'turn 90deg CCW': 
                    action, actionStateL1 = self._turn_90deg(TERMES_World, 'counterclockwise')
                elif planAction == 'move': 
                    # move forward when collision avoidance is not needed or 
                    # when robot tries to break the deadlock and the front location is robot-free
                    action, actionStateL1 = self._avoid_collisions(TERMES_World)
                    if not action or (self.breakDeadlock and not self.robotNearby[0]): 
                        action, actionStateL1 = self._move_forward(TERMES_World)
                else:
                    self._report_errorMsg(TERMES_World, "Unexpected planned action!")
            # If no action is needed, robot waits
            else:
                action, actionStateL1 = self._wait(TERMES_World)
            actionStateL2 = self.actionStateL2_list[1]
            
            # Enter the next work stage
            self.workStage = 2
            
        ''' Check if the current location is a placement location '''        
        if self.workStage == 2:
            # Perform a local examination first since checking placeability relies on the latest local information
            self.examine_local(TERMES_World)
            placeability, checkState_list = self._check_placeability(TERMES_World)
            # If the current location is placeable and not in the history of placement location, 
            # robot enters the next work mode to place a brick at current location
            if placeability and not any([np.array_equal(self.position, loc) for loc in self.place_loc_list]): 
                self.place_loc_list.append(np.copy(self.position))
                actionStateL2 = self.actionStateL2_list[0]
                self.workMode = "place brick"
                self.workStage = 0
                
                # Stop execution of the function and return action information
                return(action, [actionStateL1, actionStateL2])
            
            else:
                # Enter the next work stage
                self.workStage = 3
        
        ''' Check if the robot reaches one of the exits or ground '''
        if self.workStage == 3:
            robot_canLeave, real_canLeave = self._check_leaveConditions(TERMES_World)
            if robot_canLeave: 
                actionStateL2 = self.actionStateL2_list[0]
                self.workMode = "leave structure"
                self.workStage = 0
                
                # Stop execution of the function and return action information
                return(action, [actionStateL1, actionStateL2])
            
            else:
                # Enter the next work stage
                self.workStage = 4
                
        ''' Return action information and set the work stage for next action '''
        if self.workStage == 4:
            # If next position is reached, choose a new one
            if np.array_equal(self.position, self.next_loc): 
                self.workStage = 0
                
            # If robot waits for too long, choose a new one
            elif self.rechoose:
                self.workStage = 0
                
            # Otherwise, skip choosing new next location
            else:
                self.workStage = 1
                
            # Return action information from work stage 1
            return(action, [actionStateL1, actionStateL2])
        
        ''' Check exceptions '''
        self._report_errorMsg(TERMES_World, "Exception (move_to_next)!")
    
    
    def move_to_ground_or_exit(self, TERMES_World):
        '''
        Robot prepares for leaving the structure by going to the ground or any exit.
        In this work mode, robot tries to leave the structure ASAP by: 
            1) Robot searches for accessible ground nearby first.
            2) If there is no accessible ground, robot moves to the next location based on the policy.

        Parameters
        ----------
        TERMES_World : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        ''' Check if the work mode matches the action '''
        if self.workMode != "move to ground or exit":
            self._report_errorMsg(TERMES_World, "Wrong workMode (move_to_ground_or_exit)!")
    
        ''' Choose a new next position if the robot reaches last next position '''
        if self.workStage == 0:
            # Find all accessible ground locations
            robot_height = self.height # location and orientation from robot's belief
            ground_loc = None
            ground_loc_list = []
            for heading, localExam in self.localExam_dict.items():
                if heading != 'cur': 
                    sur_loc = localExam[0]
                    sur_height = localExam[1]
                    heightDiff = abs(sur_height - robot_height)
                    # check accessibility
                    if sur_height == 0 and heightDiff <= 1:
                        # check if the position is outside the structure
                        sur_locInfo = TERMES_World._get_locInfo(sur_loc)
                        if not sur_locInfo['loc_of_struct']: 
                            ground_loc_list.append(sur_loc)
            # Choose a ground location
            if len(ground_loc_list) >= 1:
                ground_loc = self.rng.choice(ground_loc_list)
                
            # If there is an accessible ground, set up the goal configuration
            if len(ground_loc_list) >= 1:
                self.next_loc = np.copy(ground_loc)
                self.next_heading = None
                
            # If there is no accessible ground, choose a child location as the next location
            if len(ground_loc_list) == 0:
                # Check if the current location is outside boundary
                CLIB_dict = self._check_robotInBoundary(TERMES_World)
                
                # If robot believes that it is outside the structure, stop the navigation
                if not CLIB_dict['robotInStructure']: 
                    # Check real leaving condition
                    robot_canLeave, real_canLeave = self._check_leaveConditions(TERMES_World)
                    # If real leaving condition is NOT satisfied, raise error.
                    if not real_canLeave: 
                        self._report_errorMsg(TERMES_World, "Robot lost its location on the structure (move_to_ground_or_exit)!")
                    # If real leaving condition is satisfied, leave the structure
                    else: 
                        action, actionStateL1 = self._wait(TERMES_World)
                        actionStateL2 = self.actionStateL2_list[2] # outcome is failure
                        self.workMode = "leave structure"
                        self.workStage = 0
                        self._report_warningMsg(TERMES_World, "Robot left the structure due to localization error (move_to_ground_or_exit).")
                        
                        # Stop execution of the function and return action information
                        return(action, [actionStateL1, actionStateL2])
                    
                # If robot believes that it is on the structure, start nevigation
                else: 
                    # Choose a child location
                    xr, yr = self.position
                    prob_list = []
                    child_loc_list = []
                    if TERMES_World.pathProbMapType == "Optimized":
                        for xp, yp, prob in TERMES_World.pathProbMap[xr, yr]:
                            prob_list.append(prob)
                            child_loc_list.append(np.array([xp + 1, yp + 1]))
                    elif TERMES_World.pathProbMapType == "Normal":
                        for child in self.children:
                            prob_list.append(child["child_prob"])
                            child_loc_list.append(child["child_position"])
                    # Set up goal configuration
                    self.next_loc = self.rng.choice(child_loc_list, 1, p = prob_list)[0]
                    self.next_heading = None
            
            # Enter the next work stage
            self.workStage = 1
            
        ''' Take 1st level action '''
        if self.workStage == 1:
            # Perform motion planning to travel to next loc
            planActionSeq = self.plan_motion(TERMES_World)
            # Take the 1st action
            if len(planActionSeq) > 0: 
                planAction = planActionSeq[0]
                if planAction == 'turn 90deg CW': 
                    action, actionStateL1 = self._turn_90deg(TERMES_World, 'clockwise')
                elif planAction == 'turn 90deg CCW': 
                    action, actionStateL1 = self._turn_90deg(TERMES_World, 'counterclockwise')
                elif planAction == 'move': 
                    # move forward when collision avoidance is not needed or 
                    # when robot tries to break the deadlock and the front location is robot-free
                    action, actionStateL1 = self._avoid_collisions(TERMES_World)
                    if not action or (self.breakDeadlock and not self.robotNearby[0]): 
                        action, actionStateL1 = self._move_forward(TERMES_World)
                else:
                    self._report_errorMsg(TERMES_World, "Unexpected planned action!")
            # If no action is needed, robot waits
            else:
                action, actionStateL1 = self._wait(TERMES_World)
            actionStateL2 = self.actionStateL2_list[1]
            
            # Enter the next work stage
            self.workStage = 2
        
        ''' Check if the robot reaches one of the exits or ground '''
        if self.workStage == 2:
            robot_canLeave, real_canLeave = self._check_leaveConditions(TERMES_World)
            if robot_canLeave: 
                actionStateL2 = self.actionStateL2_list[0]
                self.workMode = "leave structure"
                self.workStage = 0
                
                # Stop execution of the function and return action information
                return(action, [actionStateL1, actionStateL2])
            
            else:
                # Enter the next work stage
                self.workStage = 3
                
        ''' Return action information and set the work stage for next action '''
        if self.workStage == 3:
            # If next position is reached, choose a new one
            if np.array_equal(self.position, self.next_loc): 
                self.workStage = 0
                
            # If robot waits for too long, choose a new one
            elif self.rechoose:
                self.workStage = 0
                
            # Otherwise, skip choosing new next location
            else:
                self.workStage = 1
                
            # Return action information from work stage 1
            return(action, [actionStateL1, actionStateL2])
        
        ''' Check exceptions '''
        self._report_errorMsg(TERMES_World, "Exception (move_to_ground_or_exit)!")


    def place_brick(self, TERMES_World):
        '''
        Robot executes action sequence for placing a brick.
        Notice that the robot can only place a brick on the same level.
        In order to place a brick at the current location, the robot will: 
            0) find next position
            1) move to the next position: move / turn + move
            2) make a 180 degree turn: turn + turn
            3) place the brick: placement

        Parameters
        ----------
        TERMES_World : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        ''' Check if the work mode matches the action '''
        if self.workMode != "place brick":
            self._report_errorMsg(TERMES_World, "Wrong workMode (place_brick)!")
            
        ''' Check if it is needed to stop the placement '''
        # If there is an unresolved action error, 
        # robot abandons this placement process and goes back to "move to ground or exit" or "leave structure"
        if self.unresolvedActionError: 
            action, actionStateL1 = self._wait(TERMES_World)
            actionStateL2 = self.actionStateL2_list[2] # outcome is failure
            self.workStage = 0
            
            # Log critical simulation information to TERMES_World
            warningMsg = "Brick placement is abandoned due to unresolved action error!"
            real_cur_loc = np.copy(TERMES_World.robotConfig_dict[self.name]['position'])
            real_cur_height = TERMES_World.robotConfig_dict[self.name]['height']
            warningInfo = {'eventLoc': {'loc': real_cur_loc, 'height': real_cur_height}}
            TERMES_World.log_criticalSimInfo(self.name, warningMsg, warningInfo)
            
            # If the robot reaches one of the exits or ground, robot leaves structure
            robot_canLeave, real_canLeave = self._check_leaveConditions(TERMES_World)
            if robot_canLeave: 
                self.workMode = "leave structure"
            # If not, robot moves to ground or exit
            else:
                self.workMode = "move to ground or exit"
            
            return(action, [actionStateL1, actionStateL2])
        
        # If robot waits for too long, 
        # robot abandons this placement process and goes back to "move to next" or "leave structure"
        if self.breakDeadlock: 
            action, actionStateL1 = self._wait(TERMES_World)
            actionStateL2 = self.actionStateL2_list[2] # outcome is failure
            self.workStage = 0
            
            # Log critical simulation information to TERMES_World
            warningMsg = "Brick placement is abandoned due to waiting for too long!"
            real_cur_loc = np.copy(TERMES_World.robotConfig_dict[self.name]['position'])
            real_cur_height = TERMES_World.robotConfig_dict[self.name]['height']
            warningInfo = {'eventLoc': {'loc': real_cur_loc, 'height': real_cur_height}}
            TERMES_World.log_criticalSimInfo(self.name, warningMsg, warningInfo)
            
            # If the robot reaches one of the exits or ground, robot leaves structure
            robot_canLeave, real_canLeave = self._check_leaveConditions(TERMES_World)
            if robot_canLeave: 
                self.workMode = "leave structure"
            # If not, robot moves to ground or exit
            else:
                self.workMode = "move to next"
            
            return(action, [actionStateL1, actionStateL2])
            
        ''' Choose the next location '''
        if self.workStage == 0: 
            # If at exit, choose a location outside the structure
            if any(np.array_equal(self.position, exit_loc) for exit_loc in TERMES_World.exit_list): 
                next_loc_list = []
                for heading_vector in self.heading_map.values():
                    sur_loc = self.position + heading_vector
                    # check if the position is outside the structure
                    sur_locInfo = TERMES_World._get_locInfo(sur_loc)
                    if not sur_locInfo['loc_of_struct']: 
                        next_loc_list.append(sur_loc)
                # Set up goal configuration
                self.next_loc = self.rng.choice(next_loc_list)
                self.next_heading = self._find_heading(TERMES_World, self.position - self.next_loc)
            
            # If not at exit, choose a child location
            else:
                # Check if the current location is outside boundary
                CLIB_dict = self._check_robotInBoundary(TERMES_World)
                
                # If robot believes that it is outside the structure, stop the navigation
                if not CLIB_dict['robotInStructure']: 
                    # Check real leaving condition
                    robot_canLeave, real_canLeave = self._check_leaveConditions(TERMES_World)
                    # If real leaving condition is NOT satisfied, raise error.
                    if not real_canLeave: 
                        self._report_errorMsg(TERMES_World, "Robot lost its location on the structure (place_brick)!")
                    # If real leaving condition is satisfied, leave the structure
                    else: 
                        action, actionStateL1 = self._wait(TERMES_World)
                        actionStateL2 = self.actionStateL2_list[2] # outcome is failure
                        self.workMode = "leave structure"
                        self.workStage = 0
                        self._report_warningMsg(TERMES_World, "Robot left the structure due to localization error (place_brick).")
                        
                        # Stop execution of the function and return action information
                        return(action, [actionStateL1, actionStateL2])
                    
                # If robot believes that it is on the structure, start nevigation
                else: 
                    # Choose a child location
                    xr, yr = self.position
                    prob_list = []
                    child_loc_list = []
                    if TERMES_World.pathProbMapType == "Optimized":
                        for xp, yp, prob in TERMES_World.pathProbMap[xr, yr]:
                            prob_list.append(prob)
                            child_loc_list.append(np.array([xp + 1, yp + 1]))
                    elif TERMES_World.pathProbMapType == "Normal":
                        for child in self.children:
                            prob_list.append(child["child_prob"])
                            child_loc_list.append(child["child_position"])
                    # Set up goal configuration
                    self.next_loc = self.rng.choice(child_loc_list, 1, p = prob_list)[0]
                    self.next_heading = self._find_heading(TERMES_World, self.position - self.next_loc)
            
            # Enter the next work stage
            self.workStage = 1
            
        ''' Take 1st level actions to travel to the next location and face the current location '''
        if self.workStage == 1:
            # Perform motion planning to travel to next loc
            planActionSeq = self.plan_motion(TERMES_World)
            # Take the 1st action
            if len(planActionSeq) > 0: 
                planAction = planActionSeq[0]
                if planAction == 'turn 90deg CW': 
                    action, actionStateL1 = self._turn_90deg(TERMES_World, 'clockwise')
                elif planAction == 'turn 90deg CCW': 
                    action, actionStateL1 = self._turn_90deg(TERMES_World, 'counterclockwise')
                elif planAction == 'move': 
                    # move forward when collision avoidance is not needed or 
                    # when robot tries to break the deadlock and the front location is robot-free
                    action, actionStateL1 = self._avoid_collisions(TERMES_World)
                    if not action or (self.breakDeadlock and not self.robotNearby[0]): 
                        action, actionStateL1 = self._move_forward(TERMES_World)
                else:
                    self._report_errorMsg(TERMES_World, "Unexpected planned action!")
                    
                # Return action information
                actionStateL2 = self.actionStateL2_list[1]
                return(action, [actionStateL1, actionStateL2])
                
            # If no action is needed, enter the next work stage
            else:
                self.workStage = 2
        
        ''' Place the brick '''
        if self.workStage == 2:
            # Sense robot before placing the brick
            self._sense_robot(TERMES_World)
                
            # Sense height difference before placing the brick
            self.examine_local(TERMES_World) # Perform a local examination first
            frontHeight = self.localExam_dict['front'][1]
            heightDiff_front = frontHeight - self.height # Compute the front height difference
            
            # If the height difference is not 0 or the front location is docking location, 
            # robot abandons this placement process and enters the next work mode.
            if abs(heightDiff_front) > 0: 
                action, actionStateL1 = self._wait(TERMES_World)
                actionStateL2 = self.actionStateL2_list[2] # outcome is failure
                self.workStage = 0
                # If the robot reaches one of the exits or ground, robot leaves structure
                robot_canLeave, real_canLeave = self._check_leaveConditions(TERMES_World)
                if robot_canLeave: 
                    self.workMode = "leave structure"
                # If not, robot moves to ground or exit
                else:
                    self.workMode = "move to next"
            
            # If there is robot nearby, robot waits
            elif any(self.robotNearby): 
                action, actionStateL1 = self._wait(TERMES_World)
                actionStateL2 = self.actionStateL2_list[1]
                
            # If no height difference and robot nearby, 
            # robot places down the brick and enters the next work mode.
            elif not any(self.robotNearby) and abs(heightDiff_front) == 0: 
                action, actionStateL1 = self._place_down(TERMES_World)
                actionStateL2 = self.actionStateL2_list[0]
                self.workStage = 0
                # If the robot reaches one of the exits or ground, robot leaves structure
                robot_canLeave, real_canLeave = self._check_leaveConditions(TERMES_World)
                if robot_canLeave: 
                    self.workMode = "leave structure"
                # If not, robot moves to ground or exit
                else:
                    self.workMode = "move to ground or exit"
            
            return(action, [actionStateL1, actionStateL2])
        
        ''' Check exceptions '''
        self._report_errorMsg(TERMES_World, "Exception (place_brick)!")
        
        
    def leave_struct(self, TERMES_World):
        '''
        Robot leaves the structure and turns itself OFF.

        Parameters
        ----------
        TERMES_World : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        ''' Check if the work mode matches the action '''
        if self.workMode != "leave structure": 
            self._report_errorMsg(TERMES_World, "Wrong workMode (leave_struct)!")
            
        ''' Check if robot can really leave '''
        robot_canLeave, real_canLeave = self._check_leaveConditions(TERMES_World)
        
        ''' Robot leaves the structure when the real leaving conditions are met or the robot is forced to leave '''
        if real_canLeave: 
            # Take 1st level action
            action, actionStateL1 = self._leave(TERMES_World)
            actionStateL2 = self.actionStateL2_list[0]
            
            # Reset the robot
            self._reset_robotInWorld(TERMES_World)
            
        ''' If not, robot waits and the outcome of this action is failure '''
        if not real_canLeave: 
            # Take 1st level action
            action, actionStateL1 = self._wait(TERMES_World)
            actionStateL2 = self.actionStateL2_list[2]
            
            # Report errors
            self._report_errorMsg(TERMES_World, "Robot tries to leave but leaving conditions are not met.")
        
        return(action, [actionStateL1, actionStateL2])
    
 

# In[]

class Robot_Real(Robot):
    '''
    Realistic robot which could make mistakes.
    In the current version these mistakes happen in 1st level actions.
    '''
    
    def __init__(self, robot_name):
        '''
        Call super class constructors.
        '''
        Robot.__init__(self, robot_name)
        
        '''
        Change robot feature parameters
        '''
        self.type = "real"
        
        '''
        Add new parameters for defining the state of each action
        '''
        # success rate of each action
        self.actionSuccessProb = read_action_prob()
        # possibility of each error given that error happens
        self.actionErrorProb = {"move": {"by 0B": 0.5, "by 2B": 0.5}, 
                                "move up": {"by 0B": 0.67, "by 2B": 0.33}, 
                                "move down": {"by 0B": 0.1, "by 2B": 0.9}, 
                                "turn 90deg CW": {"by 0deg": 0.5, "by 180deg": 0.5}, 
                                "turn 90deg CCW": {"by 0deg": 0.5, "by 180deg": 0.5}}
        # construct probability dictionary of each state of 1st level actions
        self.actionProb = {}
        for action, sprob in self.actionSuccessProb.items():
            self.actionProb[action] = {}
            self.actionProb[action]["success"] = sprob
            if sprob != 1:
                eprob_dict = self.actionErrorProb[action]
                for actionError, eprob in eprob_dict.items():
                    self.actionProb[action][actionError] = eprob * (1 - sprob)
        # construct dictionary of the outcome list of each 1st level action
        self.actionStateL1_dict = {}
        for action, prob_dict in self.actionProb.items():
            self.actionStateL1_dict[action] = [[], []]
            for actionState, prob in prob_dict.items():
                self.actionStateL1_dict[action][0].append(actionState)
                self.actionStateL1_dict[action][1].append(prob)
                
                
    # =============================================================================
    #
    # Add a function for generating the action state for a given action
    # 
    # =============================================================================
    
    
    def _generate_actionStateL1(self, action):
        '''
        Randomly generate action state for 1st level actions based on the given success/error rates

        Returns
        -------
        None.

        '''
        actionState_list = self.actionStateL1_dict[action][0]
        actionStateProb_list = self.actionStateL1_dict[action][1]
        actionState = str(self.rng.choice(actionState_list, 1, p = actionStateProb_list)[0])
        
        return(actionState)
    
    
    # =============================================================================
    #
    # Modify error logging function to allow certain errors.
    # 
    # =============================================================================
    
    
    def _report_errorMsg(self, TERMES_World, errorMsg):
        '''
        This function can be used to log error messages.
        It can also be used to stop the program when error occurs.
        Following errors are allowed (program will not stop when they occur):
            1. deadlock
            2. leaving conditions are not met

        Parameters
        ----------
        errorMsg : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        # Check if the input is not string
        if type(errorMsg) != str:
            self.errorMsg = "Step#" + str(TERMES_World.cur_step) + ": " + self.name + ": " + "Invalid errorMsg type!"
            raise Exception(self.errorMsg)
            
        # Error exemption list
        exemption_list = ["Deadlock!", 
                          "Robot tries to leave but leaving conditions are not met."]
            
        # Ignore errors but record event if they are in the exemption list
        if any([errorMsg == ee for ee in exemption_list]): 
            real_cur_loc = np.copy(TERMES_World.robotConfig_dict[self.name]['position'])
            real_cur_height = TERMES_World.robotConfig_dict[self.name]['height']
            eventInfo = {'eventLoc': {'loc': real_cur_loc, 'height': real_cur_height}}
            TERMES_World.log_criticalSimInfo(self.name, errorMsg, eventInfo)
        
        # If error is not exempted, log and report error message
        else:
            self.errorMsg = "Step#" + str(TERMES_World.cur_step) + ": " + self.name + ": " + errorMsg
            raise Exception(self.errorMsg)
            
            
    # =============================================================================
    #
    # Modify some sensing functions to allow more accurate localization
    # 
    # =============================================================================
    
    
    def _sense_ground(self, TERMES_World): 
        '''
        Ground can be used as a reference since there is no marks on ground.

        Parameters
        ----------
        TERMES_World : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        correction = False # whether the robot has corrected its belief based on the sensing results
        believed_height = self.height
        real_height = TERMES_World.robotConfig_dict[self.name]['height']
        if real_height == 0 and believed_height != 0: 
            self.height = 0 
            correction = True
            
        if correction: 
            self._report_warningMsg(TERMES_World, "Current location has been corrected (_sense_ground)!")
            self._report_warningMsg(TERMES_World, "At least one action error occurred before!")
            self.unresolvedActionError = True # Make the robot aware of an unresolve action error
        
        return(correction)
    
    
    # =============================================================================
    #
    # Modify some computation function to address errors.
    # 
    # =============================================================================
    
    
    def _check_leaveConditions(self, TERMES_World):
        '''
        Check if the robot can leave the structure.
        This function returns two results based on robot's belief and reality respectively.
        This function is modified to allow a looser leaving condition based on reality.
        In reality the robot can leave the structure if any of following conditions are met: 
            1) robot is at any exit.
            2) current height is 0 (robot is on the ground).
            3) robot is outside the structure.
            4) robot is on the accessible margin of the structure

        Parameters
        ----------
        TERMES_World : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        # Get real and believed location and height
        real_loc = np.copy(TERMES_World.robotConfig_dict[self.name]['position'])
        real_height = TERMES_World.robotConfig_dict[self.name]['height']
        robot_loc = np.copy(self.position)
        robot_height = self.height
        
        # Check if the location is within the structure
        real_locInfo = TERMES_World._get_locInfo(real_loc)
        robot_locInfo = TERMES_World._get_locInfo(robot_loc)
        
        # Examine the leaving conditions based on robot's belief
        # If robot is at any exit or the current height is 0 or the robot is outside the structure, robot can leave.
        if (any(np.array_equal(robot_loc, exit_loc) for exit_loc in TERMES_World.exit_list) 
            or robot_height == 0 or not robot_locInfo['loc_of_struct']):
            robot_canLeave = True
        else:
            robot_canLeave = False
            
        # Examine the leaving conditions based on reality
        # If robot is at any exit or the current height is 0 or 
        # the robot is outside the structure or robot is on the accessible margin, robot can leave.
        if (any(np.array_equal(real_loc, exit_loc) for exit_loc in TERMES_World.exit_list) 
            or real_height == 0 or not real_locInfo['loc_of_struct'] or real_locInfo['loc_in_accessMargin']):
            real_canLeave = True
        else:
            real_canLeave = False
            
        return(robot_canLeave, real_canLeave)
    
    
    def check_actionError(self, TERMES_World, actionName, beliefCorrection = False): 
        '''
        Check if there is any action error after the robot takes the given action.
        This function should be executed immediately after the robot takes the action.
        This version uses predictive local checks (PLC) method to detect errors of following actions: 
            move # 0
            move up # 1
            move down # 2
            turn 90deg CW # 3
            turn 90deg CCW # 4
            
        If beliefCorrection = True, robot corrects its current belief based on inference (if there is any).

        Parameters
        ----------
        TERMES_World : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        # Define a function to convert local examination result to a 1D array of heights
        def localExam_dict_to_surHeight_array(localExam_dict): 
            heading_order = ['front', 'left', 'back', 'right']
            surHeight_list = []
            for heading in heading_order: 
                surHeight_list.append(int(localExam_dict[heading][1]))
            surHeight_array = np.array(surHeight_list)
            return(surHeight_array)
        
        # Define a function to check if all elements of a given array are the same
        def check_uniformness(data_array): 
            uniformness = False
            if np.all(data_array == data_array[0]): 
                uniformness = True
            return(uniformness)
        
        # Define function to extract turning angle (CCW) from actionName
        def get_turningAngle(actionName): 
            turningAngle = None
            if actionName in self.actionName_list[3:5]: 
                if actionName == 'turn 90deg CCW': 
                    turningAngle = 90
                elif actionName == 'turn 90deg CW': 
                    turningAngle = 270
            return(turningAngle)
        
        # Define function to map a given angle in degree to [0 270]
        def wrapAngle(angle): 
            # Convert angle to positive value
            if angle >= 0: 
                tt = angle // 360
                angleW = angle - tt * 360
            else:
                tt = abs(angle) // 360
                angleW = angle + (tt + 1) * 360
            # Convert 360deg to 0deg
            if angleW == 360: 
                angleW = 0
            return(angleW)
        
        #######################################################################
        
        # Record checking result and details of the event
        checkResult = {'actionName': actionName, 
                       'errorDetected': None, # None means it is uncertain whether there is an action error
                       'preAction_data': None, 
                       'postAction_data': None, 
                       'inference': None, 
                       'beliefCorrectionSuccess': None} # None means no belief correction has been performed
        
        ''' Perform PLC for turning '''
        # A turning error can be detected only when the height list is not uniform
        if actionName in self.actionName_list[3:5]: 
            preAction_localExam_dict = self.localExamMemory[-1]
            preAction_surHeight_array = localExam_dict_to_surHeight_array(preAction_localExam_dict)
            preAction_surHeight_array_uniformness = check_uniformness(preAction_surHeight_array)
            if not preAction_surHeight_array_uniformness: 
                # Perform local examination to get post-action data
                self.examine_local(TERMES_World, dataMemorization = True)
                postAction_localExam_dict = self.localExamMemory[-1]
                postAction_surHeight_array = localExam_dict_to_surHeight_array(postAction_localExam_dict)
                # Generate all possible outcomes
                predicted_surHeight_array_dict = {}
                for kk, vv in self.permMatrix_dict.items(): 
                    predicted_surHeight_array_dict[kk] = np.matmul(vv, preAction_surHeight_array)
                # Match the post-action result with the prediction list
                matchedOutcome_list = []
                for kk, vv in predicted_surHeight_array_dict.items(): 
                    if np.array_equal(postAction_surHeight_array, vv): 
                        matchedOutcome_list.append(kk)
                # Infer the result
                turnedAngle = get_turningAngle(actionName)
                # Case1: It is not possible that the robot has taken the desired action
                if turnedAngle not in matchedOutcome_list: 
                    checkResult['errorDetected'] = True
                else:
                    # Case2: It is only possible that the robot has taken the desired action
                    if len(matchedOutcome_list) == 1: 
                        checkResult['errorDetected'] = False
                    # Case3: Robot taking the desired action and robot making a mistake are both possible
                    else: 
                        # Case3.1: 'turn 90deg CW' and 'turn 90deg CCW' are possible. 
                        # However, since it is not possible for a robot to make a turn-270-degree error, there should be no error
                        if matchedOutcome_list == [90, 270]: 
                            checkResult['errorDetected'] = False
                        # Case3.2: Desired action and an error are both possible.
                        # Theoretically this case will never occur.
                        else: 
                            checkResult['errorDetected'] = 'NA'
                # Record sensing data
                checkResult['preAction_data'] = preAction_localExam_dict
                checkResult['postAction_data'] = postAction_localExam_dict
                checkResult['inference'] = matchedOutcome_list
                
                # Correct current belief based on inference
                if beliefCorrection: 
                    if checkResult['errorDetected'] and checkResult['inference']: 
                        if len(checkResult['inference']) == 1: 
                            inferredAngle = checkResult['inference'][0]
                            believedAngle = get_turningAngle(actionName)
                            correctionAngle = wrapAngle(inferredAngle - believedAngle)
                            
                            # Find the actual correct heading
                            correct_heading = None
                            if correctionAngle == 90: 
                                correct_heading = self.turning_map_CCW[self.heading]
                            elif correctionAngle == 180: 
                                correct_heading = self.opposite_heading[self.heading]
                            elif correctionAngle == 270: 
                                correct_heading = self.turning_map_CW[self.heading]
                            else:
                                self._report_errorMsg(TERMES_World, "Belief correction fails!")
                                
                            # Verify belief correction
                            real_cur_heading = TERMES_World.robotConfig_dict[self.name]['heading']
                            if real_cur_heading == correct_heading: 
                                checkResult['beliefCorrectionSuccess'] = True
                            else: 
                                checkResult['beliefCorrectionSuccess'] = False
                            
                            # Record event location
                            correctionMsg = "Heading correction occurs: " + self.heading + "->" + correct_heading
                            real_cur_loc = np.copy(TERMES_World.robotConfig_dict[self.name]['position'])
                            real_cur_height = TERMES_World.robotConfig_dict[self.name]['height']
                            checkResult['eventLoc'] = {}
                            checkResult['eventLoc']['loc'] = real_cur_loc
                            checkResult['eventLoc']['height'] = real_cur_height
                            self._report_warningMsg(TERMES_World, correctionMsg, checkResult)
                            
                            # Make correction
                            self.heading = correct_heading
                            
                
        ''' Perform PLC for moving '''
        # A moving error can be detected only when the robot is moving up/down
        # Current detector can only tell us if there is an error. It cannot tell us if the action is correct. 
        if actionName in self.actionName_list[0:3]: 
            preAction_localExam_dict = self.localExamMemory[-1]
            # Perform local examination to get post-action data
            self.examine_local(TERMES_World, dataMemorization = True)
            postAction_localExam_dict = self.localExamMemory[-1]
            preAction_surHeight_array = localExam_dict_to_surHeight_array(preAction_localExam_dict)
            postAction_surHeight_array = localExam_dict_to_surHeight_array(postAction_localExam_dict)
            preAction_surHeightDiff_array = preAction_surHeight_array - preAction_localExam_dict['cur'][1]
            postAction_surHeightDiff_array = postAction_surHeight_array - postAction_localExam_dict['cur'][1]
            # check back height difference of post-action result and front height difference of pre-action result
            if postAction_surHeightDiff_array[2] != -1 * preAction_surHeightDiff_array[0]: 
                checkResult['errorDetected'] = True
            # Record sensing data
            checkResult['preAction_data'] = preAction_localExam_dict
            checkResult['postAction_data'] = postAction_localExam_dict
        
        ''' Report warning messages when error is detected '''
        ########## For test purpose only (start) ##########
        # This test code resets the condition of reporting warning to help the analysis
        # With the test code, system logs the error detection whenever there is an action error
        actionErrorOccurence = False
        if TERMES_World.cur_step in self.warningMsg_dict: 
            for warningMsg in self.warningMsg_dict[TERMES_World.cur_step]: 
                if 'Error occurs with action' in warningMsg: 
                    actionErrorOccurence = True
                    break
        ########## For test purpose only (end) ##########
        # if checkResult['errorDetected'] == True: 
        if actionErrorOccurence: 
            # Record event location and height
            real_cur_loc = np.copy(TERMES_World.robotConfig_dict[self.name]['position'])
            real_cur_height = TERMES_World.robotConfig_dict[self.name]['height']
            checkResult['eventLoc'] = {}
            checkResult['eventLoc']['loc'] = real_cur_loc
            checkResult['eventLoc']['height'] = real_cur_height
            # Report warning. 
            self._report_warningMsg(TERMES_World, 'Error is detected with action: ' + actionName, checkResult)
            
        ''' Make the robot aware of unresolved error '''
        # If an action error is detected but robot did not correct its belief, then the error is unresolved
        if checkResult['errorDetected'] == True and checkResult['beliefCorrectionSuccess'] != True: 
            self.unresolvedActionError = True
            
        return(checkResult)
                

    # =============================================================================
    #
    # Redefine some 1st level robot actions to incorporate errors
    # 
    # =============================================================================
    
    
    def _move_forward(self, TERMES_World):
        '''
        Robot moves forward in the heading direction.
        There are three possible actions:
            a. move on a flat surface
            b. move up for one brick
            c. move down for one brick
            d. wait since the front location is not reachable
        Each action (except wait) has 3 possible states:
            1. correct movement (success)
            2. no move (by 0B)
            3. move too far (by 2B)
        For each action, an action state is generated randomly based on the given error/success rates.
        outcome = action + action state. There are 10 possible outcomes in total.

        Parameters
        ----------
        TERMES_World : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        ''' Get real location and orientation '''
        real_heading = TERMES_World.robotConfig_dict[self.name]['heading']
        real_cur_loc = np.copy(TERMES_World.robotConfig_dict[self.name]['position'])
        real_cur_height = TERMES_World.robotConfig_dict[self.name]['height']
        
        ''' Get the expected next location and height '''
        exp_next_loc = real_cur_loc + self.heading_map[real_heading]            
        exp_next_heightInfo = TERMES_World._get_heightInfo(exp_next_loc)
        exp_next_height = exp_next_heightInfo['cur_height']
        exp_heightDiff = exp_next_height - real_cur_height
        
        ''' Check the height difference and find the corresponding action '''
        action = None
        if exp_heightDiff == 0: # move
            action = self.actionName_list[0]
        elif exp_heightDiff == 1: # move up
            action = self.actionName_list[1]
        elif exp_heightDiff == -1: # move down
            action = self.actionName_list[2]
        else: # wait
            # Robot waits and stops executing this function.
            action, actionState = self._wait(TERMES_World)
            real_cur_loc = np.copy(TERMES_World.robotConfig_dict[self.name]['position'])
            real_cur_height = TERMES_World.robotConfig_dict[self.name]['height']
            eventInfo = {'eventLoc': {'loc': real_cur_loc, 'height': real_cur_height}}
            self._report_warningMsg(TERMES_World, "Robot waits since front location is not reachable.", eventInfo)
            return(action, actionState)
            
        ''' Generate the outcome of the action '''
        actionState = self._generate_actionStateL1(action)
        
        ''' Update TERMES_World based on different action states. '''
        if actionState == "success":
            real_next_loc = np.copy(exp_next_loc)
        elif actionState == "by 0B":
            real_next_loc = np.copy(real_cur_loc)
        elif actionState == "by 2B":
            # Need to check if the location 2B ahead is physically reachable
            next_2B_loc = real_cur_loc + 2 * self.heading_map[real_heading]
            next_1B_loc = real_cur_loc + 1 * self.heading_map[real_heading]
            # Check if these two locations are in map
            next_2B_locInfo = TERMES_World._get_locInfo(next_2B_loc)
            next_2B_heightInfo = TERMES_World._get_heightInfo(next_2B_loc)
            next_1B_heightInfo = TERMES_World._get_heightInfo(next_1B_loc)
            next_heightDiff = next_2B_heightInfo['cur_height'] - next_1B_heightInfo['cur_height']
            # If the location 2B ahead is occupied, robot moves by 1B, actionState is "success", but report warning
            if next_2B_locInfo['loc_has_robot']: 
                real_next_loc = np.copy(next_1B_loc)
                self._report_warningMsg(TERMES_World, "Robot hits another robot!")
                actionState = "success"
            # If the location 2B ahead is higher than the front location, robot moves by 1B, actionState is "success"
            elif next_heightDiff > 0: 
                real_next_loc = np.copy(next_1B_loc)
                actionState = "success"
            # If the location 2B ahead is too low, robot moves by 2B, but report warning
            elif next_heightDiff < -1: 
                real_next_loc = np.copy(next_2B_loc)
                self._report_warningMsg(TERMES_World, "Robot falls from cliff!")
            # If the location 2B ahead is robot free and has reachable height, robot moves by 2B
            else:
                real_next_loc = np.copy(next_2B_loc)
                    

        else:
            self._report_errorMsg(TERMES_World, "Unexpected actionState (_move_foward)!")
            
        # Update robot configuration
        self._update_robotInWorld(TERMES_World, real_next_loc, None)
            
        ''' Robot updates its belief '''
        self.position += self.heading_map[self.heading]
        self.height += exp_heightDiff # assuming that robot can always correctly detect the height difference
        
        ''' Log simulation information when action error occurs '''
        if actionState != "success": 
            eventMsg = 'Error occurs with action: ' + action
            real_cur_loc = np.copy(TERMES_World.robotConfig_dict[self.name]['position'])
            real_cur_height = TERMES_World.robotConfig_dict[self.name]['height']
            eventInfo = {'eventLoc': {'loc': real_cur_loc, 'height': real_cur_height}, 
                         'actionName': action, 
                         'actionState': actionState}
            # Report warning
            self._report_warningMsg(TERMES_World, eventMsg, eventInfo)
        
        return(action, actionState)


    def _turn_90deg(self, TERMES_World, turnDirection):
        '''
        Robot turns by 90 degree clockwise (CW) or counterclockwise (CCW).
        There are two possible actions:
            a. turn 90deg CW
            b. turn 90deg CCW
        Each action (except wait) has 3 possible states:
            1. correct movement (success)
            2. no turn (by 0deg)
            3. turn too far (by 180deg)
        For each action, an action state is generated randomly based on the given error/success rates.
        outcome = action + action state. There are 6 possible outcomes in total.

        Parameters
        ----------
        turnDirection : TYPE
            DESCRIPTION.
        TERMES_World : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        ''' Get real location and orientation '''
        real_cur_heading = TERMES_World.robotConfig_dict[self.name]['heading']
        
        ''' Check turning direction '''
        action = None
        if turnDirection == "clockwise": # turn 90deg CW
            action = self.actionName_list[3]
            exp_next_heading = self.turning_map_CW[real_cur_heading]
        elif turnDirection == "counterclockwise": # turn 90deg CCW
            action = self.actionName_list[4]
            exp_next_heading = self.turning_map_CCW[real_cur_heading]
        else:
            self._report_errorMsg(TERMES_World, "Invalid turnDirection (_turn_90deg)!")
        
        ''' Generate the outcome of the action '''
        actionState = self._generate_actionStateL1(action)
        
        ''' Update TERMES_World based on different outcomes. '''
        if actionState == "success":
            real_next_heading = exp_next_heading
        elif actionState == "by 0deg":
            real_next_heading = real_cur_heading
        elif actionState == "by 180deg":
            real_next_heading = self.opposite_heading[real_cur_heading]
        else:
            self._report_errorMsg(TERMES_World, "Unexpected actionState (_turn_90deg)!")
        self._update_robotInWorld(TERMES_World, np.array([0, 0]), real_next_heading)
            
        ''' Robot updates its belief '''
        robot_cur_heading = self.heading
        if turnDirection == "clockwise": # turn 90deg CW
            self.heading = self.turning_map_CW[robot_cur_heading]
        elif turnDirection == "counterclockwise": # turn 90deg CCW
            self.heading = self.turning_map_CCW[robot_cur_heading]
            
        ''' Log simulation information when action error occurs '''
        if actionState != "success": 
            eventMsg = 'Error occurs with action: ' + action
            real_cur_loc = np.copy(TERMES_World.robotConfig_dict[self.name]['position'])
            real_cur_height = TERMES_World.robotConfig_dict[self.name]['height']
            eventInfo = {'eventLoc': {'loc': real_cur_loc, 'height': real_cur_height}, 
                         'actionName': action, 
                         'actionState': actionState}
            # Report warning
            self._report_warningMsg(TERMES_World, eventMsg, eventInfo)
        
        return(action, actionState)
    
    
    def _place_down(self, TERMES_World):
        '''
        This action does not incorporate errors yet.
        However more checks are added to handle exceptions like IndexError.

        Parameters
        ----------
        TERMES_World : TYPE
            DESCRIPTION.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        # Get real location and orientation.
        real_heading = TERMES_World.robotConfig_dict[self.name]['heading']
        real_cur_loc = np.copy(TERMES_World.robotConfig_dict[self.name]['position'])
        real_cur_height = TERMES_World.robotConfig_dict[self.name]['height']
        
        # If the placement location is in the map, place the brick
        real_next_loc = real_cur_loc + self.heading_map[real_heading]
        real_next_locInfo = TERMES_World._get_locInfo(real_next_loc)
        real_next_heightInfo = TERMES_World._get_heightInfo(real_next_loc)
        if real_next_locInfo['loc_in_map']: 
            # Get height and height difference
            real_next_height = real_next_heightInfo['cur_height']
            real_heightDiff = real_next_height - real_cur_height
            
            # Check if the placement location has the same height.
            if real_heightDiff != 0:
                self._report_errorMsg(TERMES_World, "Location is unplaceable (different height)!")
                
            # Check if the placement location is robot-free.
            if real_next_locInfo['loc_has_robot']: 
                self._report_errorMsg(TERMES_World, "Location is unplaceable (occupied by robot)!")
            
            # Check if the robot has a brick.
            if not self.carry_brick:
                self._report_errorMsg(TERMES_World, "Location is unplaceable (no brick)!")
                
            # Get the action.
            action = self.actionName_list[6]
                
            # Generate the outcome of the action. Currently we assume that every action is successful.
            actionState = "success"
            
            ''' Update TERMES_World based on different outcomes. '''
            if actionState == "success":
                # Check the placement location before placing the brick.
                # Report warnings if the placement action is not valid
                xpr, ypr = real_next_loc
                zpr = int(real_next_heightInfo['cur_height'])
                if real_next_locInfo['loc_of_struct']: 
                    placeability, checkState_list = self._check_placeability_real(TERMES_World, np.copy(real_next_loc))
                    if not placeability: 
                        self._report_warningMsg(TERMES_World, "Wrong placement on structure: " + str([xpr, ypr, zpr]))
                else: 
                    self._report_warningMsg(TERMES_World, "Wrong placement off structure: " + str([xpr, ypr, zpr]))
                    
                # Add the brick
                TERMES_World.add_bricks(real_next_loc)
                TERMES_World.brickPlacementNum += 1 
                
            ''' Robot updates its belief '''
            self.carry_brick = False
            
            return(action, actionState)
        
        # If the placement location is outside the map, robot waits and a warning message is added
        else: 
            
            # Robot waits
            action = self.actionName_list[7]
            actionState = "success"
            
            # Add warning message
            self._report_warningMsg(TERMES_World, "Robot did not place brick since placement location is outside map.")
            
            return(action, actionState)
