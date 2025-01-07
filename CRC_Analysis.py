import numpy as np
import csv
from networkx import Graph
import os

from Utility import get_time_str


# In[]
def read_structGraph(structArray):
    '''
    Generate a graph class from structure numpy array
    '''
    def check(x1, y1, x2, y2):
        # check ends
        if x1 < 0 or x1 >= maxX:
            return False
        if x2 < 0 or x2 >= maxX:
            return False
        if y1 < 0 or y1 >= maxY:
            return False
        if y2 < 0 or y2 >= maxY:
            return False
        # check end point location
        if structure[y1][x1] == 0 or structure[y2][x2] == 0:
            return False
        if not (abs(x1 - x2) + abs(y1 - y2)) == 1:
            assert False, "Two locaitons are not neighbors"
        if abs(structure[y1][x1] - structure[y2][x2]) <= 1:
            return True
        else:
            # print('Non-Traversable Edge')
            return False
    # Start graph generation
    structure = structArray.tolist()
    maxY, maxX = structArray.shape
    startpt = (0, 0)
    endpt = (maxY - 1, maxX - 1)
    graphInput = Graph()
    # Add traversable edges
    for x in range(maxX):
        for y in range(maxY):
            '''
            check if there endges going to 'higher number neighbors
            since we loop through all points, this will be true form one
            of the two endpoints
            '''
            if check(x, y, x + 1, y):
                graphInput.add_edge((y, x), (y, x + 1))
            if check(x, y, x, y + 1):
                graphInput.add_edge((y, x), (y + 1, x))
    # Check if every location is in the graph.
    # If not, there exists at least one non-traversable location
    NTLoc_list = []
    for x in range(maxX):
        for y in range(maxY):
            if structure[y][x] != 0:
                if (y, x) not in list(graphInput.nodes):
                    NTLoc_list.append((y, x))
    if len(NTLoc_list) > 0:
        raise Exception("Blueprint has non-traversable locations: " + str(NTLoc_list))

    return graphInput, startpt, endpt


def read_structArray(structure_dir):
    '''
    Get blueprint as an numpy array
    '''
    structList = []
    with open(structure_dir, 'r') as fh:
        struct_reader = csv.reader(fh, delimiter=',')
        for r in struct_reader:
            structList.append(list(map(int, r)))
    structArray = np.array(structList)

    return structArray


def write_structArray(structArray, file_name='default', folder_dir="blueprints/temp/"):
    '''
    Write an numpy array as a csv file.
    '''
    if file_name == 'default':
        file_name = 'temp' + get_time_str() + '.csv'
    os.makedirs(folder_dir, exist_ok=True)
    structure_dir = folder_dir + file_name
    np.savetxt(structure_dir, structArray, fmt="%d", delimiter=", ") # Write csv files

    return structure_dir


def get_change_of_structure(s0, s1):
    """
    get change of structures from s0 to s1
    """
    d = (s1 - s0).astype(int)
    idx_list = np.argwhere(d).tolist()
    change_list = []
    if len(idx_list) > 0:
        for idx in idx_list:
            change_list.append((*idx, d[*idx])) 

    return change_list 


# In[]
def read_action_time():
    """
    get time of each action
    """
    file_loc = 'data/robot_info/'
    file_name = 'actionTimes.csv'
    action_info = {}
    with open(file_loc + file_name, 'r') as csv_file:
        csv_info = csv.reader(csv_file, delimiter=',')
        for info in csv_info:
            action_info[info[0]] = float(info[1])

    return action_info


def read_action_prob():
    """
    get success probability of each action
    """
    file_loc = 'data/robot_info/'
    file_name = 'actionProbs.csv'
    action_info = {}
    with open(file_loc + file_name, 'r') as csv_file:
        csv_info = csv.reader(csv_file, delimiter=',')
        for info in csv_info:
            action_info[info[0]] = float(info[1])

    return action_info


# In[]
def memorize_data(data_list, new_data, memory_length=None):
    '''
    Add new_data to data_list based on the given memory_length.

    Parameters
    ----------
    data_list : TYPE
        DESCRIPTION.
    new_data : TYPE
        DESCRIPTION.
    memory_length : TYPE, optional
        DESCRIPTION. The default is None.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    # If memory_length is not defined, just add new data
    if not memory_length:
        data_list.append(new_data)
    
    # If memory_length is defined, trim data_list to get desired length
    else:
        # If length is smaller than the setting, add new data to the list
        if len(data_list) < memory_length:
            data_list.append(new_data)
            
        # If length reaches the setting, add new data and remove the oldest one
        elif len(data_list) == memory_length:
            data_list.append(new_data)
            data_list.pop(0)
        
        # Report exceptions
        else:
            raise Exception("Unexpected memory length!")
            
            
def check_consProgress(goal_struct, cur_struct, brickNumIgnored): 
    '''
    Check the construction progress. 

    Returns
    -------
    None.

    '''
    # Compute the effective construction progress
    goalPLB_loc_list = np.transpose(np.nonzero(goal_struct)).tolist()
    goalTotalBrickNum = np.sum(goal_struct)
    correctBrickNum = 0
    for loc in goalPLB_loc_list:
        goalBrickNum = goal_struct[tuple(loc)]
        curBrickNum = cur_struct[tuple(loc)]
        if curBrickNum <= goalBrickNum:
            correctBrickNum += curBrickNum
        else:
            correctBrickNum += goalBrickNum
    consProgressEffective = (correctBrickNum - brickNumIgnored) / (goalTotalBrickNum - brickNumIgnored)
    
    # Compute the construction progress
    curTotalBrickNum = np.sum(cur_struct)
    consProgress = (curTotalBrickNum - brickNumIgnored) / (goalTotalBrickNum - brickNumIgnored)
    
    # return cons. prog. info
    consProgressInfo = {'consProgressEffective': consProgressEffective, 
                        'consProgress': consProgress}
    return(consProgressInfo)


def get_wrongPlacementLoc(rwmsg): 
    '''
    Get the wrong placement location from the warning message

    Parameters
    ----------
    rwmsg : TYPE
        warning message

    Returns
    -------
    None.

    '''
    if 'Wrong placement' not in rwmsg: 
        raise Exception("The warning message does not contain any information about wrong placement!")
    
    coodstr = rwmsg.split(': ')[-1][1:-1]
    coodstr_list = coodstr.split(', ')
    xpw = int(coodstr_list[0])
    ypw = int(coodstr_list[1])
    zpw = int(coodstr_list[2])
    wrongPlacementLoc = np.array([xpw, ypw])
    wrongPlacementHeight = zpw
    
    return(wrongPlacementLoc, wrongPlacementHeight)


def get_wrongAction(rwmsg): 
    '''
    Get the action name from the results of 'check_actionError'

    Parameters
    ----------
    rwmsg : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    if 'Error is detected with action' not in rwmsg: 
        raise Exception("The warning message does not contain any information about wrong action!")
        
    wrongAction = rwmsg.split(': ')[-1]
    
    return(wrongAction)


def generate_errorDistributionMap(errorLoc_list, mapShape): 
    '''
    Generate a numpy array indicating distribution of error occurences

    Parameters
    ----------
    errorLoc_list : TYPE
        DESCRIPTION. list of locations (in numpy array) where error occurs
    mapShape : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    errorMap = np.zeros(mapShape)
    for errorLoc in errorLoc_list: 
        xe, ye = errorLoc
        try: 
            errorMap[xe, ye] += 1
        except: 
            print("Location: " + str(errorLoc) + " is out of bounds!")
        
    return(errorMap)


def model_CAT_idealParallelNxN(N, unit='action count'):
    '''
    Model the CAT of the parallel solution of an NxN flat structure constructed by one ideal robot

    Parameters
    ----------
    N : TYPE
        DESCRIPTION.
    unit : TYPE, optional
        DESCRIPTION. The default is 'action count'.

    Returns
    -------
    CAT_total : an over-approximation of the actual total CAT
    CAT_effective : a close approximation of the effective CAT

    '''
    def CCP(N):
        '''
        Compute the patameter in coupon collector problem

        Parameters
        ----------
        N : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        ccp = 0
        for ii in range(N):
            ccp += 1 / (ii + 1)
        ccp = ccp * N
        return(ccp)
    
    def compute_tId(d):
        '''
        Compute weighted average trip time of all reachable legal assembly locations in one construction cycle

        Parameters
        ----------
        d : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        tId = actionTime_dict['move'] * d + actionTime_dict['turn 90deg CW'] * (2 + d/4) + actionTime_dict['pick up'] * 1
        return(tId)
    
    
    # Get the action time and convert it to desired unit
    actionTime_dict = read_action_time()
    for kk in actionTime_dict: 
        if unit == 'action count': 
            actionTime_dict[kk] = 1
        if unit == 'hour': 
            actionTime_dict[kk] /= 3600
        if unit == 'minute': 
            actionTime_dict[kk] /= 60
        if unit == 'second': 
            pass
    
    # Compute the time of one brick placement sequence
    tB = actionTime_dict['move'] * 1 + actionTime_dict['turn 90deg CW'] * 3 + actionTime_dict['place down'] * 1    
    sPlace = tB * (N**2 - 1)
    
    # Compute the total and effective travel time
    vec_size = 2*(N-1)
    Ex_vec = np.zeros(vec_size)
    t_vec = np.zeros(vec_size)
    NB_vec = np.zeros(vec_size)
    for ii in range(vec_size):
        t_vec[ii] = compute_tId(ii)
        if ii <= N-1:
            Ex_vec[ii] = CCP(ii + 1)
            NB_vec[ii] = ii + 1
        else:
            Ex_vec[ii] = CCP(2 * N - 1 - ii)
            NB_vec[ii] = 2 * N - 1 - ii
    sParallel = np.sum(Ex_vec * t_vec)
    sEffective = np.sum(NB_vec * t_vec)
    
    # Compute the total and effective CAT
    CAT_total = sPlace + sParallel
    CAT_effective = sPlace + sEffective
    
    return(CAT_total, CAT_effective)
