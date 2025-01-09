import numpy as np
import matplotlib.pyplot as plt

from CRC_Analysis import memorize_data


# In[]
class consProcessFunc: 
    '''
    This object contains several functions that model the construction process. 
    '''
    
    def __init__(self, 
                 fRA_paraInfo, 
                 fRR_paraInfo): 
        
        '''
        paraInfo: parameter information
            paraInfo = {'funcName': xxx, 'para': (a, b, c, ...)}
        '''
        self.fRA_paraInfo = fRA_paraInfo # parameters of scaled function of brick addition rate vs construction progress
        self.fRR_paraInfo = fRR_paraInfo # parameters of scaled function of brick removal rate vs construction progress
        
        
    # =============================================================================
    #
    # Define the function library
    # 
    # =============================================================================
    
    
    def func1(self, para, x): 
        '''
        The function is b + a*x for x < xc. 
        The function is b + a*xc for x >= xc. 

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        a, b, xc = para
        y = np.piecewise(x, [x < xc, x >= xc], [lambda x: b + a*x, lambda x: b + a*xc])
        
        # Handel scaler input
        if type(x) != np.ndarray: 
            y = float(y)
            
        return(y)
    
    
    def func2(self, para, x): 
        '''
        The function is 0 for x <= 0. 
        The function changes linearly with rate a and intercept b for x > 0. 

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        a, b = para
        y = np.piecewise(x, [x <= 0, x > 0], [lambda x: 0, lambda x: b + a*x])
        
        # Handel scaler input
        if type(x) != np.ndarray: 
            y = float(y)
        
        return(y)
    
    
    def func3(self, para, x): 
        '''
        The function is y1 for x <= xc1. 
        The function changes linearly for xc1 < x < xc2. 
        The function is y2 for x >= xc2. 

        Parameters
        ----------
        para : TYPE
            DESCRIPTION.
        x : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        xc1, xc2, y1, y2 = para
        a = (y2 - y1) / (xc2 -xc1)
        b = y1 - a*xc1
        y = np.piecewise(x, [x <= xc1, ((x > xc1) & (x < xc2)), x >= xc2], [lambda x: y1, lambda x: b + a*x, lambda x: y2])
        
        # Handel scaler input
        if type(x) != np.ndarray: 
            y = float(y)
        
        return(y)
    
    
    # =============================================================================
    #
    # Define construction modeling functions
    # 
    # =============================================================================
        
    
    def fRA(self, x): 
        '''
        Scaled function of brick addition rate vs construction progress. 

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        funcName = self.fRA_paraInfo['funcName']
        para = self.fRA_paraInfo['para']
        
        if funcName == 'func1': 
            y = self.func1(para, x)
        elif funcName == 'func2': 
            y = self.func2(para, x)
        elif funcName == 'func3': 
            y = self.func3(para, x) 
            
        return(y)
    
    
    def fRR(self, x): 
        '''
        Scaled function of brick removal rate vs construction progress. 
        The function is scaled by 1/decay rate. 
        The function is linear but is 0 at 0% cons. prog. 

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        funcName = self.fRR_paraInfo['funcName']
        para = self.fRR_paraInfo['para']
        
        if funcName == 'func1': 
            y = self.func1(para, x)
        elif funcName == 'func2': 
            y = self.func2(para, x)
        elif funcName == 'func3': 
            y = self.func3(para, x) 
            
        return(y)
    
    
    def make_plot(self): 
        '''
        Plot functions. 

        Returns
        -------
        None.

        '''
        datax = np.linspace(0, 1, 1001)
        
        plotNum = 2
        figT, axT = plt.subplots(1, plotNum)
        figT.set_size_inches((3 * plotNum, 3))
        figT.set_dpi(150)
        
        ax = axT[0]
        ax.plot(datax, self.fRA(datax))
        ax.set_xlabel('cons. prog. ')
        ax.set_title('fRA')
        
        ax = axT[1]
        ax.plot(datax, self.fRR(datax))
        ax.set_xlabel('cons. prog. ')
        ax.set_title('fRR')
        
        return(figT, axT)
    
    
# In[]
class timeEvolvingBrickDecayPlanner: 
    '''
    This object plans the decay rate at each time for the time-evolving brick decay. 
    '''
    
    def __init__(self, 
                 totalBrickNum, 
                 initialDecayRate, 
                 consProcessFuncLib): 
        
        # Initialize parameters 
        self.totalBrickNum = totalBrickNum # number of bricks in the blueprint
        self.initialDecayRate = initialDecayRate # initial decay rate
        self.consProcessFuncLib = consProcessFuncLib # functions that model the construction process
        self.cur_step = None
        
        # Initialize decay parameters at t=0
        self.removalRateSum = 0 # sum of removal rate up to t=i-1
        self.addedBrickNum = 1 # number of brick added at t=0
        self.removedBrickNum = 0 # number of brick removed at t=0
        self.structBrickNum = max(0, self.addedBrickNum - self.removedBrickNum) # number of bricks in the structure
        self.consProgress = self.structBrickNum / self.totalBrickNum # cons. prog. at t=0
        self.decayRate = self.initialDecayRate * self.consProcessFuncLib.fRA(self.consProgress) # decay rate to be set at t=0
        self.removalRate = self.decayRate * self.consProcessFuncLib.fRR(self.consProgress) # estimated number of bricks removed at t=0 
        self.removalRateSum += self.removalRate
        
        # log planning history
        self.criticalInfo = {}
        
        
    def update_decayRate(self, addedBrickNum, cur_step=None): 
        '''
        Update the decay rate at t=i, i>0

        Parameters
        ----------
        addedBrickNum : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        self.addedBrickNum = addedBrickNum 
        self.removedBrickNum = np.floor(self.removalRateSum)
        self.structBrickNum = max(0, self.addedBrickNum - self.removedBrickNum)
        self.consProgress = self.structBrickNum / self.totalBrickNum
        self.decayRate = self.initialDecayRate * self.consProcessFuncLib.fRA(self.consProgress)
        self.removalRate = self.decayRate * self.consProcessFuncLib.fRR(self.consProgress)
        self.removalRateSum += self.removalRate 
        
        # If time is provided, register the update event 
        if cur_step: 
            self.cur_step = cur_step
            self.log_criticalInfo()
    
    
    def reset(self): 
        '''
        Reset the planner. 

        Returns
        -------
        None.

        '''
        self.removalRateSum = 0 # sum of removal rate up to t=i-1
        self.addedBrickNum = 1 # number of brick added at t=0
        self.removedBrickNum = 0 # number of brick removed at t=0
        self.structBrickNum = max(0, self.addedBrickNum - self.removedBrickNum) # number of bricks in the structure
        self.consProgress = self.structBrickNum / self.totalBrickNum # cons. prog. at t=0
        self.decayRate = self.initialDecayRate * self.consProcessFuncLib.fRA(self.consProgress) # decay rate to be set at t=0
        self.removalRate = self.decayRate * self.consProcessFuncLib.fRR(self.consProgress) # estimated number of bricks removed at t=0 
        self.removalRateSum += self.removalRate
        
        
    def log_criticalInfo(self): 
        '''
        Log critical information. Following information will be logged: 
            cur_step
            addedBrickNum

        Returns
        -------
        None.

        '''
        # Create keys if missing
        if 'cur_step' not in self.criticalInfo: 
            self.criticalInfo['cur_step'] = []
        if 'addedBrickNum' not in self.criticalInfo: 
            self.criticalInfo['addedBrickNum'] = []
            
        # Log the information
        self.criticalInfo['cur_step'].append(self.cur_step)
        self.criticalInfo['addedBrickNum'].append(self.addedBrickNum)
        
        
    def plot_consProcessFunc(self): 
        '''
        Plot modeling functions

        Returns
        -------
        None.

        '''
        figT, axT = self.consProcessFuncLib.make_plot()
        
        return(figT, axT)
        
        
# In[]
class robotDeploymentPlanner: 
    '''
    This object plans the deployment of robots. 
    '''
    
    def __init__(self, 
                 TERMES_World, 
                 brickPlacementTime = 5): 
        
        # Initialize parameters 
        self.brickPlacementTime = brickPlacementTime 
        self.startLocationSafety_list = []
        self.waitingStartTime = TERMES_World.cur_step 
        
        
    def check_robotDeployment(self, TERMES_World): 
        '''
        Plan the robot deployment. 
        Returned variable decides whether it is good to depoly the robot. 
        When there are multiple robots, this function makes following checks before deploying the robot: 
            1. Check if there is not robot near the start location
            2. If there is no robot, wait for 1x brick placement time 

        Returns
        -------
        None.

        '''
        robotCanBeDeployed = False
        startLocationSafety = TERMES_World.check_start_safety() # check starting location safety
        memorize_data(self.startLocationSafety_list, startLocationSafety, 2)
        
        # Case of multiple robots 
        if len(TERMES_World.robot_list) > 1: 
            # At the start of construction, waiting is not needed
            if self.startLocationSafety_list == [True]: 
                robotCanBeDeployed = True 
            # When starting location is becomes clear, reset the starting time of waiting 
            elif self.startLocationSafety_list == [False, True]: 
                self.waitingStartTime = TERMES_World.cur_step 
            # When starting location keeps clear, count waiting time 
            elif self.startLocationSafety_list == [True, True]: 
                waitingTime = TERMES_World.cur_step - self.waitingStartTime
                if waitingTime >= (1 * self.brickPlacementTime): 
                    robotCanBeDeployed = True 
        # Case of single robot 
        else: 
            # Waiting is not needed 
            if startLocationSafety: 
                robotCanBeDeployed = True 
                
        return(robotCanBeDeployed)
        


