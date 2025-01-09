import numpy as np
import copy
from time import perf_counter

from CRC_Analysis import memorize_data, get_change_of_structure
from CRC_World import World, Eroder_Restorative, Eroder_Smart, Eroder_Stigmergic, Eroder_Probabilistic
from CRC_Strategy import robotDeploymentPlanner
from Utility import print_exception


# In[]
def simulate_crc(
        blueprint,
        policyInfo,
        robot_num=1,
        robot_type='ideal',  # robot type ["ideal", "real"]
        beliefCorrectionPLC=True,
        simStopCondition='stationarity',  # condition of stopping the simulation ['stationarity', 'consProgress']
        performErosion=False,  # perform erosion or not
        eroder_type='restorative',  # type of erosion ["probabilistic", "restorative", "smart", "stigmergic"]
        erosionSetup_dict={},  # detailed erosion setup
        make_plot=False,
        print_info=False,
        rng_seed=None,
):
    if print_info:
        runTimeStart = perf_counter()
    rng = np.random.default_rng(rng_seed)

    # In[get blueprint]
    padding_offset = np.array([1, 1])
    structRow, structCol = blueprint.shape
    padded_blueprint = np.zeros([structRow + 2, structCol + 2])
    padded_blueprint[1:structRow + 1, 1:structCol + 1] = blueprint
    start = np.array([0, 0]) + padding_offset
    exit_list = [np.array([structRow, structCol])]
    docking = start + np.array([0, -1])
    padded_blueprint[docking[0], docking[1]] = 3

    # In[get policy]
    parents_map = policyInfo['parents_map']
    children_map = policyInfo['children_map']
    padded_pathProbMap = policyInfo['padded_pathProbMap']
    pathProbMapType = policyInfo['pathProbMapType']

    # In[Initialize all simulation parameters.]
    TERMES_World = World()  # Initialize the world
    TERMES_World.create_robot(robot_num, robot_type)  # Create robot
    TERMES_World.update_nav_path(parents_map, children_map, padded_pathProbMap,
                                 pathProbMapType)  # Update structure path
    # Initialize structure
    cur_struct = np.zeros(padded_blueprint.shape)
    TERMES_World.update_goal_struct(padded_blueprint, cur_struct, start, exit_list, docking)
    TERMES_World.cur_struct[TERMES_World.start[0], TERMES_World.start[1]] = 1  # Add the initial brick at the start
    TERMES_World.brickPlacementNum += 1
    TERMES_World.cur_struct[TERMES_World.docking[0], TERMES_World.docking[
        1]] = 3  # Add 3 bricks to the docking location so that robot cannot place brick to or travel to the docking location
    TERMES_World.start_struct = np.copy(TERMES_World.cur_struct)
    # Compute the maximum number of actions during one trip
    dimMax = max(TERMES_World.cur_struct.shape) - 3
    tripLengthInActionNum_max = 2 + 4 * dimMax

    # In[Set up erosion process.]
    # Initialize the eroder
    if performErosion:
        # time-evolving decay is False by default
        timeEvolvingDecay_brick = False
        timeEvolvingDecay_robot = False
        # If restorative eroder is chosen, a single eroder is initialized
        if eroder_type == "restorative":
            eroder = Eroder_Restorative(TERMES_World, 'Eroder')
        # Otherwise two eroders are initialized to erode robots and bricks separately
        else:
            # Get erosion setup information
            robot_erosionParameter_dict = {}
            brick_erosionParameter_dict = {}
            if 'consRate' in erosionSetup_dict:
                robot_erosionParameter_dict['consRate'] = erosionSetup_dict['consRate']
                brick_erosionParameter_dict['consRate'] = erosionSetup_dict['consRate']
            if 'errorMap_dict' in erosionSetup_dict:
                robot_erosionParameter_dict['erosionBiasProbMap'] = erosionSetup_dict['errorMap_dict']['RSErrorMap']
                brick_erosionParameter_dict['erosionBiasProbMap'] = erosionSetup_dict['errorMap_dict']['WPCErrorMap']
            if 'erosionSpeedFactor_robot' in erosionSetup_dict:
                robot_erosionParameter_dict['erosionSpeedFactor'] = erosionSetup_dict['erosionSpeedFactor_robot']
            if 'erosionSpeedFactor_brick' in erosionSetup_dict:
                brick_erosionParameter_dict['erosionSpeedFactor'] = erosionSetup_dict['erosionSpeedFactor_brick']
            if 'timeEvolvingDecayInfo_brick' in erosionSetup_dict:
                timeEvolvingDecay_brick = True
                TEBDPlanner = copy.deepcopy(erosionSetup_dict['timeEvolvingDecayInfo_brick'])
            if 'timeEvolvingDecayInfo_robot' in erosionSetup_dict:
                timeEvolvingDecay_robot = True
                TERDPlanner = copy.deepcopy(erosionSetup_dict['timeEvolvingDecayInfo_robot'])
            # Initialize corresponding eroders
            if eroder_type == "probabilistic":
                robot_eroder = Eroder_Probabilistic(TERMES_World, 'EroderRobot')
                brick_eroder = Eroder_Probabilistic(TERMES_World, 'EroderBrick')
            elif eroder_type == "smart":
                robot_eroder = Eroder_Smart(TERMES_World, 'EroderRobot')
                brick_eroder = Eroder_Smart(TERMES_World, 'EroderBrick')
            elif eroder_type == "stigmergic":
                robot_eroder = Eroder_Stigmergic(TERMES_World, 'EroderRobot')
                brick_eroder = Eroder_Stigmergic(TERMES_World, 'EroderBrick')
            else:
                raise Exception("Unexpected eroder type!")
            # Set up eroders
            robot_eroder.update_erosionParameter(robot_erosionParameter_dict)
            brick_eroder.update_erosionParameter(brick_erosionParameter_dict)

    # In[Set up robot deployment planner]
    rdp = robotDeploymentPlanner(TERMES_World)

    # In[Start the construction simulation.]
    simErrorMsg = "None"
    brickPickupNum_list = []
    brickPlacementNum_list = []
    timeWindow = 20 * tripLengthInActionNum_max
    stationarity = False
    robotInfo_list = [] # robot state history
    structureInfo_list = [] # structure state history
    last_structure = np.copy(TERMES_World.start_struct) # used for computing the change of structure
    while True:
        TERMES_World.update_timeStep()
        cur_step = TERMES_World.cur_step
        robotUpdateSequence = rng.permutation(robot_num)  # random robot update sequence
        for robot_ind in robotUpdateSequence:
            robot = TERMES_World.robot_list[robot_ind]
            action = "None"
            actionOutcome_list = ["None", "None"]
            ''' 
            -------------------------------------------------------------------
            Begin defining the decision flow of each robot during each trip.
            -------------------------------------------------------------------
            '''
            try:
                if robot.robot_OnOff == "OFF":
                    # Check robot deployment condition
                    robotCanBeDeployed = rdp.check_robotDeployment(TERMES_World)
                    if robotCanBeDeployed:
                        robot.robot_on(TERMES_World)
                elif robot.robot_OnOff == "ON":
                    # Robot examines its current location
                    robot.examine_local(TERMES_World, dataMemorization=True)
                    # Robot takes an action
                    if robot.workMode == "pick brick":
                        action, actionOutcome_list = robot.pick_brick(TERMES_World)
                    elif robot.workMode == "move to next":
                        action, actionOutcome_list = robot.move_to_next(TERMES_World)
                    elif robot.workMode == "move to ground or exit":
                        action, actionOutcome_list = robot.move_to_ground_or_exit(TERMES_World)
                    elif robot.workMode == "place brick":
                        action, actionOutcome_list = robot.place_brick(TERMES_World)
                    elif robot.workMode == "leave structure":
                        action, actionOutcome_list = robot.leave_struct(TERMES_World)
                    else:
                        simErrorMsg = "Unexpected work mode!"
                        # If the robot has not been reset, perform following post-action examinations
                    if robot.robot_OnOff == "ON":
                        robot.check_actionError(TERMES_World, action, beliefCorrection=beliefCorrectionPLC)  # Check action error
                        robot.check_deadlock(TERMES_World)  # Check deadlocks
                        robot.check_localization(TERMES_World)  # Check localization error
            except Exception as inst:
                # Print exception
                print_exception(inst)
                # If error occurs, record the message.
                if robot.errorMsg == "None":
                    simErrorMsg = "Step#" + str(cur_step) + ": " + "Unknown errors!"
                else:
                    simErrorMsg = robot.errorMsg
            ''' 
            -------------------------------------------------------------------
            End of defining the decision flow of each robot during each trip.
            -------------------------------------------------------------------
            '''
            ''' 
            Stuff that happens once per time step per robot 
            '''
            # Check if error occurs in the simulation
            if simErrorMsg != "None":
                break
        '''
        Stuff that happens once per time-step
        '''
        # record robot states
        robotInfo = {}
        for robot in TERMES_World.robot_list:
            robotConfig = TERMES_World.robotConfig_dict[robot.name]  # get real robot pose
            if robotConfig == {}:
                robot_loc = np.array([0, 0])
                robot_heading = "E"
            else:
                robot_loc = robotConfig["position"]
                robot_heading = robotConfig["heading"]
            robotStates = (
                robot_loc,
                robot_heading,
                robot.carry_brick,
            )
            robotInfo[robot.name] = robotStates
        robotInfo_list.append(robotInfo)
        # record structure states
        changeInfo = get_change_of_structure(last_structure, TERMES_World.cur_struct)
        last_structure = np.copy(TERMES_World.cur_struct)
        structureInfo = (
            cur_step, 
            changeInfo,
        )
        structureInfo_list.append(structureInfo)
        # Check if error occurs in the simulation
        if simErrorMsg != "None":
            break
        # Perform the decay process
        if performErosion:
            # Update the decay rate if the decay is time-evolving
            # Time-evovling brick decay
            if timeEvolvingDecay_brick:
                addedBrickNum = TERMES_World.brickPlacementNum
                TEBDPlanner.update_decayRate(addedBrickNum, cur_step)
                erosionSpeedFactor = TEBDPlanner.decayRate / TEBDPlanner.initialDecayRate
                erosionParameter_dict = {'erosionSpeedFactor': erosionSpeedFactor}
                brick_eroder.update_erosionParameter(erosionParameter_dict)
            # Time-evolving robot decay (not defined yet)
            if timeEvolvingDecay_robot:
                pass
            # Perform decay
            if eroder_type == "restorative" or eroder_type == "ideal":
                eroder.erode(TERMES_World)
            else:
                robot_eroder.erode(TERMES_World, erosionObj='robot')
                brick_eroder.erode(TERMES_World, erosionObj='brick')
        # Compute construction progress
        consProgressInfo = TERMES_World.check_consProgress()
        ''' Define simulation termination conditions '''
        # If the construction is finished without errors, stop simulation
        # This is the strongest condition and it is always checked
        if consProgressInfo['consProgressEffective'] == 1 and consProgressInfo['consProgress'] == 1:
            if print_info:
                print("The construction is finished without errors!")
            break
        if simStopCondition == 'consProgress':
            # If there is error but the construction progress reaches 100%, stop simulation
            # This is a weaker condition since errors are accepted but 100% construction progress is required
            if consProgressInfo['consProgressEffective'] == 1 and consProgressInfo['consProgress'] != 1:
                # Record this in the error message
                simErrorMsg = "Simulation stops since construction progress reaches 100%!"
                if print_info:
                    print(simErrorMsg)
                break
        if simStopCondition == 'stationarity':
            # If the system is stationary in the time window, stop simulation
            # This is one of the weakest conditions since errors are accepted and 100% construction progress is not required
            # If any of the following 2 conditions is statisfied in the given time window, system is stationary:
            #   1) no brick has been picked up (a stronger condition)
            #   2) no brick has been placed (a weaker condition)
            memorize_data(brickPickupNum_list, TERMES_World.brickPickupNum, timeWindow)  # Record pickup history
            memorize_data(brickPlacementNum_list, np.sum(TERMES_World.cur_struct),
                          timeWindow)  # Record placement history
            if len(brickPickupNum_list) == timeWindow and len(brickPlacementNum_list) == timeWindow:
                if all([brickPickupNum_list[0] == ee for ee in brickPickupNum_list]):
                    stationarity = True
                elif all([brickPlacementNum_list[0] == ee for ee in brickPlacementNum_list]):
                    stationarity = True
            if stationarity:
                # Record this in the error message
                simErrorMsg = "Simulation stops due to stationarity!"
                if print_info:
                    print(simErrorMsg)
                break
        if 'consTime' in simStopCondition:
            simStopTime = int(simStopCondition.split(': ')[-1])
            # If the given construction time is reached, stop simulation
            # This is one of the weakest conditions
            if cur_step >= simStopTime:
                # Record this in the error message
                simErrorMsg = "Simulation stops due to timeout!"
                if print_info:
                    print(simErrorMsg)
                break
    '''
    Stuff that happens after the loop is broken
    '''
    # Get the erosion history if there is any
    if performErosion:
        if eroder_type == "restorative" or eroder_type == "ideal":
            eroder_dict = {'eroder': eroder}
        else:
            eroder_dict = {'robot_eroder': robot_eroder, 'brick_eroder': brick_eroder}

        # Get the time evolving decay planner if any
        if timeEvolvingDecay_brick:
            eroder_dict['TEBDPlanner'] = TEBDPlanner
        if timeEvolvingDecay_robot:
            eroder_dict['TERDPlanner'] = TERDPlanner
    else:
        eroder_dict = None

    # In[]
    simInfo = (
        TERMES_World,
        structureInfo_list,
        robotInfo_list,
    )
    if print_info:
        runTime = perf_counter() - runTimeStart
        print("Simulation time (s): " + str(runTime))

    return simInfo