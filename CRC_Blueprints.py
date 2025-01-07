import random
import matplotlib.pyplot as plt
from time import perf_counter
from itertools import combinations
import numpy as np
import os

from CRC_Policy import Solver, LocationCSP
from CRC_Visualization import plot_blueprint, plot_blueprint_3d
from CRC_Analysis import read_structGraph, write_structArray
from Utility import print_exception, get_time_str


# In[]
def check_blueprintValidity(
        blueprint,
        timeout=3600,
        printInfo=False,
):
    '''
    Check if the blueprint is valid.
    A valid blueprint has at least one solution and is fully connected.

    More analysis can be done to verify the blueprint.
    The function backtracking_search_spec has difficulty with following invalid blueprints:
        1) Blueprints with dangling component. E.g. a blueprint that has one part (not containing the exit)
           connected to the main body through only one node (if that node is removed, the graph is disconnected).
           Following functions from networkx can be used to check dangling components:
               minimum_st_edge_cut
               minimum_edge_cut

    backtracking_search_spec has a timeout feature to solve these unexpected issues.
    '''
    # Two functions will examine the blueprint:
    #   read_structure
    #   backtracking_search_spec
    blueprintValidity = False
    try:
        # Find a policy
        # If there is a policy found, the blueprint is valid
        sn = Solver()
        gridExampleL = LocationCSP(*read_structGraph(blueprint))
        if printInfo:
            print("Start validation")
        _, sol_dir_list = sn.backtracking_search(gridExampleL, sol_limit=1, timeout=timeout)
        if len(sol_dir_list) > 0:
            blueprintValidity = True
            for sol_dir in sol_dir_list:
                os.remove(sol_dir[0] + sol_dir[1])
        else:
            if printInfo:
                print("Blueprint is invalid")
    except Exception as inst:
        if printInfo:
            print("Blueprint is invalid")
            print_exception(inst)

    if blueprintValidity:
        if printInfo:
            print("Blueprint is valid")

    return blueprintValidity


def generate_random_blueprint(
        dimension,
        num_brick='random',
        max_attempt=10,
        save_data=False,
        make_plot=False,
        print_info=False,
        rng_seed=None,
):
    num_attempt = 0
    rng = np.random.default_rng(rng_seed)
    row_num, col_num = dimension
    rfg = Random_Footprint_Generator(dimension, rng_seed=rng_seed)
    if num_brick == 'random':
        num_brick = int(row_num * col_num * (1 + rng.random()))
    holeSizeSum = int(2 * num_brick / (row_num * col_num))
    bp_success = False
    while not bp_success:
        try:
            num_attempt += 1
            max_height = int(2 * num_brick / (row_num * col_num))
            fp_list = rfg.generate_footprints(holeSizeSum=holeSizeSum)
            rbg = Random_Blueprint_Generator(num_brick, fp_list[0], max_height)
            bp = rbg.gen_valid_blueprint(timeout=10)
            bp_success = True
        except Exception as inst:
            if num_attempt > max_attempt:
                print('All attempts fail.')
                bp = None
                break
            if num_brick > row_num * col_num:
                num_brick -= 1
            if holeSizeSum > 1:
                holeSizeSum -= 1
            if print_info:
                print('Blueprint generation fails. Try a new footprint.')
                print_exception(inst)
    if bp_success and save_data:
        file_name = str(row_num) + 'x' + str(col_num) + '_' + get_time_str()
        write_structArray(bp, file_name, folder_dir='blueprints/')
    if bp_success and make_plot:
        plot_blueprint(bp)
        plot_blueprint_3d(bp)

    return bp


# In[]
class Random_Blueprint_Generator:
    """
    A class for generating "random" blueprints. There is a restriction on the
    number of bricks to have in the structure and the dimension of the structure
    """

    def __init__(self,
                 n_bricks,
                 footprint,
                 max_height,
                 blueprint_dir="blueprints/random/",
                 min_directions_with_one_diff=4,
                 require_one_brick=True,
                 ):
        self.n_bricks = n_bricks
        self.dimension = footprint.shape
        self.min_directions_with_one_diff = min_directions_with_one_diff
        self.require_one_brick = require_one_brick
        self.max_height = max_height
        self.footprint = footprint
        self.start = [0, 0]
        self.exit = [self.dimension[0] - 1, self.dimension[1] - 1]
        self.startExit = np.asarray([self.start, self.exit])
        self.blueprint_dir = blueprint_dir
        os.makedirs(self.blueprint_dir, exist_ok=True)
        self.color_map = {0: 6 / 6 * np.array([1, 1, 1]),
                          1: 5 / 6 * np.array([1, 1, 1]),
                          2: 4 / 6 * np.array([1, 1, 1]),
                          3: 3 / 6 * np.array([1, 1, 1]),
                          4: 2 / 6 * np.array([1, 1, 1]),
                          5: 1 / 6 * np.array([1, 1, 1]),
                          6: 0 / 6 * np.array([1, 1, 1])}
        self.old_blueprints = []

    def gen_valid_blueprint(self, base_name='default', timeout=100, make_plot=False):
        """ Generates a single valid blueprint"""
        # loops tries to randomly generate blueprints and checks if they are
        # valid. the blueprint is saved in the blueprint directory with the
        # file name equal to base_name
        if base_name == 'default':
            base_name = 'r' + str(self.dimension[0]) + 'x' + str(self.dimension[1]) + '_' + get_time_str()
        timeStart = perf_counter()  # function starting time
        valid = False
        while not valid:
            blueprint = self.gen_pseudo_random_blueprint(timeout=timeout)
            valid = check_blueprintValidity(blueprint, timeout=2)
            valid = valid and not self.is_old_blueprint(blueprint)

            # time the function execution
            timeNow = perf_counter()
            timeRun = timeNow - timeStart
            if timeRun > timeout:
                raise Exception("Timeout: gen_valid_blueprint")
        self.old_blueprints.append(blueprint)
        if make_plot:
            plot_blueprint(blueprint)

        return blueprint

    def gen_pseudo_random_blueprint(self, timeout=10):
        """ Generates a random blueprint that satisfies the specs but may not
        have a valid path. Makes the assumption that the start and end bricks
        are the upper left and lower right corners for now.
        Tries to make more likely to be valid by
        requiring that the brick has at least min_directions_with_one_diff
        directions with height differences less than 1"""
        timeStart = perf_counter()  # function starting time

        # If the blueprint needs to have at least one brick at each location
        if self.require_one_brick:
            blueprint = np.ones(self.dimension, dtype=int)
        else:
            blueprint = np.zeros(self.dimension, dtype=int)

        # Checks the footprint and sets anything outside the footprint to 0
        blueprint[blueprint > self.footprint] = 0

        blueprint[0, 0] = 1
        blueprint[self.dimension[0] - 1, self.dimension[1] - 1] = 1

        # Places bricks until the required number of bricks is placed.
        # Places a brick if it is a potentially valid location
        while np.sum(blueprint) < self.n_bricks:
            valid_loc = True
            potential_locs = np.where(blueprint > 0)
            # expanded_potential_locs = [[loc[0], loc[1]] for loc in potential_locs]
            potential_locs = np.append(potential_locs[0][:, np.newaxis], potential_locs[1][:, np.newaxis], axis=1)
            expanded_potential_locs = np.copy(potential_locs)
            # expanded_potential_locs = np.rot90(expanded_potential_locs)
            offsets = np.array([[0, 1],
                                [0, -1],
                                [1, 0],
                                [-1, 0]])
            for row in potential_locs:
                for offset in offsets:
                    test_loc = row + offset
                    if not np.any(np.all(expanded_potential_locs == test_loc, axis=1), axis=0) \
                            and 0 <= test_loc[0] < self.dimension[0] \
                            and 0 <= test_loc[1] < self.dimension[1]:
                        expanded_potential_locs = np.append(expanded_potential_locs, test_loc[np.newaxis, :], axis=0)
            n_locs = len(expanded_potential_locs)
            idx = random.randint(1, n_locs - 1)
            # n = idx % self.dimension[0]
            # m = int(idx / self.dimension[0])
            loc = expanded_potential_locs[idx]
            n = loc[0]
            m = loc[1]

            # checks that there will not be huge steps
            new_h = blueprint[n, m] + 1
            north = blueprint[max(0, n - 1), m]
            south = blueprint[min(self.dimension[0] - 1, n + 1), m]
            east = blueprint[n, max(0, m - 1)]
            west = blueprint[n, min(self.dimension[0] - 1, m + 1)]
            directions_with_one_diff = int(np.abs(new_h - north) <= 1) + \
                                       int(np.abs(new_h - south) <= 1) + \
                                       int(np.abs(new_h - east) <= 1) + \
                                       int(np.abs(new_h - west) <= 1)

            if directions_with_one_diff < self.min_directions_with_one_diff:
                valid_loc = False
            if new_h > self.max_height:
                valid_loc = False
            if self.footprint[n, m] != 1:
                valid_loc = False
            if n == self.dimension[0] - 1 and m == self.dimension[1] - 1:
                valid_loc = False

            if valid_loc:
                blueprint[n, m] += 1

            # time the function execution
            timeNow = perf_counter()
            timeRun = timeNow - timeStart
            if timeRun > timeout:
                raise Exception("Timeout: gen_pseudo_random_blueprint")

        # print("pseudorandom blueprint:\n{}".format(np.array2string(blueprint)))
        return blueprint

    def is_old_blueprint(self, blueprint):
        """Checks if a blueprint has already been created"""
        for b in self.old_blueprints:
            if np.array_equal(b, blueprint):
                return True
        return False


# In[]
class Random_Footprint_Generator:
    def __init__(
            self,
            dimension=(10, 10), # tuple
            startLoc='default',
            exitLoc='default',
            rng_seed=None,
    ):
        self.dimension = dimension
        self.rowNum, self.colNum = dimension
        if startLoc == 'default':
            self.startLoc = np.array([0, 0])
        else:
            self.startLoc = np.copy(startLoc)
        if exitLoc == 'default':
            self.exitLoc = np.array([self.rowNum-1, self.colNum-1])
        else:
            self.exitLoc = np.copy(exitLoc)
        self.rng = np.random.default_rng(rng_seed)
        '''
        Pool of different holes. 
        Categorization: size -> shape
        '''
        holePool1B_dict = {1: np.array([[1]])}
        holePool2B_dict = {1: np.array([[1,1]])}
        holePool3B_dict = {1: np.array([[1,1,1]]),
                           2: np.array([[1,1],[1,0]])}
        holePool4B_dict = {1: np.array([[1,1,1,1]]),
                           2: np.array([[1,1,1],[0,1,0]]),
                           3: np.array([[1,1,1],[0,0,1]]),
                           4: np.array([[1,1],[1,1]])}
        self.holePool_dict = {1: holePool1B_dict,
                              2: holePool2B_dict,
                              3: holePool3B_dict,
                              4: holePool4B_dict}


    def rotate_2DMatrix_by_90deg(self, matrix2D):
        '''
        Rotate a 2D matrix by 90deg clockwise.
        '''
        # Transpose given matrix
        matrix2DT = matrix2D.transpose()

        # Flip the order of the column
        colNum = matrix2DT.shape[1]
        matrix2DR = matrix2DT[:, np.flip(np.arange(colNum))]

        return (matrix2DR)


    def rotate_2DMatrix_by_90deg_xTimes(self, matrix2D, times=1):
        '''
        Rotate a 2D matrix by 90deg clockwise multiple times.
        '''
        for tt in range(times):
            matrix2D_copy = np.copy(matrix2D)
            matrix2D = self.rotate_2DMatrix_by_90deg(matrix2D_copy)

        return (matrix2D)


    def gene_setWithFixedSum(self, setSum, setSize, ordered=False):
        '''
        Generate all distinct solutions to: x1+x2+...+xn=k, where xi∈{0,1,2,3,...}, where:
            n = setSize
            k = setSum
        '''
        combSet_list = list(combinations(range(setSize + setSum - 1), setSize - 1))
        numSet_list = []
        for combSet in combSet_list:
            combSetList = list(combSet)
            combSetList.insert(0, -1)
            combSetList.append(setSize + setSum - 1)
            numSet = []
            for ii in range(len(combSetList) - 1):
                numSet.append(combSetList[ii + 1] - combSetList[ii] - 1)

            if not ordered:
                numSet.sort()
                if numSet not in numSet_list:
                    numSet_list.append(numSet)
            else:
                numSet_list.append(numSet)

        return (numSet_list)


    def gene_product(self, prodSpace_list):
        '''
        Generate a list of Cartesian product of input iterables.
        product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
        product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
        '''
        pools = [tuple(pool) for pool in prodSpace_list]
        result = [[]]
        for pool in pools:
            result = [x + [y] for x in result for y in pool]
        for prod in result:
            yield tuple(prod)


    def get_combinations_of_holes(self, holeSizeSum):
        '''
        Generate all distinct combinations of hole shapes
        '''
        # Select number of holes
        holeSizeMax = np.amax(list(self.holePool_dict.keys()))
        holeSizeMin = np.amin(list(self.holePool_dict.keys()))
        holeNumMin = np.ceil(holeSizeSum / holeSizeMax)
        holeNumMax = holeSizeSum
        holeNumCandidate_list = np.arange(start=holeNumMin, stop=holeNumMax + 1, dtype=int)

        # Generate all distinct sets of hole shapes encoded by keys of holePool_dict
        holeShapeKeySet_list = []
        for holeNum in holeNumCandidate_list:
            # Select combinations of hole sizes
            numSet_list = self.gene_setWithFixedSum(holeSizeSum - holeNum, holeNum)
            holeSizeSetCandidate_list = []
            for numSet in numSet_list:
                numSetArray = np.array(numSet) + 1
                if not np.any(numSetArray > holeSizeMax):
                    holeSizeSetCandidate_list.append(numSetArray)

            # Select set of shapes based on the set of hole sizes
            for holeSizeSet in holeSizeSetCandidate_list:
                # Generate a list of choices for each hole size from the pool
                holeShapeChoiceList_list = []
                for holeSize in holeSizeSet:
                    holeShapeChoiceList_list.append(list(self.holePool_dict[int(holeSize)].keys()))

                # Generate all distinct sets of hole shapes encoded by the keys of the hole pool dictionary
                holeShapeIndSet_list = list(self.gene_product(holeShapeChoiceList_list))
                for holeShapeIndSet in holeShapeIndSet_list:
                    holeShapeKeySet = []
                    for ii in range(len(holeSizeSet)):
                        holeShapeKeySet.append((holeSizeSet[ii], holeShapeIndSet[ii]))
                    holeShapeKeySet.sort()
                    if holeShapeKeySet not in holeShapeKeySet_list:
                        holeShapeKeySet_list.append(holeShapeKeySet)

        # Generate all distinct combinations of hole shapes in numpy array
        holeShapeSet_list = []
        for holeShapeKeySet in holeShapeKeySet_list:
            holeShapeSet = []
            for holeShapeKey in holeShapeKeySet:
                holeSizeInd = holeShapeKey[0]
                holeShapeInd = holeShapeKey[1]
                holeShapeSet.append(self.holePool_dict[holeSizeInd][holeShapeInd])
            holeShapeSet_list.append(holeShapeSet)

        return holeShapeSet_list


    def generate_footprints(self, fp_num=1, holeSizeSum='random', make_plot=False):
        if holeSizeSum == 'random':
            holeSizeSum = self.rng.choice(int(self.colNum*self.rowNum*0.05))
        if holeSizeSum > 0: 
            holeShapeSet_list = self.get_combinations_of_holes(holeSizeSum)
            footprint_list = []
            for idx in range(fp_num):
                validFootprint = False
                holeShapeSet = holeShapeSet_list[self.rng.choice(len(holeShapeSet_list))]
                while not validFootprint:
                    ''' Choose a set of hole shapes and choose their locations '''
                    holeLoc_list = []
                    for hh, holeShape in enumerate(holeShapeSet):
                        # Randomly rotate the shape
                        holeShapeRotated = self.rotate_2DMatrix_by_90deg_xTimes(holeShape, times=self.rng.choice(3) + 1)
                        locArray = np.argwhere(holeShapeRotated > 0)
                        # Randomly choose the hole location
                        validHoleLoc = False
                        while not validHoleLoc:
                            validHoleLoc = True
                            holeLoc = [self.rng.choice(self.rowNum), self.rng.choice(self.colNum)]
                            locArray_New = np.copy(locArray)
                            for aa in locArray_New:
                                aa += holeLoc
                                # Examine the new location. A valid hole location meets following conditions:
                                #   inside the map
                                #   not the start location or the exit location
                                #   does not repeat
                                if (aa[0] >= self.rowNum or aa[1] >= self.colNum
                                        or np.array_equal(aa, self.startLoc) or np.array_equal(aa, self.exitLoc)
                                        or aa.tolist() in holeLoc_list):
                                    validHoleLoc = False
                                    break
                        # Add hole locations
                        holeLoc_list += locArray_New.tolist()
                    ''' Create the footprint and check its validity '''
                    footprint = np.ones(self.dimension)
                    for loc in holeLoc_list:
                        footprint[loc[0], loc[1]] = 0
                    validFootprint = check_blueprintValidity(footprint, timeout=2)
                    # store the footprint
                    if validFootprint:
                        footprint_list.append(footprint)
                        if make_plot:
                            fig, ax = plt.subplots(1, 1)
                            ax.matshow(footprint)
                            ax.set_title(str(idx))
                            plt.show() 
        else: 
            footprint_list = [np.ones(self.dimension)] 

        return footprint_list













