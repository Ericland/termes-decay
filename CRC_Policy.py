import os

import numpy as np
from scipy.optimize import minimize
from collections import deque
import matplotlib.pyplot as plt
import copy
import seaborn as sns
import networkx as nx
from itertools import product
from abc import ABC, abstractmethod
from time import perf_counter

from CRC_Analysis import read_structGraph
from CRC_Visualization import plot_blueprint
from Utility import save_data, load_data


# In[helper functions for compilers]
def count(seq):
    """Count the number of items in sequence that are interpreted as true."""
    return sum(bool(x) for x in seq)


def edge_compatibility(A, a, B, b):
    # check that the same edge is in both the assignments
    if (A, B) in a and (A, B) in b:
        return True
    if (B, A) in a and (B, A) in b:
        return True
    return False


def reachable_from_marked(G, source):
    seen = set([source])
    frontier = [source]
    while len(frontier) > 0:
        node = frontier.pop()
        for child in (c for c in G[node] if G.edges[node, c]['active'] == True):
            if child not in seen:
                seen.add(child)
                frontier.append(child)
    return seen


def reachable_to_marked(G, target):
    seen = set([target])
    frontier = [target]
    while len(frontier) > 0:
        node = frontier.pop()
        for child in (c for c in G[node] if G.edges[c, node]['active'] == True):
            if child not in seen:
                seen.add(child)
                frontier.append(child)
    return seen


def can_topological_sort_marked(G):
    """
    FROM Network X topolgical_sort

    Return a list of nodes in topological sort order.

    A topological sort is a nonunique permutation of the nodes
    such that an edge from u to v implies that u appears before v in the
    topological sort order.

    Parameters
    ----------
    G : NetworkX digraph
        A directed graph

    Notes
    -----
    This algorithm is based on a description and proof in
    The Algorithm Design Manual [1]_ .

    See also
    --------

    References
    ----------
    .. [1] Skiena, S. S. The Algorithm Design Manual  (Springer-Verlag, 1998).
        http://www.amazon.com/exec/obidos/ASIN/0387948600/ref=ase_thealgorithmrepo/
    """

    # nonrecursive version
    seen = set()
    order = []
    explored = set()

    nbunch = G.nodes

    for v in nbunch:  # process all vertices in G
        if v in explored:
            continue
        fringe = [v]  # nodes yet to look at
        while fringe:
            w = fringe[-1]  # depth first search
            if w in explored:  # already looked down this branch
                fringe.pop()
                continue
            seen.add(w)  # mark as seen
            # Check successors for cycles and for new nodes
            new_nodes = []
            for n in (u for u in G[w] if G.edges[w, u]['active'] == True):
                # print('check new nodes')
                if n not in explored:
                    # print( 'Size Explored: ' + str(len(explored)) + '  Size Seen: ' + str(len(seen)))
                    if n in seen:  # CYCLE !!
                        return False
                    new_nodes.append(n)
            if new_nodes:  # Add new_nodes to fringe
                fringe.extend(new_nodes)
            else:  # No new nodes so w is fully explored
                explored.add(w)
                order.append(w)
                fringe.pop()  # done considering this node
    return True


# In[]
class CSP(ABC):
    """This class describes finite-domain Constraint Satisfaction Problems.
    A CSP is specified by the following inputs:
        variables   A list of variables; each is atomic (e.g. int or string).
        domains     A dict of {var:[possible_value, ...]} entries.
        neighbors   A dict of {var:[var,...]} that for each variable lists
                    the other variables that participate in constraints.
        constraints A function f(A, a, B, b) that returns true if neighbors
                    A, B satisfy the constraint when they have values A=a, B=b

    In the textbook and in most mathematical definitions, the
    constraints are specified as explicit pairs of allowable values,
    but the formulation here is easier to express and more compact for
    most cases. (For example, the n-Queens problem can be represented
    in O(n) space using this notation, instead of O(N^4) for the
    explicit representation.) In terms of describing the CSP as a
    problem, that's all there is.

    However, the class also supports data structures and methods that help you
    solve CSPs by calling a search function on the CSP. Methods and slots are
    as follows, where the argument 'a' represents an assignment, which is a
    dict of {var:val} entries:
        assign(var, val, a)     Assign a[var] = val; do other bookkeeping
        unassign(var, a)        Do del a[var], plus other bookkeeping
        nconflicts(var, val, a) Return the number of other variables that
                                conflict with var=val
        curr_domains[var]       Slot: remaining consistent values for var
                                Used by constraint propagation routines.
    The following methods are used only by graph_search and tree_search:
        actions(state)          Return a list of actions
        result(state, action)   Return a successor of state
        goal_test(state)        Return true if all constraints satisfied
    The following are just for debugging purposes:
        nassigns                Slot: tracks the number of assignments made
        display(a)              Print a human-readable representation
    """

    def __init__(self, variables, domains, neighbors, constraints):
        """Construct a CSP problem. If variables is empty, it becomes domains.keys()."""
        variables = variables or list(domains.keys())

        self.variables = variables
        self.domains = domains
        self.neighbors = neighbors
        self.constraints = constraints
        self.initial = ()
        self.curr_domains = None
        self.nassigns = 0

    def assign(self, var, val, assignment):
        """Add {var: val} to assignment; Discard the old value if any."""
        assignment[var] = val
        self.nassigns += 1

    def unassign(self, var, assignment):
        """Remove {var: val} from assignment.
        DO NOT call this if you are changing a variable to a new value;
        just call assign for that."""
        if var in assignment:
            del assignment[var]

    def nconflicts(self, var, val, assignment):
        """Return the number of conflicts var=val has with other variables."""

        def conflict(var2):
            return (var2 in assignment and
                    not self.constraints(var, val, var2, assignment[var2]))

        return count(conflict(v) for v in self.neighbors[var])

    def display(self, assignment):
        """Show a human-readable representation of the CSP."""
        print('CSP:', self, 'with assignment:', assignment)

    # These methods are for the tree and graph-search interface:

    def actions(self, state):
        """Return a list of applicable actions: nonconflicting
        assignments to an unassigned variable."""
        if len(state) == len(self.variables):
            return []
        else:
            assignment = dict(state)
            var = [v for v in self.variables if v not in assignment][0]
            return [(var, val) for val in self.domains[var] if self.nconflicts(var, val, assignment) == 0]

    def result(self, state, action):
        """Perform an action and return the new state."""
        (var, val) = action
        return state + ((var, val),)

    def goal_test(self, state):
        """The goal is to assign all variables, with all constraints satisfied."""
        assignment = dict(state)
        return (len(assignment) == len(self.variables)
                and all(self.nconflicts(variables, assignment[variables], assignment) == 0
                        for variables in self.variables))

    def support_pruning(self):
        """Make sure we can prune values from domains."""
        if self.curr_domains is None:
            self.curr_domains = {v: list(self.domains[v]) for v in self.variables}

    def suppose(self, var, value):
        """Start accumulating inferences from assuming var=value."""
        self.support_pruning()
        removals = [(var, a) for a in self.curr_domains[var] if a != value]
        self.curr_domains[var] = [value]
        return removals

    def prune(self, var, value, removals):
        """Rule out var=value."""
        self.curr_domains[var].remove(value)
        if removals is not None:
            removals.append((var, value))

    def infer_assignment(self):
        """Return the partial assignment implied by the current inferences."""
        self.support_pruning()
        return {v: self.curr_domains[v][0]
                for v in self.variables if 1 == len(self.curr_domains[v])}

    def infer_extra_assignments(self, current_assignments):
        """Return addidinal assignments implied by the current inferences."""
        self.support_pruning()

        return {v: self.curr_domains[v][0]
                for v in self.variables if 1 == len(self.curr_domains[v]) and (v not in current_assignments.keys())}

    def restore(self, removals):
        """Undo a supposition and all inferences from it."""
        for B, b in removals:
            self.curr_domains[B].append(b)

    @abstractmethod
    def check_global(self):
        '''used to implement checking global constraints on the CPS'''
        pass

    @abstractmethod
    def check_partial(self):
        '''used to implement checking partial assignments of the CPS'''
        pass

    # In[]


'''
CSP Forumlaitons, both take in graphs

       CSP
        |
        v
    TERMS_CSP (set up graph stuff)
    /       \
   v         v
 EdgaCSP LocationCSP

'''


class TERMES_CSP(CSP, ABC):
    '''
    Used in the inherited CSP problems to tie the assigned graph to the
    assignments. This class is not meant to be used directely
    '''

    def setup_graph(self, graph, entry_point=None, exit_point=None):
        '''
        Store graph to generate neighbors and do graph-based checks
        '''

        self.graph = graph
        self.num_nodes = len(graph.nodes())

        self.assigned_graph = nx.DiGraph()

        for n in graph.nodes():
            self.assigned_graph.add_node(n)

        if entry_point == None:
            entry_point = min(graph.nodes())

        if exit_point == None:
            exit_point = max(graph.nodes())

        self.entry_point = entry_point
        self.exit_point = exit_point

        # add assigned graph with all edges and atribute FALSE
        # This way dicitionaries hash tables don't need to be re-computed
        for e in graph.edges():
            self.assigned_graph.add_edge(e[0], e[1], active=False)
            self.assigned_graph.add_edge(e[1], e[0], active=False)

        self.global_checks = 0
        self.partial_checks = 0
        self.tried_assignments = 0

    def check_global(self):
        # check the global properties of the assignment

        self.global_checks = self.global_checks + 1

        if not len(reachable_from_marked(self.assigned_graph, self.entry_point)) == self.num_nodes:
            return False
        elif not len(reachable_to_marked(self.assigned_graph, self.exit_point)) == self.num_nodes:
            return False
        return True

    def check_partial(self):
        '''
        1) check for cycles in the assigned graph
        '''

        self.partial_checks = self.partial_checks + 1

        if can_topological_sort_marked(self.assigned_graph):
            return True
        else:
            return False

        # In[]


'''
CSP defined in terms of locations
'''


class LocationCSP(TERMES_CSP):

    def __init__(self, graph, entry_point=None, exit_point=None):

        '''Parse the graph and populate the assigned state varialbes'''
        TERMES_CSP.setup_graph(self, graph, entry_point, exit_point)

        '''
        Add a dicionary entry for each edge to see if it is active due to a particular varialbe
        This is needed, since otherwise unassigning a variable might set an edge 'active'=False
        even when it is set to active by a different variable
        '''
        for e in self.assigned_graph.edges:
            self.assigned_graph.edges[e[0], e[1]][e[0]] = False
            self.assigned_graph.edges[e[0], e[1]][e[1]] = False

        # Varialbes are the graph nodes
        variables = list(self.graph.nodes())

        path_len = nx.shortest_path_length(graph, source=self.entry_point)

        variables.sort(key=(lambda x: path_len[x]))

        neighbors = {}

        for v in variables:
            neighbors[v] = list(self.graph[v].keys())

        domains = {}
        for v in variables:
            # make list of edge choices
            edges = []
            for n in neighbors[v]:
                edges.append([(v, n), (n, v)])

            dom = []
            # loop over possible combinations of edges and check if they are OK
            for a in product(*edges):
                # each a is apossible combiantion of edges that involve v
                if self.dom_ok(v, a):
                    dom.append(a)

            domains[v] = dom

            # find domains
        ## Find neighbors, assign direction for _each_ neighbor
        ## Get rid of non-sensical combinations

        TERMES_CSP.__init__(self, variables, domains, neighbors, edge_compatibility)
        pass

    # this needs to be mdified for multiple exits
    def dom_ok(self, n, el):
        '''
        n is  anode
        el is possible domain element (i.e. list of edges to all neighbors)
        '''

        # check two opposing, incoming edges
        if ((((n[0] - 1, n[1]), n)) in el) and ((((n[0] + 1, n[1]), n)) in el):
            return False

        elif ((((n[0], n[1] - 1), n)) in el) and ((((n[0], n[1] + 1), n)) in el):
            return False

        # all leaving edges
        elif (not n == self.entry_point) and all(e[0] == n for e in el):
            return False

        # all incoming edges
        elif (not n == self.exit_point) and all(e[1] == n for e in el):
            return False

        elif n == self.entry_point and not all(e[0] == n for e in el):
            return False

        elif n == self.exit_point and not all(e[1] == n for e in el):
            return False

        else:
            return True

    '''
    Init the variables as the self.graph nodes
    Init the domains as the possible i/o directions for edes

    Override the assign and un-assign variables to set multiple edges

    '''

    # var is a node, value is list of all edges that var is involved in
    def assign(self, var, val, assignment):

        for e in val:
            self.assigned_graph.edges[e[0], e[1]]['active'] = True
            self.assigned_graph.edges[e[0], e[1]][var] = True

        TERMES_CSP.assign(self, var, val, assignment)

    # wrap unassign funciton from CSP to include graph
    # inactive edges might get un-assigned
    def unassign(self, var, assignment):

        edges = assignment[var]

        for e in edges:
            self.assigned_graph.edges[e[0], e[1]][var] = False
            if not self.assigned_graph.edges[e[0], e[1]][e[0]] and not self.assigned_graph.edges[e[0], e[1]][e[1]]:
                self.assigned_graph.edges[e[0], e[1]]['active'] = False
        TERMES_CSP.unassign(self, var, assignment)

    def unassign_edge(self, var, assignment):

        edges = assignment[var]

        for e in edges:
            self.assigned_graph.edges[e[0], e[1]][var] = False
            if not self.assigned_graph.edges[e[0], e[1]][e[0]] and not self.assigned_graph.edges[e[0], e[1]][e[1]]:
                self.assigned_graph.edges[e[0], e[1]]['active'] = False


# In[]
class Solver:
    global sol_num
    sol_num = 0

    def AC3(self, csp, queue=None, removals=None):

        if queue is None:
            queue = [(Xi, Xk) for Xi in csp.variables for Xk in csp.neighbors[Xi]]
        csp.support_pruning()
        while queue:
            (Xi, Xj) = queue.pop()
            if self.revise(csp, Xi, Xj, removals):
                if not csp.curr_domains[Xi]:
                    return False
                for Xk in csp.neighbors[Xi]:
                    if Xk != Xi:
                        queue.append((Xk, Xi))
        # print(csp.curr_domains)
        return True

    def revise(self, csp, Xi, Xj, removals):
        """Return true if we remove a value."""
        revised = False
        for x in csp.curr_domains[Xi][:]:

            # If Xi=x conflicts with Xj=y for every possible y, eliminate Xi=x
            if all(not csp.constraints(Xi, x, Xj, y) for y in csp.curr_domains[Xj]):
                csp.prune(Xi, x, removals)
                revised = True

        return revised

    def backtracking_search(self, csp, sol_limit=1, timeout=3600):
        global sol_num
        global sol_dir_list
        sol_num = 0
        sol_dir_list = []
        timeStart = perf_counter()

        def backtrack(assignment):
            global sol_num
            csp.tried_assignments = csp.tried_assignments + 1

            if not csp.check_partial() or sol_num >= sol_limit:
                return None

            elif len(assignment) == len(csp.variables):
                # return assignment
                if csp.check_global():
                    save_loc, file_name = save_data(assignment, print_msg=False)
                    sol_dir_list.append((save_loc, file_name))
                    sol_num += 1
                    return None
                    # return assignment
                else:
                    return None

            # Find first un-assigned variable
            for vari in csp.variables:
                if (vari not in assignment):
                    var = vari
                    break

            # Loop through possible values that variable could have
            for value in (csp.curr_domains or csp.domains)[var]:

                if 0 == csp.nconflicts(var, value, assignment):
                    csp.assign(var, value, assignment)
                    removals = csp.suppose(var, value)

                    if self.AC3(csp, [(X, var) for X in csp.neighbors[var]], removals):
                        # add the new inferred assignments
                        # since the needs to be updated, th implicit size 1 domain trick does not work well
                        new_assignments = csp.infer_extra_assignments(assignment)
                        for v in new_assignments:
                            csp.assign(v, new_assignments[v], assignment)

                        result = backtrack(assignment)
                        # result = backtrack(csp.infer_assignment())

                        if result is not None:
                            return result

                        for v in new_assignments:
                            csp.unassign(v, assignment)

                    csp.restore(removals)
                    # mark edge in 'assigned_graph' as inactive
                    csp.unassign_edge(var, assignment)

            csp.unassign(var, assignment)

            timeNow = perf_counter()
            timeRun = timeNow - timeStart
            if timeRun > timeout:
                raise Exception("Timeout: backtracking_search_spec")

            return None

        result = backtrack({})
        assert result is None or csp.goal_test(result)
        return result, sol_dir_list


# In[]:
def optimize_pathProb(
        opt_info,
        approx_digit=3,
        timeout=3600,
        maxiter=100000,
        optThreshold=1,
        showNormal = False, # show the unoptimized visiting rates
        showDetails = False, # show optimization details
        make_plot=False,
        print_info=False,
):
    parents = opt_info['parents']
    children = opt_info['children']
    indices = opt_info['indices']
    startExits = opt_info['startExits']
    optObjFunc = opt_info['optObjFunc']
    exitList = []
    for i in range(1, len(startExits)):
        exitList.append(startExits[i][0])


    # In[]
    # Create the GridInd2VecInd
    gridInd2VecInd = {}
    variableList = []
    siteList = []  # site position in sol.x
    transList = []  # # transition probability position in sol.x
    pos = 0
    for i in range(len(indices)):
        gridKey = str(i + 1)
        # Create the variable index for grid probility
        gridInd2VecInd[gridKey] = pos
        variableList.append(gridKey)
        siteList.append(pos)
        pos += 1
        # Create the variable index for transition probility
        for j in range(1, len(children[i])):
            transitionKey = gridKey + str('_') + children[i][j]
            gridInd2VecInd[transitionKey] = pos
            variableList.append(transitionKey)
            transList.append(pos)
            pos += 1


    # In[Generate Objective Function]
    def objective_NN(x):
        obj = 0
        for subList in disList:
            if len(subList) > 1:
                layer_ref = x[gridInd2VecInd[str(subList[0] + 1)]]
                obj += sum([(x[gridInd2VecInd[str(ind + 1)]] - layer_ref) ** 2 for ind in subList])
        return obj

    def jac_NN(x):
        derivatives = np.zeros(len(x))
        for subList in disList:
            if len(subList) > 1:
                layer_ref = x[gridInd2VecInd[str(subList[0] + 1)]]
                for ind in subList:
                    derivatives[gridInd2VecInd[str(subList[0] + 1)]] -= 2 * x[
                        gridInd2VecInd[str(ind + 1)]] - 2 * layer_ref
                    derivatives[gridInd2VecInd[str(ind + 1)]] = 2 * x[gridInd2VecInd[str(ind + 1)]] - 2 * layer_ref
        return derivatives

    def objective_MVR(x):
        obj = 0
        for ii in range(len(indices)):
            obj += np.exp(alpha * (VR_Ref - x[gridInd2VecInd[str(ii + 1)]]))
        return (obj)

    def jac_MVR(x):
        derivatives = np.zeros(len(x))
        for ii in range(len(indices)):
            derivatives[gridInd2VecInd[str(ii + 1)]] = -alpha * np.exp(
                alpha * (VR_Ref - x[gridInd2VecInd[str(ii + 1)]]))
        return (derivatives)


    # In[]:
    def disFromExit():
        disList = []
        q = deque()
        visited = [False for i in range(len(parents))]
        for exit in exitList:
            q.append(int(exit) - 1)
            visited[int(exit) - 1] = True
        length = 0
        while q:
            length = len(q)
            if length > 0:
                disList.append(copy.deepcopy(q))
            for i in range(length):
                cur = q.popleft()
                for parent in parents[cur][1:]:
                    if visited[int(parent) - 1] == False:
                        q.append(int(parent) - 1)
                        visited[int(parent) - 1] = True
        return disList

    def treeWidthFromStart():
        childrenDict = {}
        for info in children:
            childrenDict[info[0]] = info[1:]

        treeWidth = 1
        treeWidth_list = deque()
        pp_list = deque(['1'])

        while treeWidth > 0:
            cc_list = deque()
            for pp in pp_list:
                cc = childrenDict[pp]
                for c in cc:
                    cc_list.append(c)
            cc_list = deque(list(dict.fromkeys(cc_list)))  # remove duplicates
            treeWidth = len(cc_list)
            treeWidth_list.append(treeWidth)
            pp_list = copy.deepcopy(cc_list)
            # print(cc_list)
        max_TreeWidth = np.amax(treeWidth_list)

        return (max_TreeWidth)


    # In[helper function for generating the constraint]:
    # The helper function for generating the constraint of start point
    # constraint: x[0] = 1
    def startCons(x):
        return 1 - x[gridInd2VecInd[startExits[0][0]]]

    # The helper function for generating parent constraints
    # return a tuple of both the constraint function and the jacobian
    # Input are strings
    # constraint: sum_j( lambda_j * P_ji ) - lambda_i = 0
    def geneParentsCons(gridInd, parentInds):

        def constraints(x):
            cons = 0
            for parentInd in parentInds:
                cons = cons + x[gridInd2VecInd[parentInd]] * x[gridInd2VecInd[parentInd + "_" + gridInd]]
            return cons - x[gridInd2VecInd[gridInd]]

        def jac(x):
            ret = np.zeros(len(x))
            # minus P of grid cell
            ret[gridInd2VecInd[gridInd]] = -1
            for parentInd in parentInds:
                ret[gridInd2VecInd[parentInd + "_" + gridInd]] = x[gridInd2VecInd[parentInd]]
                ret[gridInd2VecInd[parentInd]] = x[gridInd2VecInd[parentInd + "_" + gridInd]]
            return ret

        return constraints, jac

    # The helper function for generating children constraints
    # Input are strings
    # constraint: 1 - sum_i( P_ji ) = 0
    def geneChildrenCons(gridInd, childrenInds):

        def constraints(x):
            cons = 1
            for childInd in childrenInds:
                cons = cons - x[gridInd2VecInd[gridInd + "_" + childInd]]
            return cons

        def jac(x):
            ret = np.zeros(len(x))
            for childInd in childrenInds:
                ret[gridInd2VecInd[gridInd + "_" + childInd]] = -1
            return ret

        return constraints, jac

    def appendCons(consFuncList, con):
        # Generate list of constraints in SciPy format
        # Add either a callable equality constraint or tuple where
        # the second term is a jacobian
        if callable(con):
            consFuncList.append({'type': 'eq', 'fun': con})
        else:
            consFuncList.append({'type': 'eq', 'fun': con[0], 'jac': con[1]})

        return consFuncList


    # In[Generate Constraint Functions]:
    consList = []

    # Add constraint to the start point
    consList = appendCons(consList, startCons)

    # Add two types of constraints
    for i in range(len(indices)):
        gridInd = str(i + 1)

        # Parent constraints apply to all sites except start point
        if gridInd not in startExits[0]:
            parentInds = parents[i][1:]
            consList = appendCons(consList, geneParentsCons(gridInd, parentInds))

        # Children constraints apply to all sites except end points
        if gridInd not in exitList:
            childrenInds = children[i][1:]
            consList = appendCons(consList, geneChildrenCons(gridInd, childrenInds))


    # In[Generate initial conditions and boundaries]
    x0 = np.ones(len(variableList)) * 0.1  # initial guess
    bnds = [(0.001, 1.001) for vv in range(len(variableList))]  # boundaries


    # In[Solve the optimization problem]
    timeStart = perf_counter()  # function starting time

    # Select solver method
    method_list = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'trust-constr', 'SLSQP']
    method = method_list[-1]

    # Set up optimization based on selected objective function
    opt_msg = None
    opt_hist = None

    if optObjFunc == "NN":
        objFunc = objective_NN
        jacFunc = jac_NN
        disList = disFromExit() 
        if print_info: 
            print("Equal-visiting-rate objective function is used.")
            print("disList: ", disList)
            print("Start solving the optimization problem...")
        sol = minimize(objFunc, x0,
                       method=method,
                       bounds=bnds,
                       constraints=consList,
                       options={'maxiter': maxiter},
                       jac=jacFunc)

    elif optObjFunc == "MVR":
        objFunc = objective_MVR
        jacFunc = jac_MVR
        alpha = 10
        delta_alpha = 1
        maxTreeWidth = treeWidthFromStart()
        VR_Ref = 1 / maxTreeWidth
        find_alpha = True
        VRmin_rate_threshold = optThreshold * 0.001
        opt_hist = {'alpha': [], 'success': [], 'VRmin': [], 'VRsum': []} 
        if print_info: 
            print("Minimum-visiting-rate objective function is used.")
            print("VR_Ref: ", VR_Ref)
            print("Start solving the optimization problem...")
        while find_alpha == True: 
            if print_info: 
                print('.', end='')
            sol = minimize(objFunc, x0,
                           method=method,
                           bounds=bnds,
                           constraints=consList,
                           options={'maxiter': maxiter},
                           jac=jacFunc)

            VR_list = []
            for pp in siteList:
                VR_list.append(sol['x'][pp])
            opt_hist['alpha'].append(alpha)
            opt_hist['success'].append(sol['success'])
            opt_hist['VRmin'].append(np.amin(VR_list))
            opt_hist['VRsum'].append(np.sum(VR_list))

            # Automatic tuning of alpha
            if len(opt_hist['alpha']) > 2:
                VRmin_rate = (opt_hist['VRmin'][-1] - opt_hist['VRmin'][-2]) / (
                        opt_hist['alpha'][-1] - opt_hist['alpha'][-2])
                VRmin_rate_pre = (opt_hist['VRmin'][-2] - opt_hist['VRmin'][-3]) / (
                        opt_hist['alpha'][-2] - opt_hist['alpha'][-3])

                if VRmin_rate > VRmin_rate_threshold and opt_hist['success'][-1] == True:
                    # slope is larger than the threshold, increase alpha
                    find_alpha = True
                    delta_alpha = 1
                elif VRmin_rate <= VRmin_rate_threshold and VRmin_rate >= 0 and VRmin_rate_pre > 0 and \
                        opt_hist['success'][-1] == True:
                    # slope is positive but smaller than the threshold, stop finding alpha
                    find_alpha = False
                    opt_msg = "Case #1. Final alpha is: " + str(alpha)
                elif VRmin_rate < 0 and VRmin_rate_pre > 0 and opt_hist['success'][-1] == True:
                    # a local minima/maxima is encountered, choose the alpha leads to the highest minimum visiting rate
                    find_alpha = False
                    alpha_ind = -(np.argmax([opt_hist['VRmin'][-1], opt_hist['VRmin'][-2], opt_hist['VRmin'][-3]]) + 1)
                    alpha = opt_hist['alpha'][alpha_ind]
                    sol = minimize(objFunc, x0, method=method, bounds=bnds, constraints=consList,
                                   options={'maxiter': maxiter}, jac=jacFunc)
                    opt_msg = "Case #2. Final alpha is: " + str(alpha)

                elif VRmin_rate < -1 * VRmin_rate_threshold and opt_hist['success'][-1] == True:
                    # slope is samller than the negative threshold, decrease alpha
                    find_alpha = True
                    delta_alpha = -1
                elif VRmin_rate <= 0 and VRmin_rate >= -1 * VRmin_rate_threshold and VRmin_rate_pre < 0 and \
                        opt_hist['success'][-1] == True:
                    # slope is negative but larger than the negative threshold, stop finding alpha
                    find_alpha = False
                    opt_msg = "Case #3. Final alpha is: " + str(alpha)
                elif VRmin_rate > 0 and VRmin_rate_pre < 0 and opt_hist['success'][-1] == True:
                    # a local minima/maxima is encountered, choose the alpha leads to the highest minimum visiting rate
                    find_alpha = False
                    alpha_ind = -(np.argmax([opt_hist['VRmin'][-1], opt_hist['VRmin'][-2], opt_hist['VRmin'][-3]]) + 1)
                    alpha = opt_hist['alpha'][alpha_ind]
                    sol = minimize(objFunc, x0, method=method, bounds=bnds, constraints=consList,
                                   options={'maxiter': maxiter}, jac=jacFunc)
                    opt_msg = "Case #4. Final alpha is: " + str(alpha)

                else:
                    # for other conditions, go back to initial alpha
                    alpha = 10
                    find_alpha = False
                    sol = minimize(objFunc, x0, method=method, bounds=bnds, constraints=consList,
                                   options={'maxiter': maxiter}, jac=jacFunc)
                    opt_msg = "Case #5. Automatic tuning fails. Final alpha is 10"

            alpha += delta_alpha

            # time the optimization process
            timeNow = perf_counter()
            timeRun = timeNow - timeStart
            if timeRun > timeout:
                raise Exception("Timeout: optimize_pathProb")
            
        if print_info: 
            print(opt_msg)

        if showDetails:
            figmvr, axmvr = plt.subplots(1, 2)
            axmvr[0].plot(opt_hist['alpha'], opt_hist['VRmin'])
            axmvr[0].set_xlabel("alpha")
            axmvr[0].set_ylabel("minimum visiting rate")
            axmvr[1].plot(opt_hist['alpha'], opt_hist['VRsum'])
            axmvr[1].set_xlabel("alpha")
            axmvr[1].set_ylabel("sum of visiting rates")
            figmvr.set_size_inches(14, 5)
            # plt.savefig(file_loc + 'alpha_' + file_tag + '.png')

    else: 
        print("No objective function is selected.")


    # In[Find the optimized transition probabilities and visiting rates of each location]:
    # =================== Construct pathProbMap matrix and transProbDict ===================
    colLen, rowLen = int(indices[-1][0]) + 1, int(indices[-1][1]) + 1
    pathProbMap = np.zeros([colLen, rowLen], dtype=object)
    transProbDict = {}
    exitInd = []
    for ee in exitList:
        exitInd.append(int(ee))

    # add the normalized transition probabilities
    for i in range(len(indices)):
        x, y = int(indices[i][0]), int(indices[i][1])
        gridInd = str(i + 1)
        childrenInds = children[i][1:]
        pathProbMap[x, y] = []
        if (i + 1) not in exitInd:
            tpSum = 0
            for childInd in childrenInds:
                tpSum += sol.x[gridInd2VecInd[gridInd + "_" + childInd]]
            for childInd in childrenInds:
                tpNormalized = np.round((sol.x[gridInd2VecInd[gridInd + "_" + childInd]] / tpSum),
                                        decimals=approx_digit)
                pathProbMap[x, y].append([int(item) for item in indices[int(childInd) - 1]] + [tpNormalized])
                transProbDict[gridInd + "_" + childInd] = tpNormalized

    # make sure for every location, probabilities sum to 1
    for rows in pathProbMap:
        for pathProb in rows:
            if type(pathProb) == list:
                if len(pathProb) > 0:
                    sumProb = 0
                    for pb in pathProb:
                        sumProb += pb[-1]
                    if sumProb != 1:
                        for pb in pathProb:
                            pb[-1] /= sumProb

    # =================== Construct siteProbMap matrix ===================
    m = 0
    n = 0
    for ind in indices:
        m = max(m, int(ind[0]))
        n = max(n, int(ind[1]))
    siteProbMap = np.zeros((m + 1, n + 1))

    for i in range(len(indices)):
        ind = str(i + 1)
        x = int(indices[i][0])
        y = int(indices[i][1])
        siteProbMap[x][y] = sol.x[gridInd2VecInd[ind]]

    if make_plot:
        # Plot the pathProbMap
        fig1, ax1 = plt.subplots(1, 1)
        vline = np.arange(0, rowLen + 1) - 0.5
        hline = np.arange(0, colLen + 1) - 0.5
        ax1.vlines(vline, -0.5, colLen - 0.5, linestyle='dotted', color='grey')
        ax1.hlines(hline, -0.5, rowLen - 0.5, linestyle='dotted', color='grey')

        for i in range(len(children)):
            childrenInds = children[i][1:]
            x = int(indices[i][0])
            y = int(indices[i][1])
            qx = y
            qy = x
            for childInd in childrenInds:
                childx = int(indices[int(childInd) - 1][1])
                childy = int(indices[int(childInd) - 1][0])
                qu = childx - y
                qv = -(childy - x)
                ax1.quiver(qx, qy, qu, qv)
            ax1.scatter(qx, qy, color='b')
            ax1.text(qx - 0.4, qy - 0.4, str(i + 1) + ':' + '(' + indices[i][0] + ',' + indices[i][1] + ')',
                     color='g', ha='left', va='top')

        for i in range(pathProbMap.shape[0]):
            for j in range(pathProbMap.shape[1]):
                if pathProbMap[i, j] == 0:
                    continue
                for x, y, prob in pathProbMap[i, j]:
                    tx = (x + i) / 2
                    ty = (y + j) / 2
                    ax1.text(ty, tx, str(round(prob, 2)), color='r', horizontalalignment='center',
                             verticalalignment='center')
        fig1.set_size_inches([colLen * 1.5, rowLen * 1.5])
        ax1.axis('equal')
        ax1.set_ylim(ax1.get_ylim()[::-1])  # flip y-axis

        # Plot the siteProbMap
        fig2, ax2 = plt.subplots(1, 1)
        sns.heatmap(siteProbMap, annot=True, ax=ax2)
        ax2.set_title('Optimized Visiting Rates')
        ax2.axis('equal')
        fig2.set_size_inches([colLen, rowLen])


    # In[Find the unoptimized visiting rates of each location]
    if showNormal:
        # Create the map converting indeics to coordinates in matrix
        indToCoord = {}
        for i in range(len(indices)):
            indToCoord[str(i + 1)] = list(map(int, indices[i]))

        siteProbMap_Normal = np.zeros([colLen, rowLen], dtype=np.float)
        siteProb = [None] * len(indices)
        siteVisit = [False] * len(indices)

        start = int(startExits[0][0]) - 1
        exits = list(map(lambda x: int(x[0]) - 1, startExits[1:]))

        def calProb(sitePos, localParents):
            if sitePos == start:
                return 1

            thisProb = 0
            for parent in localParents:
                nextSitePos = int(parent) - 1
                if siteVisit[nextSitePos]:
                    thisProb += siteProb[nextSitePos] / (len(children[nextSitePos]) - 1)
                else:
                    nextLocalParents = parents[nextSitePos][1:]
                    siteProb[nextSitePos] = calProb(nextSitePos, nextLocalParents)
                    thisProb += siteProb[nextSitePos] / (len(children[nextSitePos]) - 1)

            siteVisit[sitePos] = True
            # print(sitePos)

            return thisProb

        for exit in exits:
            sitePos = exit
            localParents = parents[sitePos][1:]
            siteProb[sitePos] = calProb(sitePos, localParents)

        for i in range(len(siteProb)):
            x, y = indToCoord[str(i + 1)]
            siteProbMap_Normal[x, y] = siteProb[i]

        # =================== Plot the siteProbMap_Normal ===================
        fig3, ax3 = plt.subplots(1, 1)
        sns.heatmap(siteProbMap_Normal, annot=True, ax=ax3)
        ax3.set_title('Unoptimized Visiting Rates')
        fig3.set_size_inches([colLen, rowLen])
        ax3.axis('equal')


    # In[output]:
    optOutput = {}
    optOutput['pathProbMap'] = pathProbMap
    optOutput['siteProbMap'] = siteProbMap
    optOutput['transProbDict'] = transProbDict
    optOutput['sol'] = sol
    optOutput['opt_msg'] = opt_msg
    optOutput['opt_hist'] = opt_hist

    # For testing only
    # print('min. visiting rate:', np.amin(siteProbMap[siteProbMap > 0]))

    return optOutput


# In[]
def generate_CRC_policy(
        blueprint,
        sol_limit=1, # number of paths to be simulated: int
        optObjFunc='NN', # objective function for optimization ["Uni", "NN", "MVR"]
        optTimeout=60, # timeout for the trans. prob. optimization
        optMaxiter=100000, # max. iteration for trans. prob. optimization
        optThreshold=1, # optimization termination threshold. lower value gives more optimal result
        make_plot=False,
        print_info=False,
):
    '''
    This simulator performs following tasks sequentially:
        1) generate all feasible paths (policies) to navigate through the structure
        2) optimize transition probabilities of generated policies
        3) perform CRC simulations with ideal or non-ideal agents
        4) log all error information for later analysis
    '''

    # In[setup running parameters]
    # start timing
    runTimeStart = perf_counter()


    # In[Get information of the structure.]
    # The padding offset around the structure.
    padding_offset = np.array([1, 1])
    structRow, structCol = blueprint.shape
    # Get padded structure map.
    padding_goal_struct = np.zeros([structRow + 2, structCol + 2])
    padding_goal_struct[1:structRow + 1, 1:structCol + 1] = blueprint
    # Get start location in the padded structure map.
    start = np.array([0, 0]) + padding_offset
    # Docking is at the west of the start.
    docking = start + np.array([0, -1])
    # Docking location has height of 3 so that robot cannot place brick at or travel to the docking location
    padding_goal_struct[docking[0], docking[1]] = 3
    

    # In[generate all tha feasible policies]
    policyInfo_dict = {} # save all information of each policy
    sn = Solver()
    graph, startpt, endpt = read_structGraph(blueprint)
    gridExampleL = LocationCSP(graph, startpt, endpt)
    _, sol_dir_list = sn.backtracking_search(csp=gridExampleL, sol_limit=sol_limit)
    """
    Description of parent and path:
        Each row corresponds to each location
        1st & 2nd column: location coordinate
        3rd column: height (number of bricks) of that location
        4th, 5th, 6th & 7th column:
            For parent: (0/1) whether there is a parent at each direction (NSWE)
            For path: the probability of going to each direction (NSWE).
    Format of PathProb map:
    Row,Columns matching the raw_goal_struct
    Each entry [i][j]
    holds a LIST of tiplets, where the first two elemetns are the i',j' locations of the 
    location that can be reached from i,j and the third element the probability of taking the
    transition, the probabilities in each list should sum to 1. 
    """
    direct_map = [[-1, 0], [1, 0], [0, -1], [0, 1]]  # orientation map of NSWE
    for ss, sol_dir in enumerate(sol_dir_list):
        compilerSolution = load_data(*sol_dir)
        policyInfo_dict[ss] = {}
        policyInfo_dict[ss]['compilerSolution'] = compilerSolution

        # Generate path and parent map
        parent = np.zeros((len(compilerSolution), 7))
        path = np.zeros((len(compilerSolution), 7))
        for i, key in enumerate(compilerSolution):
            parent[i, 0] = key[0]
            parent[i, 1] = key[1]
            path[i, 0] = key[0]
            path[i, 1] = key[1]
            if [parent[i, 0], parent[i, 1]] == [0, 0]:
                parent[0, 3:7] = np.array([0, 0, 0, 0])
        for v_list in compilerSolution.values():
            for v in v_list:
                for row in range(parent.shape[0]):
                    parent[row, 2] = blueprint[int(parent[row, 0])][int(parent[row, 1])]
                    if [parent[row, 0], parent[row, 1]] == [v[1][0], v[1][1]]:
                        p_ind = direct_map.index([v[0][0] - v[1][0], v[0][1] - v[1][1]])
                        parent[row, 3 + p_ind] = 1
                for row in range(path.shape[0]):
                    path[row, 2] = blueprint[int(path[row, 0])][int(path[row, 1])]
                    if [path[row, 0], path[row, 1]] == [v[0][0], v[0][1]]:
                        p_ind = direct_map.index([v[1][0] - v[0][0], v[1][1] - v[0][1]])
                        path[row, 3 + p_ind] = 1
        for row in range(path.shape[0]):
            summ = sum(path[row, 3:7])
            for i in range(3, 7):
                if path[row, i] > 0:
                    path[row, i] = path[row, i] / summ

        # get children Map
        children_map = np.zeros([structRow + 2, structCol + 2], dtype=np.ndarray)
        for row in path:
            position = np.array([row[0], row[1]], dtype=int) + padding_offset
            children_list = []
            for i in range(3, 7):
                if str(row[i]) != "nan" and row[i] > 0:
                    child_position = position + direct_map[i - 3]
                    child_prob = row[i]
                    child_dict = {"position": child_position, "prob": child_prob}
                    children_list.append(child_dict)
            children_map[position[0], position[1]] = children_list
        policyInfo_dict[ss]['children_map'] = children_map

        # get parent Map
        parents_map = np.zeros([structRow + 2, structCol + 2], dtype=np.ndarray)
        for row in parent:
            position = np.array([row[0], row[1]], dtype=int) + padding_offset
            parent_list = []
            for i in range(3, 7):
                if row[i] == 1:
                    parent_position = position + direct_map[i - 3]
                    parent_dict = {"position": parent_position}
                    parent_list.append(parent_dict)
            parents_map[position[0], position[1]] = parent_list
        policyInfo_dict[ss]['parents_map'] = parents_map

        if make_plot:
            plot_blueprint(blueprint, compilerSolution)


    # In[Optimize the path probability]
    if optObjFunc != "Uni":
        for ss, sol_dir in enumerate(sol_dir_list):
            compilerSolution = policyInfo_dict[ss]['compilerSolution']

            # Generate inputs to the trans. prob. optimization
            optInfo = {}
            opt_parents = []
            opt_children = []
            opt_indices = []
            opt_dict_loc2ind = {}
            opt_startExits = [[], []]
            opt_dict_ind2bricks = {}
            for ii, key in enumerate(compilerSolution):
                opt_indices.append(key)
            opt_indices.sort()
            for ii, loc in enumerate(opt_indices):
                opt_dict_loc2ind[loc] = ii + 1
                opt_parents.append([str(ii + 1)])
                opt_children.append([str(ii + 1)])
            for ii, loc in enumerate(opt_indices):
                bn = blueprint[loc]
                opt_dict_ind2bricks[str(ii + 1)] = bn
            for ii, loc in enumerate(opt_indices):
                if loc == startpt:
                    opt_startExits[0].append(str(ii + 1))
                if loc == endpt:
                    opt_startExits[1].append(str(ii + 1))
            for ii, loc in enumerate(opt_indices):
                loc_info = compilerSolution[loc]
                for info in loc_info:
                    if info[0] == loc:
                        opt_children[ii].append(str(opt_dict_loc2ind[info[1]]))
                    elif info[1] == loc:
                        opt_parents[ii].append(str(opt_dict_loc2ind[info[0]]))
            for ii, loc in enumerate(opt_indices):
                opt_indices[ii] = [str(loc[0]), str(loc[1])]

            # Collect and save all the input arguments
            optInfo['parents'] = opt_parents
            optInfo['children'] = opt_children
            optInfo['indices'] = opt_indices
            optInfo['startExits'] = opt_startExits
            optInfo['optObjFunc'] = optObjFunc
            optInfo['bricks'] = opt_dict_ind2bricks
            policyInfo_dict[ss]['optInfo'] = optInfo

            # Optimize trans. prob.
            if print_info: 
                print('Path probability optimization for path' + str(ss) + ' starts!')
            optOutput = optimize_pathProb(
                optInfo,
                approx_digit=2,
                timeout=optTimeout,
                maxiter=optMaxiter,
                optThreshold=optThreshold,
                make_plot=make_plot,
                print_info=print_info,
            )
            policyInfo_dict[ss]['optOutput'] = optOutput

            # Show optimization results
            if not optOutput['sol']['success']:
                raise Exception("Optimization fails.")
            else: 
                if print_info: 
                    print(optOutput['sol']['message'])
                    print('____________________')  # prints line breaks

    # In[Get simulation information]
    for ss, policyInfo in policyInfo_dict.items():
        if 'optOutput' in policyInfo:
            pathProbMap = policyInfo['optOutput']['pathProbMap']
            padded_pathProbMap = np.zeros([structRow + 2, structCol + 2], dtype=object)
            padded_pathProbMap[1:structRow + 1, 1:structCol + 1] = pathProbMap
            pathProbMapType = "Optimized"
        else:
            padded_pathProbMap = np.zeros([structRow + 2, structCol + 2], dtype=object)
            pathProbMapType = "Normal"
        policyInfo['padded_pathProbMap'] = padded_pathProbMap
        policyInfo['pathProbMapType'] = pathProbMapType

    # In[]
    # delete compiler files
    for sol_dir in sol_dir_list:
        os.remove(sol_dir[0] + sol_dir[1])
    # count simulation time
    runTimeEnd = perf_counter()
    runTime = runTimeEnd - runTimeStart 
    if print_info: 
        print("Policy generation is finished in (s): " + str(runTime))
        print('\n' * 2)

    return policyInfo_dict

