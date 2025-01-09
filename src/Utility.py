# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 13:01:56 2023

@author: ericl
"""

# from pymunk import Vec2d
import numpy as np
import scipy 
import matplotlib.pyplot as plt
# import time
# import pandas as pd
import datetime
import pickle
import cv2
from PIL import Image
import os
from functools import reduce


# In[]
def get_kwargs(kwargs, varName, varDefault=None):
    '''
    Get plotting variables from kwargs

    Parameters
    ----------
    varName : str
        DESCRIPTION.
    varDefault : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    '''
    if varName in kwargs:
        var = kwargs[varName]
    else:
        var = varDefault
    
    return var


# In[]
def make_video(
        image_loc='video/video_image/', 
        save_loc='video/', 
        file_prefix='img', 
        file_extension='.png', 
        fps=15, 
        video_name='default', 
        video_format='mp4', 
        delete_images_afterwards=False,
        ):
    """
    file naming rule: file_prefix + '_' + img_number + file_extension
    default file naming: img_xxx.png
    """
    if video_name == 'default': 
        file_time = get_time_str()
        video_name = 'video_' + file_time + '.' + video_format 
    
    # get all image file names 
    ff_list = []
    for ff in os.listdir(image_loc):
        if ff.endswith(file_extension) and ff[0:len(file_prefix)]==file_prefix:
            ff_list.append(ff)
            
    # get img_number 
    img_number_list = []
    for ff in ff_list:
        img_number = int(ff[len(file_prefix)+1:-len(file_extension)])
        img_number_list.append(img_number)
    img_number_list.sort()
    img_number_min = img_number_list[0]
    
    # sort image based on img_number 
    ff_list_sorted = [None] * len(ff_list)
    for ff in ff_list:
        img_number = int(ff[len(file_prefix)+1:-len(file_extension)])
        ff_list_sorted[img_number - img_number_min] = ff
    
    # generate mp4 
    if video_format == 'mp4': 
        # get video frame size 
        img0 = cv2.imread(image_loc + ff_list_sorted[0])
        height, width, layers = img0.shape
        size = (width, height)
        # write video 
        vid = cv2.VideoWriter(save_loc + video_name, cv2.VideoWriter_fourcc(*'MP4V'), fps, size)
        for ff in ff_list_sorted:
            vid.write(cv2.imread(image_loc + ff))
        vid.release()
        cv2.destroyAllWindows()
        
    # generate gif
    elif video_format == 'gif': 
        img_paths = [image_loc + ff for ff in ff_list_sorted] 
        imgs = [Image.open(img_path) for img_path in img_paths] 
        imgs[0].save(
            save_loc + video_name, 
            save_all=True, 
            append_images=imgs[1:], 
            optimize=True, 
            duration=int(1000/fps), # duration of each frame
            loop=0, # loop video 
            )
    
    # delete images after video generation 
    if delete_images_afterwards:
        for ff in ff_list_sorted:
            os.remove(image_loc + ff)
        print('Video images are deleted!')
        
    return save_loc, video_name


# In[]
def join_images(folder_loc, file_tag_list=None, file_name='imgJoined', delImg=False, sort_file=False, **kwargs):
    """
    Join all images in the given folder.
    """
    if not file_tag_list:
        file_tag_list = os.listdir(folder_loc)

    if sort_file:
        ''' sort file name like "xxx (7)" '''

        def sortFunc(ff): return (int(ff[ff.find('(') + 1:ff.find(')')]))

        file_tag_list.sort(key=sortFunc)

    ll = len(file_tag_list)

    if 'rowNum' in kwargs and 'colNum' in kwargs:
        m = kwargs['rowNum']
        n = kwargs['colNum']
    else:
        n = int(np.ceil(np.sqrt(ll)))
        m = 0
        if n * (n - 1) >= ll:
            m = np.ceil(np.sqrt(ll)) - 1
        elif n * (n - 1) < ll:
            m = np.ceil(np.sqrt(ll))

    im0 = cv2.imread(folder_loc + file_tag_list[0])
    dm1, dm2, dm3 = im0.shape
    # print(np.int(m),np.int(n),dm1,dm2,dm3)
    im_array = np.ones([int(m), int(n), dm1, dm2, dm3], dtype=np.uint8) * 255
    for ii in range(int(m * n)):
        jj = int(ii // n)
        kk = int(ii % n)
        # print(ii,jj,kk)
        if ii < ll:
            file_tag = file_tag_list[ii]
            im_array[jj, kk, :, :, :] = cv2.imread(folder_loc + file_tag)

    im_tile = cv2.vconcat([cv2.hconcat(im_row) for im_row in im_array])
    cv2.imwrite(folder_loc + file_name + '.png', im_tile)

    if delImg:
        for ff in file_tag_list:
            os.remove(folder_loc + ff)
        print('All added images are removed!')
        
        
# In[]
def get_time_str(): 
    today = datetime.datetime.now()
    time_str = today.strftime("%Y-%m-%d-%H-%M-%S")
    
    return time_str


def save_data(data, save_loc='default', file_name='default', print_msg=True):
    """
    Save data as pickle file.

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # get file naming parameters
    file_time = get_time_str()
    if file_name == 'default': 
        file_name = 'tempData_' + file_time + '.pkl'
    if save_loc == 'default': 
        save_loc = 'data/temp/'

    # check if such file already exists
    # if yes add '_c#' to the file_name
    timeout = 1000
    file_dir = save_loc + file_name
    for ii in range(timeout):
        if os.path.exists(file_dir):
            tag = file_name.split('_')[-1].split('.')[0]
            pkl = file_name.split('_')[-1].split('.')[1]
            if tag[0] == 'c' and tag[1:].isnumeric() and pkl == 'pkl':
                cc = int(tag[1:])
                file_name = file_name[:-len(file_name.split('_')[-1])] + 'c' + str(cc+1) + '.pkl'
            else:
                file_name = file_name[:-4] + '_c1' + '.pkl'
            file_dir = save_loc + file_name
        else:
            break
    if os.path.exists(file_dir):
        print(file_dir + ' already exists and is replaced!')
    
    # save data 
    with open(file_dir, 'bw') as fh:
        pickle.dump(data, fh)
    if print_msg: print("Data is saved in: " + file_dir)
    
    return save_loc, file_name
    
    
def load_data(save_loc, file_name):
    '''
    Load data from pickle file 

    Parameters
    ----------
    file_tag : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    file_dir = save_loc + file_name
    with open(file_dir, 'rb') as fh:
        data = pickle.load(fh)
    
    return data


# In[exception handling]
def print_exception(inst):
    '''
    This function print details of the exception.

    Parameters
    ----------
    inst : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    print(type(inst)) # the exception instance
    print(inst.args) # arguments stored in .args
    print(inst)


# In[array analysis]
def sample_list(data, sampling_period): 
    new_data = []
    if len(data) > 0: 
        for ii, dd in enumerate(data): 
            if ii % sampling_period == 0: 
                new_data.append(dd)
            
    return new_data


def get_intersection(arr_list): 
    """
    Get intersection from a list of arrays 
    """
    intersection = reduce(np.intersect1d, arr_list) 
    
    return intersection  


def get_union(arr_list): 
    """
    Get union from a list of arrays 
    """
    union = reduce(np.union1d, arr_list) 
    
    return union


def shift_elements(arr, num, fill_value=0):
    """
    shift an array by num and fill in with fill_value 
    """
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result 


def extract_connected_regions(arr, dx=1): 
    """
    generate a list of connected region 
    a conncted region is a sorted array whose difference of two consecutive elements is dx 
    """
    arr.sort()
    diff = np.diff(arr) 
    ind_list = list(np.where(diff!=dx)[0] + 1) 
    ind_list.insert(0, 0) 
    ind_list.append(arr.size) 
    cr_list = [] 
    for jj in range(len(ind_list)-1): 
        cr_list.append(arr[ind_list[jj]:ind_list[jj+1]]) 
        
    return cr_list 


def compute_1d_integral_function(x, integrand): 
    """
    inverse of np.gradient in 1d
    """
    matf = np.tri(x.size) * integrand
    Fc = np.trapz(matf, x, axis=1)
    
    return Fc 
        
        
# In[]
def rad2pipi(rad): 
    """
    Conver any angle in radians to [-pi, pi]
    Input must be a numpy array 
    """
    x = rad % (np.pi*2)
    y = np.piecewise(x, [x < -np.pi, ((x >= -np.pi) & (x <= np.pi)), x > np.pi], [lambda x: x + np.pi*2, lambda x: x, lambda x: x - np.pi*2])
        
    return y


# In[]
def point_inside_ellipse(x, y, a, b, theta, a0, b0): 
    c, s = np.cos(theta), np.sin(theta)
    xt = x - a0
    yt = y - b0
    u = c*xt + s*yt
    v = -s*xt + c*yt
    f = u**2 / a**2 + v**2 / b**2
    if f <= 1: 
        result = True
    else:
        result = False
        
    return result 


def point_inside_parabola(x, y, h, w, theta, x0, y0): 
    c, s = np.cos(theta), np.sin(theta)
    xt = x - x0
    yt = y - y0
    u = c*xt + s*yt
    v = -s*xt + c*yt
    if v >= (h/w**2) * u**2 - h: 
        result = True 
    else:
        result = False
        
    return result


def point_above_line(x, y, x1, y1, x2, y2): 
    if x1 != x2: 
        a = (y2-y1)/(x2-x1)
        b = y1-a*x1
        if y >= a*x+b: 
            result = True
        else:
            result = False
    else:
        if x >= x1:  
            result = True
        else:
            result = False
            
    return result


def point2line_distance(x, y, x1, y1, x2, y2): 
    deno = ((x2-x1)**2 + (y2-y1)**2)**0.5
    nume = abs((x2-x1)*(y1-y) - (x1-x)*(y2-y1))
    dist = nume/deno
    
    return dist


def point_inside_box(x, y, h, w, theta, x0, y0, h_margin=0, w_margin=0): 
    '''
    Test whether a point (x, y) is inside a box or the point is too close to the borders
    The box with width 2w and height 2h and orientation theta is centered at (x0, y0),
    with four corners shown below: 
        p1  ____________________  p4
           |                    |
           |                    |              
           |         p0         | 2h            orientation = theta
           |                    |
        p2 |____________________| p3
                     2w  
    If point (x, y) has distance to line p1-p4 or line p2-p3 less than h_margin, 
    or has distance to line p1-p2 or line p3-p4 less than w_margin, 
    then the point is claimed to be too close to borders 

    '''
    c, s = np.cos(theta), np.sin(theta) 
    R = np.array(((c, -s), (s, c)))
    p0 = np.array([x0, y0])
    v1 = np.array([-w, h])
    v2 = np.array([-w, -h])
    v3 = np.array([w, -h])
    v4 = np.array([w, h])
    p1 = R@v1 + p0
    p2 = R@v2 + p0
    p3 = R@v3 + p0
    p4 = R@v4 + p0
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    dist1 = point2line_distance(x, y, x1, y1, x2, y2)
    dist2 = point2line_distance(x, y, x3, y3, x4, y4)
    dist3 = point2line_distance(x, y, x2, y2, x3, y3)
    dist4 = point2line_distance(x, y, x1, y1, x4, y4)
    
    margin = 1e-3
    if abs((dist1+dist2)-w*2) <= margin and abs((dist3+dist4)-h*2) <= margin: 
        inside = True
    else:
        inside = False
        
    close_to_border = np.array([
        dist1 < w_margin, # left
        dist2 < w_margin, # right
        dist3 < h_margin, # bottom
        dist4 < h_margin, # top 
        ])
    
    return inside, close_to_border


# In[]
def compute_arc_length_piecewise(xs, ys):
    diffx = np.diff(xs)
    diffy = np.diff(ys)
    length = np.sum(np.sqrt(diffx**2 + diffy**2))

    return length


# In[]
def func_cone(x, xt, yt, k1, k2):
    """
          (xt, yt)
     k1       *    k2
         *      *
    *             *
    """
    b1 = yt - k1*xt
    b2 = yt - k2*xt
    y = np.piecewise(x, [x<=xt, x>xt], [lambda x: k1*x+b1, lambda x: k2*x+b2])

    return y    


# In[Optimal Transport]
def compute_W1_distance(p1, p2, x): 
    """
    compute the Wasserstein1-distance of two 1d probability mass function evaluated on x

    Parameters
    ----------
    p1 : TYPE
        DESCRIPTION.
    p2 : TYPE
        DESCRIPTION.
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    W1 : TYPE
        DESCRIPTION.

    """
    F1 = np.cumsum(p1)
    F2 = np.cumsum(p2)
    Fdiff = np.abs(F1 - F2)
    W1 = np.trapz(Fdiff, x=x)
    
    return W1 


def compute_Wp_distance(p, p1, p2, x): 
    F1 = np.cumsum(p1)
    F2 = np.cumsum(p2)
    QF1 = lambda q: np.interp(q, F1, x) 
    QF2 = lambda q: np.interp(q, F2, x) 
    qs = np.arange(1001)/1000
    Finv1 = QF1(qs) 
    Finv2 = QF2(qs) 
    Fint = np.abs(Finv1-Finv2)**p 
    Wp = np.trapz(Fint, x=qs)**(1/p)
    
    return Wp 


def compute_W1_distance_of_structures(h1, h2, x): 
    p1 = h1 / np.trapz(h1, x) 
    p2 = h2 / np.trapz(h2, x) 
    W1 = compute_W1_distance(p1, p2, x) 
    
    return W1 


def compute_Wp_distance_of_structures(p, h1, h2, x): 
    p1 = h1 / np.trapz(h1, x) 
    p2 = h2 / np.trapz(h2, x) 
    Wp = compute_Wp_distance(p, p1, p2, x) 
    
    return Wp


def get_W2_geodesics(xc, h0, h1, t_list): 
    """
    Get the intermediate structure based on W2 geodesics
        xc: x-axis of height function 
        h0: initial structure 
        h1: goal structure 
        t_list: list of floats in [0, 1]
    """
    # convert height function to probability density function
    v0 = np.trapz(h0, xc)
    p0 = h0 / v0
    F0 = np.cumsum(p0)
    v1 = np.trapz(h1, xc)
    p1 = h1 / v1
    F1 = np.cumsum(p1)
    
    # compute PDF, quntile function, and optimal transport function
    PDF0 = lambda x: np.interp(x, xc, p0) 
    CDF0 = lambda x: np.interp(x, xc, F0) 
    QF1 = lambda q: np.interp(q, F1, xc) 
    fopt = lambda x: QF1(CDF0(x))
    
    # compute points on the geodesics 
    ht_list = [] 
    for t in t_list: 
        if t < 1: 
            ft = lambda x: (1-t)*x + t*fopt(x)
            yft = ft(xc)
            ftinv = lambda y: np.interp(y, yft, xc) 
            dyft = np.gradient(yft, xc) 
            dft = lambda x: np.interp(x, xc, dyft) 
            PDFt = lambda x: PDF0(ftinv(x)) / dft(ftinv(x)) 
            pt = PDFt(xc)
            ht = pt*v0 
        else: 
            ht = np.copy(h1) 
        ht_list.append(ht)
    
    return ht_list 


# In[]
def remove_outlier(x, y, kmax, return_ind_remove=False): 
    """
    Remove any outlier with steepness > kmax
    Notes: 
        The point has to be a local single outlier
        The algorithm cannot remove consecutive outliers
        This function may give false positive result (non-outlier point is detected as outlier)
    Example: 
        [..., 0, 0, 10, 0, 0, ...] ---> [..., 0, 0, 0, 0, 0, ...]
        This algorithm cannot filter something like: [..., 0, 0, 10, 10, 0, 0, ...]
    """
    # check data points between 1st and last one 
    N = y.size
    k = np.diff(y)/ np.diff(x)
    xs_list = []
    ys_list = []
    ind_remove_list = []
    for ii in range(N): 
        remove = False
        if ii > 0 and ii < N-1: 
            if np.sign(k[ii-1]) != np.sign(k[ii]): 
                if abs(k[ii-1]) > kmax and abs(k[ii]) > kmax: 
                    remove = True 
                    ind_remove_list.append(ii)
        if not remove: 
            ys_list.append(y[ii])
            xs_list.append(x[ii])     
            
    # check 1st data point
    dx = xs_list[1] - xs_list[0]
    dy = ys_list[1] - ys_list[0]
    if abs(dy/dx) > kmax: 
        ys_list.pop(0)
        xs_list.pop(0)
        ind_remove_list.insert(0, 0)
        
    # check last data point 
    dx = xs_list[-1] - xs_list[-2]
    dy = ys_list[-1] - ys_list[-2]
    if abs(dy/dx) > kmax: 
        ys_list.pop()
        xs_list.pop()
        ind_remove_list.append(N-1) 
        
    if return_ind_remove: 
        return np.array(xs_list), np.array(ys_list), ind_remove_list 
    else: 
        return np.array(xs_list), np.array(ys_list) 


def moving_average(x, y, filter_length): 
    hf = np.ones(filter_length) / filter_length 
    yf = scipy.signal.fftconvolve(y, hf, mode='valid')
    indi = int(np.floor(filter_length/2))
    xf = x[indi:indi+yf.size]
    
    return xf, yf 


# In[]
def pearson_correlation(x, y, **kwargs): 
    plot = get_kwargs(kwargs, 'plot', True) 
    xlabel = get_kwargs(kwargs, 'xlabel', 'x') 
    ylabel = get_kwargs(kwargs, 'ylabel', 'y') 
    res = scipy.stats.pearsonr(x, y)
    if plot: 
        fig, ax = plt.subplots()
        fig.dpi = 200 
        ax.plot(x, y, '.') 
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(str(res)) 
    
    return res 


# In[structure analysis]
def characterize_structure(structure): 
    x, h = structure 
    slope_max = np.amax(np.abs(np.gradient(h, x))) 
    volume = np.trapz(h, x) 
    
    return slope_max, volume

