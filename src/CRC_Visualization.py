#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 14:47:03 2020

@author: Jiahe Chen
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy

from holoviews.plotting.bokeh.styles import font_size
from tqdm import tqdm

from Utility import get_time_str, make_video


# In[]
def plot_blueprint(
        blueprintArray,
        padding=False,
        navigation_map=None,
        colorbar=False,
        maxheight='default',
        file_loc='images/temp/',
        file_tag='default',
        subplots=None,
        save=False,
):
    """
    plot blueprint without the policy and padding
    """
    blueprint = blueprintArray.astype(int)
    if subplots == None:
        fig, ax = plt.subplots()
    else:
        fig, ax = subplots
    fig.set_dpi(100)
    fig.set_size_inches(blueprint.shape)
    if file_tag == 'default':
        file_tag = get_time_str()
    if padding:
        padding_offset = (1, 1)
    else:
        padding_offset = (0, 0)
    if maxheight == 'default':
        maxheight = np.amax(blueprint)
    # plot blueprints
    m, n = blueprint.shape
    # draw colorbar
    if colorbar:
        plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=maxheight), cmap='Blues'),
                     ticks=mpl.ticker.LinearLocator(maxheight + 1), ax=ax, label='Structure Height')
    # draw structures
    ax.imshow(blueprint, cmap='Blues', vmin=0, vmax=maxheight)
    # draw start and exit and docking
    startLoc = np.array([0, 0]) + padding_offset
    exitLoc = np.array([m-1, n-1]) - padding_offset
    ax.text(startLoc[1], startLoc[0], 'S', fontsize=21, color='black', ha='center', va='center')
    ax.text(exitLoc[1], exitLoc[0], 'E', fontsize=21, color='black', ha='center', va='center')
    if padding:
        dockingLoc = startLoc + np.array([0, -1])
        ax.text(dockingLoc[1], dockingLoc[0], 'D', fontsize=21, color='black', ha='center', va='center')
    # draw navigation map
    if navigation_map != None:
        for key in navigation_map:
            for ee in navigation_map[key]:
                x1 = ee[0][1] + padding_offset[1]
                y1 = ee[0][0] + padding_offset[0]
                x2 = ee[1][1] + padding_offset[1]
                y2 = ee[1][0] + padding_offset[0]
                dx = (x2 - x1) * 0.7
                dy = (y2 - y1) * 0.7
                ax.arrow(x1, y1, dx, dy, head_width=0.4, head_length=0.3, fc='r', ec='r')
    # set axis
    # ax.set_xticks(np.arange(n))
    # ax.set_yticks(np.arange(m))
    # ax.set_xlim(-0.5, n-0.5)
    # ax.set_ylim(-0.5, m-0.5)
    # ax.set_ylim(ax.get_ylim()[::-1])  # flip y-axis
    ax.set_axis_off()
    if save:
        plt.savefig(file_loc + file_tag + '.png')

    return fig, ax


def plot_blueprint_3d(
        blueprintArray, # blueprint as numpy array
        padding=False,
        file_loc='images/temp/',
        file_tag='default',
        subplots=None,
        save=False,
):
    blueprint = blueprintArray.astype(int)
    if subplots == None:
        fig = plt.figure(constrained_layout=True)
        ax = fig.add_subplot(projection='3d')
    else:
        fig, ax = subplots
    if file_tag == 'default':
        file_tag = get_time_str()
    if padding:
        padding_offset = (1, 1)
    else:
        padding_offset = (0, 0)
    rowNum, colNum = blueprint.shape
    levelNum = np.amax(blueprint)
    x, y, z = np.indices((rowNum, colNum, levelNum))
    # set start, exit and docking
    startLoc = np.array([0, 0]) + padding_offset
    exitLoc = np.array([rowNum-1, colNum-1]) - padding_offset
    startarray = (x == startLoc[0]) & (y == startLoc[1]) & (z < 1)
    exitarray = (x == exitLoc[0]) & (y == exitLoc[1]) & (z < 1)
    if padding:
        dockingLoc = startLoc + np.array([0, -1])
        dockingarray = (x == dockingLoc[0]) & (y == dockingLoc[1]) & (z < blueprint[*dockingLoc] )
    voxelarray = np.copy(startarray)
    for ii in range(rowNum):
        for jj in range(colNum):
            levelarray = (x == ii) & (y == jj) & (z < blueprint[ii, jj])
            voxelarray = np.logical_or(voxelarray, levelarray)
    # set the colors of each object
    colors = np.empty(voxelarray.shape, dtype=object)
    colors[voxelarray] = 'grey'
    colors[startarray] = 'blue'
    colors[exitarray] = 'green'
    if padding:
        colors[dockingarray] = 'orange'
    # plot structure
    ax.voxels(voxelarray, facecolors=colors, edgecolor='k')
    # ax.set_aspect('equal')
    ax.set_box_aspect((rowNum, colNum, levelNum * 0.5))
    ax.set_axis_off()
    ax.view_init(elev=45)
    if save:
        plt.savefig(file_loc + file_tag + '.png', dpi=150)

    return fig, ax


def plot_robots(
        robotInfo,
        subplots=None,
):
    if subplots == None:
        fig, ax = plt.subplots()
    else:
        fig, ax = subplots
    heading_map = {
        "N": np.array([-1, 0]),
        "E": np.array([0, 1]),
        "S": np.array([1, 0]),
        "W": np.array([0, -1]),
    }
    for robotName, robotStates in robotInfo.items():
        ridx = int(robotName[6:])
        loc, heading, cb = robotStates
        yr, xr = loc
        ax.scatter(xr, yr, s=40 ** 2, c='yellow', zorder=10)
        ax.text(xr, yr, str(ridx), fontsize=20, ha='center', va='center', zorder=10)
        dyr = heading_map[heading][0] * 0.7
        dxr = heading_map[heading][1] * 0.7
        ax.arrow(xr, yr, dxr, dyr, head_width=0.4, head_length=0.3, width=0.05, fc='red', ec='red', zorder=10)
        if cb:
            ax.scatter(xr, yr, s=25 ** 2, c='grey', marker='s', zorder=10)

    return fig, ax


def visualize_construction(
        structure_start,
        blueprint_padded,
        structureInfo_list,
        robotInfo_list,
        style='2d', # ['2d', '3d']
        frame_sampling_period=10,
        video_format='mp4',
):
    image_loc = 'video/video_image/'
    delete_images_afterwards = True
    frame_index = 0
    s_cur = np.copy(structure_start)
    maxheight = np.amax(blueprint_padded)
    duration = len(structureInfo_list)
    for tt in tqdm(range(duration)):
        structureInfo = structureInfo_list[tt]
        robotInfo = robotInfo_list[tt]
        cur_step, changeInfo = structureInfo
        if len(changeInfo) > 0:
            for change in changeInfo:
                s_cur[change[0], change[1]] += change[2]
        # make plots
        if tt % frame_sampling_period == 0 or tt == duration - 1:
            if style == '2d':
                subplots = plot_blueprint(s_cur, padding=True, colorbar=False, maxheight=maxheight)
                fig, ax = plot_robots(robotInfo, subplots=subplots)
            elif style == '3d':
                fig, ax = plot_blueprint_3d(s_cur, padding=True)
            ax.set_title('step: ' + str(tt), fontsize=21)
            # save figure
            file_name = image_loc + 'img_' + str(frame_index) + '.png'
            plt.savefig(file_name, dpi=200)
            plt.close()
            frame_index += 1
    make_video(video_format=video_format, delete_images_afterwards=delete_images_afterwards)


