# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 19:19:35 2025

@author: ericl
"""
import sys
sys.path.insert(0, './src')

from CRC_Blueprints import generate_random_blueprint
from CRC_Visualization import visualize_construction 
from CRC_Policy import generate_CRC_policy
from CRC_Simulation import simulate_crc
from matplotlib import pyplot as plt


# In[]
bp = generate_random_blueprint((5, 4), make_plot=True, save_data=True)
policyInfo_dict = generate_CRC_policy(bp, sol_limit=1, make_plot=True)
simInfo = simulate_crc(bp, policyInfo_dict[0], robot_num=5, print_info=True, make_plot=True)
tw, sh, rh = simInfo 
visualize_construction(tw.start_struct, tw.goal_struct, sh, rh, style='2d', frame_sampling_period=100, video_format='gif')
visualize_construction(tw.start_struct, tw.goal_struct, sh, rh, style='3d', frame_sampling_period=100, video_format='gif')
plt.show()





