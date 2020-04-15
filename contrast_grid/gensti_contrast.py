# coding: utf8
import sys 
sys.path.append('..')

import matplotlib.pyplot as plt
from VisualSti import GenerateVisualStimuli as gvs
from VisualSti import MotionDetection as md
import matplotlib.pyplot as plt
import pickle
import os
import gc
import numpy as np

def main():
    pass

# This file is to generate visual stimulus data

if __name__ == '__main__':

    folder_name = 'grid_contrast_test'
    
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    vrange = range(-2,-1)
    # print ((vrange[-1]))
    # print (max(vrange[-1], 3))
    num = 0
    velo_p = -2
    wx = 2
    for contrast in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        for wx in np.arange(3,6):
            velo = 2.0 ** velo_p
            wl = 2.0 ** (wx)
            print ('n = ', num)

            sw0 = gvs.sti()
            sw0.width = 64
            sw0.height = 64
            sw0.V = velo
            sw0.wlen = wl
            sw0.sec = 10
            sw0.contrast = contrast
            sw0.singrat()
            sw0.genavi('./{}/v2exp{}_wx2exp{}_contrast{}.avi'.format(folder_name, velo_p, wx,contrast))
            sw0.savpickle('./{}/v2exp{}_wx2exp{}_contrast{}.sti'.format(folder_name, velo_p, wx,contrast))
            del sw0
            gc.collect()
        print ('n = ', num)


    main()