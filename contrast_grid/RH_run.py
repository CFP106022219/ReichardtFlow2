# coding: utf8

import sys 
sys.path.append('..')

import matplotlib.pyplot as plt
from VisualSti import GenerateVisualStimuli as gvs
from VisualSti import MotionDetection as md
from VisualSti import Reichardt as rh
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
import os
import gc
import numpy as np
import sys
from scipy import signal
import time
import cv2


def basic_stimulus_null(inputfilen, outputfilen):
    with open(inputfilen, 'rb') as input:
        st = pickle.load(input).movie
    sw2 = rh.RH(inputframe = st[...,0],
                A1_sigma = 5,
                A2_sigma = 5,)
    rharr = np.empty_like(st)
    A2B1 = np.empty_like(st)
    A1B2 = np.empty_like(st)
    for i in range(0,120):
        rharr[...,i] = sw2.run(st[...,120-i])
        A2B1[...,i] = sw2.A2B1
        A1B2[...,i] = sw2.A1B2
        print ('frame', i)
    rharr = rharr*1
    A2B1 = A2B1*1
    A1B2 = A1B2*1
    
    fig, ax1 = plt.subplots()
    fig.subplots_adjust(right=0.88)
    ax2 = ax1.twinx()
    ax1.set_ylim([-2,2])
    p1, = ax1.plot(st[40,40,1:120], color='black', label='Image')
    p2, = ax2.plot(rharr[40,40,1:120], color='blue', label='Reichardt')
    p3, = ax2.plot(A2B1[40,40,1:120], '--', color='green', label='A2B1')
    p4, = ax2.plot(A1B2[40,40,1:120], '--', color='red', label='A1B2')

    ax1.set_xlabel('frame')
    ax1.set_ylabel('image intensity', color='black')
    ax2.set_ylabel('response', color='blue')
    lines = [p1, p2, p3, p4]
    labs = [l.get_label() for l in lines]
    ax1.legend(lines, labs, loc='best')
    plt.savefig(outputfilen)
    plt.clf()

def basic_stimulus(inputfilen, outputfilen):
    with open(inputfilen, 'rb') as input:
        st = pickle.load(input).movie
    sw2 = rh.RH(inputframe = st[...,0],
                # A1_phase = 'cos',
                # A2_phase = 'sin',
                A1_sigma = 5,
                A2_sigma = 5,
                # A1_Lambda = 4.0,
                # A2_Lambda = 4.0,
                # fl = 0.1,
                # fh = 0.1,
                # winsize = 2
                )
    # sw2.print_gabor()
    rharr = np.empty_like(st)
    A2B1 = np.empty_like(st)
    A1B2 = np.empty_like(st)
    # sw2.print_gabor('gab3')
    frame_range = 120
    for i in range(0,frame_range):
        rharr[...,i] = sw2.run(st[...,i])
        A2B1[...,i] = sw2.A2B1
        A1B2[...,i] = sw2.A1B2
        print ('frame', i)
    rharr = rharr*1
    A2B1 = A2B1*1
    A1B2 = A1B2*1    
    
    fig, ax1 = plt.subplots()
    fig.subplots_adjust(right=0.88)
    ax2 = ax1.twinx()
    ax1.set_ylim([-2,2])
    p1, = ax1.plot(st[40,40,1:frame_range], color='black', label='Image')
    p2, = ax2.plot(rharr[40,40,1:frame_range], color='blue', label='Reichardt')
    p3, = ax2.plot(A2B1[40,40,1:frame_range], '--', color='green', label='A2B1')
    p4, = ax2.plot(A1B2[40,40,1:frame_range], '--', color='red', label='A1B2')

    ax1.set_xlabel('frame')
    ax1.set_ylabel('image intensity', color='black')
    ax2.set_ylabel('response', color='blue')
    lines = [p1, p2, p3, p4]
    labs = [l.get_label() for l in lines]
    ax1.legend(lines, labs, loc='best')
    plt.show()
    plt.savefig(outputfilen)
    plt.clf()

def main():
    pass



if __name__ == '__main__':

    # basic_stimulus('./data/basic/gradient.sti', 'result/rh_sti/gradient.png')
    # basic_stimulus_null('./data/basic/gradient.sti', 'result/rh_sti/gradient_null.png')

    # basic_stimulus('./data/basic/peak.sti', 'result/rh_sti/peak_AB.png')
    # basic_stimulus_null('./data/basic/peak.sti', 'result/rh_sti/peak_AB_null.png')

    # basic_stimulus_null('./data/basic/peak.sti', 'result/rh_sti/peak.png')

    # basic_stimulus('./data/basic/peak_minus.sti', 'result/rh_wxwt/test.png')
    # basic_stimulus_null('./data/basic/peak_minus.sti', 'result/rh_sti/peak_minus.png')

    basic_stimulus('./grid_contrast_test/v2exp-2_wx2exp4_contrast0.8.sti', 'v2exp-2_wx2exp4_contrast0.8.png')