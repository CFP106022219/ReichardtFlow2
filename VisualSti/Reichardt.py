import numpy as np
import pickle
import cv2
from scipy import signal

import matplotlib.pyplot as plt
import matplotlib as mpl
# from GenerateVisualStimuli import sti
# from GenerateVisualStimuli import sti

def gabor_fn(sigma, theta, Lambda, psi, gamma, phase, winsize):
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    # Bounding box
    nstds = 2 # Number of standard deviation sigma
    xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
    # print ('xmax', xmax)
    xmax = np.ceil(max(1, xmax))
    # print ('xmax', xmax)
    ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
    ymax = np.ceil(max(1, ymax))

    if winsize > 0:
        xmax = winsize
        ymax = winsize
    
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    # Rotation 
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    if (phase == 'cos'):
        gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
    elif (phase == 'sin'):
        gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.sin(2 * np.pi / Lambda * x_theta + psi)
    # print (np.mean(gb))    
    return gb




class RH(object):
    """docstring for ClassName"""
    def __init__(self,

        # spatial filter
        # A1 gabor filter
        A1_sigma = 5.0,
        A1_theta = np.pi/2,
        A1_Lambda = 15.0,
        A1_psi = 0,
        A1_gamma = 1.0,
        A1_phase = 'cos',

        # A2 gabor filter
        A2_sigma = 5.0,
        A2_theta = np.pi/2,
        A2_Lambda = 15.0,
        A2_psi = 0,
        A2_gamma = 1.0,
        A2_phase = 'sin',
        inputframe = np.zeros((2,2)),

        # lowpass filter
        fl = 0.25,
        fh = 0.25,

        winsize = 0 
        ):

        self.A1_sigma = A1_sigma
        self.A1_theta = A1_theta
        self.A1_Lambda = A1_Lambda
        self.A1_psi = A1_psi
        self.A1_gamma = A1_gamma
        self.A1_phase = A1_phase


        self.A2_sigma = A2_sigma
        self.A2_theta = A2_theta
        self.A2_Lambda = A2_Lambda
        self.A2_psi = A2_psi
        self.A2_gamma = A2_gamma
        self.A2_phase = A2_phase

        self.winsize = winsize


        self.gb_A1 = gabor_fn(self.A1_sigma, self.A1_theta, self.A1_Lambda, self.A1_psi, self.A1_gamma, self.A1_phase, self.winsize)
        self.gb_A2 = gabor_fn(self.A2_sigma, self.A2_theta, self.A2_Lambda, self.A2_psi, self.A2_gamma, self.A2_phase, self.winsize)

        self.fl = fl
        self.fh = fh
        self.lowpassA1 = signal.correlate2d(inputframe, self.gb_A1, "same")
        self.lowpassA2 = signal.correlate2d(inputframe, self.gb_A2, "same")

    def run(self, input):
        self.convA1 = signal.correlate2d(input, self.gb_A1, "same")
        self.convA2 = signal.correlate2d(input, self.gb_A2, "same")

        temp1 = (1-self.fh)*self.lowpassA1 + self.fh*self.convA1
        temp2 = (1-self.fl)*self.lowpassA2 + self.fl*self.convA2

        self.lowpassA1 = temp1
        self.lowpassA2 = temp2

        self.A2B1 = np.multiply(self.lowpassA1, self.convA2)
        self.A1B2 = np.multiply(self.lowpassA2, self.convA1)

        Rh = self.A2B1 - self.A1B2

        return Rh

    def print_gabor(self, fname0 = 'gabor0', fname1 = 'gabor1'):
        norm = mpl.colors.Normalize(vmin=-1, vmax=1)
        imgplot = plt.imshow(self.gb_A1, cmap = plt.cm.gray , norm = norm)
        # plt.title('degree = {}, sigma = {}'.format(self.A1_theta, self.A1_sigma))
        plt.colorbar(imgplot)
        # plt.axis('off')
        plt.tight_layout()
        # plt.show()
        plt.savefig(fname0)
        plt.clf()
        imgplot = plt.imshow(self.gb_A2,
                            cmap = plt.cm.gray, 
                            norm = norm
                            )
        # plt.colorbar(imgplot)
        # plt.axis('off')
        plt.tight_layout()
        # plt.show()
        plt.savefig(fname1)
        plt.clf()





def main():
    pass

if __name__ == '__main__': 

    '''
    img = cv2.imread('frame08.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    


    with open('test.sti', 'rb') as input:
        st = pickle.load(input).movie

    sw0 = RH(inputframe = st[...,0])
    rh = np.empty_like(st)
    for i in range(0,120):
        rh[...,i] = sw0.run(st[...,i])
        print ('frame', i)

    rh = rh*0.05

    # fig, ax = plt.subplots()
    # ax.axis([1, 150, -0.5, 5])
    plt.plot(st[40,40,1:120], color='black', label='Image')
    # plt.plot(convR[40,40,1:120], color='blue', label='normal flow')
    # plt.plot(convL[40,40,1:120], color='red', label='Borst')
    plt.plot(rh[40,40,1:120], color='blue', label='Reichardt')
    # plt.plot(st[40,40,1:120], color='green', label='Reichardt(NuLL)')
    plt.legend(loc='best')

    plt.show()
    '''

    # imgplot = plt.imshow(sw0.convA1)
    # plt.show()
    # plt.clf()

    gamma = 0.8

    img = cv2.imread('frame08.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sw0 = RH(inputframe = img,
            A1_gamma = gamma,
            A2_gamma = gamma,
            )
    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    imgplot = plt.imshow(sw0.gb_A2, cmap = plt.cm.gray , norm = norm)
    plt.title('gamma = {}'.format(gamma))
    plt.colorbar(imgplot)
    plt.show()
    plt.clf()

    # sw0 = Reichardt()
    # norm = mpl.colors.Normalize(vmin=-1, vmax=1)

    # imgplot = plt.imshow(sw0.gb_A2, cmap = plt.cm.gray , norm = norm)
    # plt.title('degree = {}, sigma = {}'.format(sw0.A1_theta, sw0.A1_sigma))
    # plt.colorbar(imgplot)
    # plt.show()
    # plt.clf()
