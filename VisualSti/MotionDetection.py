from VisualSti import borst as bs
import numpy as np
import pickle
import cv2

def lp_filer():
    pass


class MD_borst(object):
    """docstring for ClassName"""
    def __init__(self):

        self.input = np.zeros((2,2,1))
        self.result = np.zeros((2,2,1))

        self.lptau = 5.0
        self.hptau = 25.0

        self.Eexc = +50.0
        self.Einh = -20.0
        self.gleak = 1.0
        self.mean = 0.0
        
    def run(self):

        noff = self.input.shape[1]

        lp = bs.lowpass(self.input, self.lptau)
        hp = bs.highpass(self.input, self.hptau)

        Mi9 = bs.rect(1.0-lp[:,0:noff-2,:],0)
        Mi1 = bs.rect(hp[:,1:noff-1,:],0)
        Mi4 = bs.rect(lp[:,2:noff-0,:],0)

        gexc = bs.rect(Mi1,0)
        ginh = bs.rect(Mi9+Mi4,0)

        # self.result = (self.Eexc*gexc+self.Einh*ginh)/(gexc+ginh+self.gleak)
        self.result = bs.rect((self.Eexc*gexc+self.Einh*ginh)/(gexc+ginh+self.gleak),0)
        self.mean = np.mean(self.result)
        del lp
        del hp
        del Mi9
        del Mi1
        del Mi4
        del gexc
        del ginh
        # T4a_rect=bs.rect(T4a,0)
        # self.result = T4a
        return self.result

    def savpickle(self, clsfname = 'test.bs'):
        self.classfname = clsfname
        with open(self.classfname, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def loadpicke(self, ldfname = 'load.sti'):
        self.loadfname = ldfname
        with open(self.loadfname, 'rb') as input:
            self.input = pickle.load(input).movie

class MD_borst_null(object):
    """docstring for ClassName"""
    def __init__(self):

        self.input = np.zeros((2,2,1))
        self.result = np.zeros((2,2,1))

        self.lptau = 5.0
        self.hptau = 25.0

        self.Eexc = +50.0
        self.Einh = -20.0
        self.gleak = 1.0
        self.mean = 0.0
        
    def run(self):

        noff = self.input.shape[1]

        lp = bs.lowpass(self.input, self.lptau)
        hp = bs.highpass(self.input, self.hptau)

        Mi4 = bs.rect(1.0-lp[:,2:noff-0,:],0)
        Mi1 = bs.rect(hp[:,1:noff-1,:],0)
        Mi9 = bs.rect(lp[:,0:noff-2,:],0)

        gexc = bs.rect(Mi1,0)
        ginh = bs.rect(Mi9+Mi4,0)

        # self.result = (self.Eexc*gexc+self.Einh*ginh)/(gexc+ginh+self.gleak)
        self.result = bs.rect((self.Eexc*gexc+self.Einh*ginh)/(gexc+ginh+self.gleak),0)
        self.mean = np.mean(self.result)
        del lp
        del hp
        del Mi9
        del Mi1
        del Mi4
        del gexc
        del ginh
        # T4a_rect=bs.rect(T4a,0)
        # self.result = T4a
        return self.result

    def savpickle(self, clsfname = 'test.bs'):
        self.classfname = clsfname
        with open(self.classfname, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def loadpicke(self, ldfname = 'load.sti'):
        self.loadfname = ldfname
        with open(self.loadfname, 'rb') as input:
            self.input = pickle.load(input).movie


class NormalFlow_x(object):
    """docstring for ClassName"""
    def __init__(self):

        self.input = np.zeros((2,2,1))
        self.result = np.zeros((2,2,1))
        self.alpha = 0.000000001
        self.mean = 0.0
        
    def run(self):

        noffx = self.input.shape[0]
        noffy = self.input.shape[1]
        nofft = self.input.shape[2]

        Ix = self.input[1:noffx-1,2:noffy-0,1:nofft] - self.input[1:noffx-1,0:noffy-2,1:nofft]
        Iy = self.input[2:noffx-0,1:noffy-1,1:nofft] - self.input[0:noffx-2,1:noffy-1,1:nofft]
        It = self.input[1:noffx-1,1:noffy-1,1:nofft] - self.input[1:noffx-1,1:noffy-1,0:nofft-1]

        self.result = -2*It*Ix/(np.square(Ix)+np.square(Iy)+self.alpha)
        resulty = -2*It*Iy/(np.square(Ix)+np.square(Iy)+self.alpha)
        print (np.mean(resulty))
        print (np.mean(self.result))
        self.mean = np.mean(self.result)
        del Ix, Iy, It, noffx, noffy, nofft
        return self.result

    def savpickle(self, clsfname = 'test.bs'):
        self.classfname = clsfname
        with open(self.classfname, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def loadpicke(self, ldfname = 'load.sti'):
        self.loadfname = ldfname
        with open(self.loadfname, 'rb') as input:
            self.input = pickle.load(input).movie

class STfilter(object):
    """docstring for ClassName"""
    def __init__(self, arg):
        super(ClassName, self).__init__()
        self.arg = arg
        self.result = np.zeros((2,2,1))
        self.fh = 0.4
        self.fl = 0.05

        self.sigma = 5.0
        self.theta = np.pi/2
        self.Lambda = 10.0
        self.psi = 0
        self.gamma = 1.0
        
    def gabor_fn(self, sigma, theta, Lambda, psi, gamma):
        sigma_x = self.sigma
        sigma_y = float(self.sigma) / self.gamma

    # Bounding box
        nstds = 3 # Number of standard deviation sigma
        xmax = max(abs(nstds * self.sigma_x * np.cos(self.theta)), abs(nstds * sigma_y * np.sin(self.theta)))
        print ('xmax', xmax)
        xmax = np.ceil(max(1, xmax))
        print ('xmax', xmax)
        ymax = max(abs(nstds * sigma_x * np.sin(self.theta)), abs(nstds * sigma_y * np.cos(self.theta)))
        ymax = np.ceil(max(1, ymax))
        xmin = -xmax
        ymin = -ymax
        (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

        # Rotation 
        x_theta = x * np.cos(self.theta) + y * np.sin(self.theta)
        y_theta = -x * np.sin(self.theta) + y * np.cos(self.theta)

        gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.sin(2 * np.pi / self.Lambda * x_theta + self.psi)
        return gb