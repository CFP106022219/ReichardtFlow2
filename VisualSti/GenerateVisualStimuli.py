# coding: utf8
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from numba import autojit
# from VisualSti import borst as bs
import pandas as pd
import pickle
import gc

class sti:
    def __init__(self):
        self.width = 320
        self.height = 240
        self.fps = 120
        self.sec = 10
        self.contrast = 1.0
        # self.maxt = self.fps*self.sec
        # self.movie = np.zeros((self.height, self.width, self.maxt))

        # sine grating parameter
        self.wlen = 80
        self.degr = 0
        self.angl = (self.degr/180)*np.pi
        self.V = 0
        # self.V = self.wlen/self.fps
        # self.V_t = np.ones(self.maxt) * self.V
        # self.degr = 45
        # self.angl = (self.degr/180)*np.pi
        self.Vdgr = self.angl
        self.Vy = self.V*np.cos(self.Vdgr)
        self.Vx = self.V*np.sin(self.Vdgr)

        # genavi pareter
        # self.avifname = 'out.avi'

    @autojit
    def singrat(self):
        self.angl = (self.degr/180)*np.pi
        self.Vdgr = self.angl

        self.Vx = self.V*np.cos(self.Vdgr)
        print ('Vx', self.Vx)
        self.Vy = self.V*np.sin(self.Vdgr)
        print ('Vy', self.Vy)

        self.maxt = self.fps*self.sec
        [xv, yv, tt] = np.meshgrid(range(0, self.width), range(0, self.height), range(0, self.maxt))
        self.movie = np.cos(2.0*np.pi*((np.sin(self.angl)/self.wlen)*(yv-(self.Vy*tt)) + (np.cos(self.angl)/(self.wlen))*(xv-(self.Vx*tt))))
        self.movie = self.movie*self.contrast*0.5+0.5
        gc.collect()
        return self.movie

    def grid(self):
        self.angl = (self.degr/180)*np.pi
        self.Vdgr = self.angl

        self.Vx = self.V*np.cos(self.Vdgr)
        print ('Vx', self.Vx)
        self.Vy = self.V*np.sin(self.Vdgr)
        print ('Vy', self.Vy)

        ang2 = np.pi/2

        self.maxt = self.fps*self.sec
        [xv, yv, tt] = np.meshgrid(range(0, self.width), range(0, self.height), range(0, self.maxt))
        self.movie = np.cos(2.0*np.pi*((np.sin(self.angl)/self.wlen)*(yv-(self.Vy*tt)) + (np.cos(self.angl)/(self.wlen))*(xv-(self.Vx*tt)))) * np.cos(2.0*np.pi*((np.sin(ang2)/self.wlen)*(yv-(0*tt)) + (np.cos(ang2)/(self.wlen))*(xv-(0*tt))))
        self.movie = self.movie*self.contrast*0.5+0.5
        gc.collect()
        return self.movie

    @autojit
    # rightward only 
    def gradient(self):
        self.angl = (self.degr/180)*np.pi
        self.Vdgr = self.angl

        self.Vx = self.V*np.cos(self.Vdgr)
        print ('Vx', self.Vx)
        self.Vy = self.V*np.sin(self.Vdgr)
        print ('Vy', self.Vy)

        self.maxt = self.fps*self.sec
        [xv, yv, tt] = np.meshgrid(range(0, self.width), range(0, self.height), range(0, self.maxt))
        self.movie = np.cos(2.0*np.pi*((np.sin(self.angl)/self.wlen)*(yv-(self.Vy*tt)) + (np.cos(self.angl)/(self.wlen))*(xv-(self.Vx*tt))))
        self.movie = self.movie*self.contrast*0.5+0.5
        for i in range(0,self.fps*self.sec):
            self.movie[:,0:int((self.V*i)),i] = 1
            self.movie[:,int((self.wlen)/2+(self.V*i)):,i] = 0
        gc.collect()
        return self.movie

    def gradient_minus(self):
        self.angl = (self.degr/180)*np.pi
        self.Vdgr = self.angl

        self.Vx = self.V*np.cos(self.Vdgr)
        print ('Vx', self.Vx)
        self.Vy = self.V*np.sin(self.Vdgr)
        print ('Vy', self.Vy)

        self.maxt = self.fps*self.sec
        [xv, yv, tt] = np.meshgrid(range(0, self.width), range(0, self.height), range(0, self.maxt))
        self.movie = np.cos(2.0*np.pi*((np.sin(self.angl)/self.wlen)*(yv-(self.Vy*tt)) + (np.cos(self.angl)/(self.wlen))*(xv-(self.Vx*tt)))+np.pi)
        self.movie = self.movie*self.contrast*0.5+0.5
        for i in range(0,self.fps*self.sec):
            self.movie[:,0:int((self.V*i)),i] = 0
            self.movie[:,int((self.wlen)/2+(self.V*i)):,i] = 1
        gc.collect()
        return self.movie


    def peak(self):
        self.angl = (self.degr/180)*np.pi
        self.Vdgr = self.angl

        self.Vx = self.V*np.cos(self.Vdgr)
        print ('Vx', self.Vx)
        self.Vy = self.V*np.sin(self.Vdgr)
        print ('Vy', self.Vy)

        self.maxt = self.fps*self.sec
        [xv, yv, tt] = np.meshgrid(range(0, self.width), range(0, self.height), range(0, self.maxt))
        self.movie = np.cos(2.0*np.pi*((np.sin(self.angl)/self.wlen)*(yv-(self.Vy*tt)) + (np.cos(self.angl)/(self.wlen))*(xv-(self.Vx*tt)))+np.pi)
        self.movie = self.movie*0.5+0.5
        self.movie = self.movie*self.contrast
        for i in range(0,self.fps*self.sec):
            self.movie[:,0:int((self.V*i)),i] = 0
            self.movie[:,int((self.wlen)+(self.V*i)):,i] = 0
        gc.collect()
        return self.movie

    def peak_minus(self):
        self.angl = (self.degr/180)*np.pi
        self.Vdgr = self.angl

        self.Vx = self.V*np.cos(self.Vdgr)
        print ('Vx', self.Vx)
        self.Vy = self.V*np.sin(self.Vdgr)
        print ('Vy', self.Vy)

        self.maxt = self.fps*self.sec
        [xv, yv, tt] = np.meshgrid(range(0, self.width), range(0, self.height), range(0, self.maxt))
        self.movie = np.cos(2.0*np.pi*((np.sin(self.angl)/self.wlen)*(yv-(self.Vy*tt)) + (np.cos(self.angl)/(self.wlen))*(xv-(self.Vx*tt))))
        self.movie = self.movie*0.5+0.5
        self.movie = self.movie*self.contrast
        for i in range(0,self.fps*self.sec):
            self.movie[:,0:int((self.V*i)),i] = 1
            self.movie[:,int((self.wlen)+(self.V*i)):,i] = 1
        gc.collect()
        return self.movie

    def peak_avg_bg(self):
        self.angl = (self.degr/180)*np.pi
        self.Vdgr = self.angl

        self.Vx = self.V*np.cos(self.Vdgr)
        print ('Vx', self.Vx)
        self.Vy = self.V*np.sin(self.Vdgr)
        print ('Vy', self.Vy)

        self.maxt = self.fps*self.sec
        [xv, yv, tt] = np.meshgrid(range(0, self.width), range(0, self.height), range(0, self.maxt))
        self.movie = np.cos(2.0*np.pi*((np.sin(self.angl)/self.wlen)*(yv-(self.Vy*tt)) + (np.cos(self.angl)/(self.wlen))*(xv-(self.Vx*tt)))+np.pi)
        for i in range(0,self.fps*self.sec):
            self.movie[:,0:int((self.V*i)),i] = -1
            self.movie[:,int((self.wlen)+(self.V*i)):,i] = -1
        self.movie = self.movie*self.contrast*0.25+0.5
        gc.collect()
        return self.movie

    def peak_avg_bg_minus(self):
        self.angl = (self.degr/180)*np.pi
        self.Vdgr = self.angl

        self.Vx = self.V*np.cos(self.Vdgr)
        print ('Vx', self.Vx)
        self.Vy = self.V*np.sin(self.Vdgr)
        print ('Vy', self.Vy)

        self.maxt = self.fps*self.sec
        [xv, yv, tt] = np.meshgrid(range(0, self.width), range(0, self.height), range(0, self.maxt))
        self.movie = np.cos(2.0*np.pi*((np.sin(self.angl)/self.wlen)*(yv-(self.Vy*tt)) + (np.cos(self.angl)/(self.wlen))*(xv-(self.Vx*tt))))
        for i in range(0,self.fps*self.sec):
            self.movie[:,0:int((self.V*i)),i] = 1
            self.movie[:,int((self.wlen)+(self.V*i)):,i] = 1
        self.movie = self.movie*self.contrast*0.25+0.5
        gc.collect()
        return self.movie

    @autojit
    def singrat_uint8(self):
        self.Vx = self.V*np.cos(self.Vdgr)
        self.Vy = self.V*np.sin(self.Vdgr)
        self.angl = (self.degr/180)*np.pi
        self.maxt = self.fps*self.sec
        [xv, yv, tt] = np.meshgrid(range(0, self.width), range(0, self.height), range(0, self.maxt))
        self.movieuint8 = np.cos(2.0*np.pi*((np.sin(self.angl)/self.wlen)*(yv-(self.Vy*tt)) + (np.cos(self.angl)/(self.wlen))*(xv-(self.Vx*tt))))
        self.movieuint8 = (((np.cos(2.0*np.pi*((np.sin(self.angl)/self.wlen)*(yv-(self.Vy*tt)) + (np.cos(self.angl)/(self.wlen))*(xv-(self.Vx*tt)))))*self.contrast+0.5)*255 + 0.5).astype(np.uint8)
        # self.movieuint8 = self.movie*self.contrast*0.5+0.5
        return self.movieuint8

    def genavi(self, avifname = 'out.avi'):
        self.avifname = avifname
        normalizedImg = np.zeros((self.height, self.width))
        out = cv.VideoWriter(self.avifname, cv.VideoWriter_fourcc(*'XVID'), self.fps, (self.width, self.height))
        for i in range(self.maxt):
            normalizedImg = self.movie[:,:,i] * 255 + 0.5
            # normalizedImg = cv.normalize(self.movie[:,:,i],  normalizedImg, 0, 255, cv.NORM_MINMAX)
            normalizedImg = normalizedImg.astype(np.uint8)
            normalizedImg = cv.cvtColor(normalizedImg,cv.COLOR_GRAY2RGB)
            out.write(normalizedImg)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        gc.collect()

    def genavi_null(self, avifname = 'out.avi'):
        m_null = np.empty_like(self.movie)
        print (self.maxt)
        for i in range(self.maxt):
            m_null[:,:,i] = self.movie[:,:,self.maxt-i-1]
        self.avifname = avifname
        normalizedImg = np.zeros((self.height, self.width))
        out = cv.VideoWriter(self.avifname, cv.VideoWriter_fourcc(*'XVID'), self.fps, (self.width, self.height))
        for i in range(self.maxt):
            normalizedImg = m_null[:,:,i] * 255 + 0.5
            # normalizedImg = cv.normalize(self.movie[:,:,i],  normalizedImg, 0, 255, cv.NORM_MINMAX)
            normalizedImg = normalizedImg.astype(np.uint8)
            normalizedImg = cv.cvtColor(normalizedImg,cv.COLOR_GRAY2RGB)
            out.write(normalizedImg)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        gc.collect()
        
    def genavi_uint8(self, avifname = 'out.avi'):
        self.avifname = avifname
        normalizedImg = np.zeros((self.height, self.width))
        out = cv.VideoWriter(self.avifname, cv.VideoWriter_fourcc(*'XVID'), self.fps, (self.width, self.height))
        for i in range(self.maxt):
            normalizedImg = self.movieuint8[:,:,i]
            # normalizedImg = cv.normalize(self.movie[:,:,i],  normalizedImg, 0, 255, cv.NORM_MINMAX)
            normalizedImg = normalizedImg.astype(np.uint8)
            normalizedImg = cv.cvtColor(normalizedImg,cv.COLOR_GRAY2RGB)
            out.write(normalizedImg)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    def savpickle(self, clsfname = 'test.sti'):
        self.classfname = clsfname
        with open(self.classfname, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)




def main():
    pass

if __name__ == '__main__': 

    sw0 = sti()
    sw0.wlen = 32
    sw0.fps = 120
    sw0.sec = 4
    sw0.degr = 0
    sw0.V = 0.5
    sw = sw0.gradient_minus()
    # sw0.genavi('out4.avi')
    sw0.savpickle()

    # plt.plot(sw0.movie[100,100,150:250])
    # plt.show()

    # with open('test.sti', 'rb') as input:
    #     sw0 = pickle.load(input)
    # print (sw0.degr)
    # main()