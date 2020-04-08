# -*- coding: utf-8 -*-
import numpy as np
import scipy as scipy
import matplotlib.pyplot as plt

def lowpass(x,tau):
    mydim=x.ndim
    result=np.zeros_like(x)
    n=x.shape
    if tau<1:
        result=x
    if tau>1:
        if mydim==1:
            n=n[0]
            result[0]=x[0]
            for i in range(0,n-1):
                result[i+1]=1.0/tau*(x[i]-result[i])+result[i]
        if mydim==2:
            n=n[1]
            result[0::,0]=x[0::,0]
            for i in range(0,n-1):
                result[:,i+1]=1.0/tau*(x[:,i]-result[:,i])+result[:,i]       
        if mydim==3:      
            n=n[2]
            result[0::,0::,0]=x[0::,0::,0]
            for i in range(0,n-1):
                result[:,:,i+1]=1.0/tau*(x[:,:,i]-result[:,:,i])+result[:,:,i]      
    return result
    
def highpass(x,tau):
    if tau<1:
        result=x
    if tau>1:
        result=x-lowpass(x,tau)
    return result
    
def normalize(x):
    result=x/np.nanmax(x)
    return result
    
def blurr(inp_image,FWHM):
    if inp_image.ndim==1: z=Gauss1D(FWHM,4*FWHM)
    if inp_image.ndim==2: z=Gauss2D(FWHM,4*FWHM)
    result=scipy.ndimage.convolve(inp_image,z)
    return result

def Gauss1D(FWHM,RFsize):
    myrange=RFsize/2
    sigma=FWHM/(2.0*np.sqrt(2*np.log(2)))
    x=np.arange(-myrange,(myrange+1),1)*1.0
    z=np.exp(-x**2/(2*(sigma**2)))
    z=z/np.sum(z)
    return z
    
def Gauss2D(FWHM,RFsize):
    myrange=RFsize/2
    sigma=FWHM/(2.0*np.sqrt(2*np.log(2)))
    x=np.arange(-myrange,(myrange+1),1)
    y=np.arange(-myrange,(myrange+1),1)
    x,y=np.meshgrid(x,y)
    r=np.sqrt(x**2+y**2)
    z=np.exp(-r**2/(2*(sigma**2)))
    z=z/np.sum(z)
    return z

# calculates a rebinned array of input x
# all new dims must be integer fractions or multiples of input dims

def rebin(x,f0,f1=0,f2=0):
    mydim=x.ndim
    n=x.shape
    if mydim==1:
        result=np.zeros((f0))
        if f0 <=  n[0]:
            result=x[0:n[0]:n[0]/f0]
        if f0 >  n[0]:
            result=np.repeat(x,f0/n[0])
    if mydim==2:
        result=np.zeros((f0,f1))
        interim=np.zeros((f0,n[1]))
        #handling 1st dim
        if f0 <=  n[0]:
            interim=x[0:n[0]:n[0]/f0,:]
        if f0 >  n[0]:
            interim=np.repeat(x,f0/n[0],axis=0)
        #handling 2nd dim
        if f1 <=  n[1]:
            result=interim[:,0:n[1]:n[1]/f1]
        if f1 >  n[1]:
            result=np.repeat(interim,f1/n[1],axis=1)
    if mydim==3:
        result=np.zeros((f0,f1,f2))
        interim1=np.zeros((f0,n[1],n[2]))
        interim2=np.zeros((f0,f1,n[2]))
        #handling 1st dim
        if f0 <=  n[0]:
            interim1=x[0:n[0]:n[0]/f0,:,:]
        if f0 >  n[0]:
            interim1=np.repeat(x,f0/n[0],axis=0)
        #handling 2nd dim
        if f1 <=  n[1]:
            interim2=interim1[:,0:n[1]:n[1]/f1,:]
        if f1 >  n[1]:
            interim2=np.repeat(interim1,f1/n[1],axis=1)
        #handling 3rd dim
        if f2 <=  n[2]:
            result=interim2[:,:,0:n[2]:n[2]/f2]
        if f2 >  n[2]:
            result=np.repeat(interim2,f2/n[2],axis=2)

    return result.copy()

def rect(x,thrld):
    result=x-thrld
    result=result*(result>0)
    result=result+thrld
    return result
    
def ceil(x,thrld):
    result=x-thrld
    result=result*(result<0)
    result=result+thrld
    return result
    
def add_photonnoise(signal,meanlum):
    noisysignal=np.random.poisson(signal*meanlum)/(1.0*meanlum)
    noiselevel=np.sqrt(np.mean((noisysignal-signal)**2))/np.sqrt(np.mean((signal)**2))
    
    print ('noiselevel=', noiselevel)
    
    return noisysignal
    
# --------------- MODELS OF MOTION DETECTORS -----------------------------

# calculates the output of a 2dim array of EMDs
# deltat determines temporal resolution in ms
    
def calc_4QHR(R16,deltat):
    
    noff=40
    
    lp=lowpass(R16,250/deltat)
    hp=highpass(R16,250/deltat)
    
    Txa=lp[:,0:noff-1,:]*hp[:,1:noff,:]
    Txb=lp[:,1:noff,:]*hp[:,0:noff-1,:]
    
    mdout=Txa-Txb
    HS=np.mean(np.mean(mdout,axis=0),axis=0)
    
    return HS
    
def calc_2QHR(lp,hp):
    
    noff=40
    
    Txa=rect(lp[:,0:noff-1,:]*hp[:,1:noff,:],0)
    Txb=rect(lp[:,1:noff,:]*hp[:,0:noff-1,:],0)
    
    return Txa, Txb

def calc_HRBL(lp,hp):
    
    noff=40
    DC=0.02
    
    A=rect(lp[:,0:noff-2,:],0)
    B=rect(hp[:,1:noff-1,:],0)
    C=rect(lp[:,2:noff-0,:],0)
    
    Txa=rect(A*B/(DC+C),DC)
    
    A=rect(lp[:,2:noff-0,:],0)
    B=rect(hp[:,1:noff-1,:],0)
    C=rect(lp[:,0:noff-2,:],0)
    
    Txb=rect(A*B/(DC+C),DC)
    
    return Txa, Txb
    
def calc_cb_HRBL(lp,hp):
    
    noff=40
    
    Eexc=+50.0
    Einh=-20.0
    gleak=1.0
    
    Mi9=rect(1.0-lp[:,0:noff-2,:],0)
    Mi1=rect(hp[:,1:noff-1,:],0)
    Mi4=rect(lp[:,2:noff-0,:],0)
    
    gexc=rect(Mi1,0)
    ginh=rect(Mi9+Mi4,0)

    Txa=(Eexc*gexc+Einh*ginh)/(gexc+ginh+gleak)
    
    Mi9=rect(1.0-lp[:,2:noff-0,:],0)
    Mi1=rect(hp[:,1:noff-1,:],0)
    Mi4=rect(lp[:,0:noff-2,:],0)
    
    gexc=rect(Mi1,0)
    ginh=rect(Mi9+Mi4,0)
    
    Txb=(Eexc*gexc+Einh*ginh)/(gexc+ginh+gleak)
    
    return Txa, Txb
    
def calc_cb_HR(lp,hp):
    
    noff=40
    
    Eexc=+50.0
    Einh=-20.0
    gleak=1.0
    
    Mi9=rect(1.0-lp[:,0:noff-2,:],0)
    Mi1=rect(hp[:,1:noff-1,:],0)
    
    gexc=rect(Mi1,0)
    ginh=rect(Mi9,0)

    Txa=(Eexc*gexc+Einh*ginh)/(gexc+ginh+gleak)
    
    Mi9=rect(1.0-lp[:,2:noff-0,:],0)
    Mi1=rect(hp[:,1:noff-1,:],0)
    
    gexc=rect(Mi1,0)
    ginh=rect(Mi9,0)
    
    Txb=(Eexc*gexc+Einh*ginh)/(gexc+ginh+gleak)
    
    return Txa, Txb
    
def calc_cb_BL(lp,hp):
    
    noff=40
    
    Eexc=+50.0
    Einh=-20.0
    gleak=1.0

    Mi1=rect(hp[:,1:noff-1,:],0)
    Mi4=rect(lp[:,2:noff-0,:],0)
    
    gexc=rect(Mi1,0)
    ginh=rect(Mi4,0)

    Txa=(Eexc*gexc+Einh*ginh)/(gexc+ginh+gleak)
    
    Mi1=rect(hp[:,1:noff-1,:],0)
    Mi4=rect(lp[:,0:noff-2,:],0)
    
    gexc=rect(Mi1,0)
    ginh=rect(Mi4,0)
    
    Txb=(Eexc*gexc+Einh*ginh)/(gexc+ginh+gleak)
    
    return Txa, Txb
    
def calc_cb_eeiHRBL(lp,hp):
    
    noff=40
    
    Eexc=+50.0
    Einh=-20.0
    gleak=1.0
    
    Mi9=rect(lp[:,0:noff-2,:],0)
    Mi1=rect(hp[:,1:noff-1,:],0)
    Mi4=rect(lp[:,2:noff-0,:],0)
    
    gexc=rect(Mi1,0)+rect(Mi9,0)
    ginh=rect(Mi4,0)

    Txa=(Eexc*gexc+Einh*ginh)/(gexc+ginh+gleak)
    
    Mi9=rect(lp[:,2:noff-0,:],0)
    Mi1=rect(hp[:,1:noff-1,:],0)
    Mi4=rect(lp[:,0:noff-2,:],0)
    
    gexc=rect(Mi1,0)+rect(Mi9,0)
    ginh=rect(Mi4,0)
    
    Txb=(Eexc*gexc+Einh*ginh)/(gexc+ginh+gleak)
    
    return Txa, Txb
    
def calc_cb_eeHR(lp,hp):
    
    noff=40
    
    Eexc=+50.0
    gleak=1.0
    
    Mi9=rect(lp[:,0:noff-2,:],0)
    Mi1=rect(hp[:,1:noff-1,:],0)
    
    gexc=rect(Mi1,0)+rect(Mi9,0)

    Txa=(Eexc*gexc)/(gexc+gleak)
    
    Mi9=rect(lp[:,2:noff-0,:],0)
    Mi1=rect(hp[:,1:noff-1,:],0)
    
    gexc=rect(Mi1,0)+rect(Mi9,0)
    
    Txb=(Eexc*gexc)/(gexc+gleak)
    
    return Txa, Txb
    
def NewEMD(stimulus,deltat,det_switch,ret_switch,noisefac=0, resting_factor=0):
    
    n=stimulus.shape
    maxtime=n[2]
    noff=40
        
    ON_lptau=50.0/deltat
    OFF_lptau=50.0/deltat

    L1gain=1.0
    L2gain=1.0
    ONlpgain=1.0
    ONhpgain=1.0
    OFFlpgain=1.0
    OFFhpgain=1.0
    T4gain=1.0
    T5gain=1.0
    LPigain=1.0
    
    # add noise 
    
    if noisefac!=0: 
        stimulus=add_photonnoise(stimulus,noisefac)

    R16=rebin(stimulus,noff,noff,maxtime)
    
    # tilt the image slightly
    
    for i in range(3):
        j=10*i
        k=10*(i+1)
        R16[k:k+10,:,:]=np.roll(R16[j:j+10:,:],1,axis=1)
 
    if det_switch==0:
        
        print ('4QD HR')
        HS=calc_4QHR(R16,deltat)
        result=HS
        print ('HS cell')
        
    if det_switch > 0:
        
        interim=highpass(R16,250/deltat)
        interim=interim+0.1*R16
        L1=L1gain*rect(interim,0)
        L2=L2gain*rect(-(interim-0.05),0)
        
        ONlp=ONlpgain*lowpass(R16,ON_lptau)
        ONhp=ONhpgain*L1
        OFFlp=OFFlpgain*lowpass(1.0-R16,OFF_lptau)
        OFFhp=OFFhpgain*L2
        
        if det_switch==1:
            print ('2QD HR')
            T4a,T4b=calc_2QHR(ONlp,ONhp)
            T5a,T5b=calc_2QHR(OFFlp,OFFhp)
        
        if det_switch==2:
            print ('a*b/c HRBL')
            T4a,T4b=calc_HRBL(ONlp,ONhp)
            T5a,T5b=calc_HRBL(OFFlp,OFFhp)
        
        if det_switch==3:
            print ('conduct based HRBL')
            T4a,T4b=calc_cb_HRBL(ONlp,ONhp)
            T5a,T5b=calc_cb_HRBL(OFFlp,OFFhp)
        
        if det_switch==4:
            print ('conduct based HR')
            T4a,T4b=calc_cb_HR(ONlp,ONhp)
            T5a,T5b=calc_cb_HR(OFFlp,OFFhp)
                             
        if det_switch==5:
            print ('conduct based BL')
            T4a,T4b=calc_cb_BL(ONlp,ONhp)
            T5a,T5b=calc_cb_BL(OFFlp,OFFhp)
            
        if det_switch==6:
            print ('conduct based exc-exc HRBL')
            T4a,T4b=calc_cb_eeiHRBL(ONlp,ONhp)
            T5a,T5b=calc_cb_eeiHRBL(OFFlp,OFFhp)
            
        if det_switch==7:
            print ('conduct based exc-exc HR')
            T4a,T4b=calc_cb_eeHR(ONlp,ONhp)
            T5a,T5b=calc_cb_eeHR(OFFlp,OFFhp)
            
        T4rest=np.mean(T4a[:,:,0:50])
    
        T4a=T4gain*rect(T4a,T4rest*resting_factor)
        T4b=T4gain*rect(T4b,T4rest*resting_factor)
        T5a=T5gain*rect(T5a,T4rest*resting_factor)
        T5b=T5gain*rect(T5b,T4rest*resting_factor)
        
        T4a_mean=np.mean(np.mean(T4a,axis=0),axis=0)
        T4b_mean=np.mean(np.mean(T4b,axis=0),axis=0)
        T5a_mean=np.mean(np.mean(T5a,axis=0),axis=0)
        T5b_mean=np.mean(np.mean(T5b,axis=0),axis=0)
        
        HS=T4a_mean+T5a_mean-LPigain*(T4b_mean+T5b_mean)    
    
        if ret_switch == 0: 
            result=T4a_mean
            print ('T4a_mean')
        if ret_switch == 1: 
            result=T4b_mean
            print ('T4b_mean')
        if ret_switch == 2: 
            result=HS
            print ('HS cell')
        if ret_switch == 3: 
            result=T4a[10,10,:]
            print ('T4a[10,10,:]')
        if ret_switch == 4: 
            result=T5a[10,10,:]
            print ('T5a[10,10,:]')
        if ret_switch == 5: 
            result=ONlp[10,10,:]
            print ('ONlp[10,10,:]')
        if ret_switch == 6: 
            result=ONhp[10,10,:]
            print ('ONhp[10,10,:]')
        
    return result
    
def calc_singlewave(img_size,new_spat_freq,x,y):
    sinewave=np.sin((np.linspace(0,img_size-1,img_size)-x+y)/img_size*2.0*np.pi*new_spat_freq)
    return sinewave

def calc_sinegrating(tf, img_rot, spat_freq, contrast):
    n=tf.shape
    maxtime=n[0]
    img_size=200
    movie=np.zeros((img_size,img_size,maxtime))
    img_rot_rad=img_rot/360.0*2*np.pi
    new_spat_freq=spat_freq*np.cos(img_rot_rad)
    spat_wlength = img_size/spat_freq
    velo=tf*spat_wlength/100.0   
    print ('Temp Freq (dt=10 msec) =', np.max(tf), 'Hz')
    yshift=np.tan(img_rot_rad)
    new_velo=velo/np.cos(img_rot_rad)
    if img_rot==0:
        image=np.zeros((img_size,img_size))
        interim=calc_singlewave(img_size,spat_freq,0,0)
        for i in range(img_size):
            image[i,0::]=interim 
        for i in range(maxtime):
            movie[:,:,i]=np.roll(image,int(sum(velo[0:i])),axis=1)      
    else:        
        for i in range(maxtime):
                for y in range(img_size):
                    movie[y,:,i]=calc_singlewave(200,new_spat_freq,int(sum(new_velo[0:i])),int(yshift*y))  
    movie=movie*contrast*0.5+0.5
    return movie
    
def calc_motion_noise(maxtime,direction,perccoher,blurrindex):
    
    print
    print ('coherence [%]', perccoher)
    print
    
    resol=100
    mostart=50
    mostop=maxtime-50
    movie=np.zeros((resol,resol,maxtime))
    nofdots=500
    seq=np.random.permutation(np.linspace(0,nofdots-1,nofdots))
    xpos=np.random.randint(resol, size=nofdots)
    ypos=np.random.randint(resol, size=nofdots)
    xvec=np.random.randint(-2,3, size=nofdots)
    yvec=np.random.randint(-2,3, size=nofdots)
    for i in range(maxtime):
        movie[xpos,ypos,i]=0
        if np.remainder(i,5) == 0:
                xvec=np.random.randint(-2,3, size=nofdots)
                yvec=np.random.randint(-2,3, size=nofdots)
                if mostart<i<mostop:
                    seq=np.random.permutation(np.linspace(0,nofdots-1,nofdots))
                    xvec[seq.astype(int)[0:nofdots/100*perccoher]]=direction
                    yvec[seq.astype(int)[0:nofdots/100*perccoher]]=0
        xpos=np.remainder(xpos+xvec,resol)
        ypos=np.remainder(ypos+yvec,resol)
        movie[ypos,xpos,i]=10.0
    movie=rebin(movie,200,200,maxtime)
    if blurrindex==1:
        for i in range(maxtime):
            print (i,)
            movie[:,:,i]=blurr(movie[:,:,i],5)
    plt.imshow(np.transpose(movie[0,:,:]), cmap='gray')
    return movie
    
def calc_PDND_motion_noise(maxtime,perccoher,blurrindex):
    
    mystim1=calc_motion_noise(maxtime/2,1,perccoher,blurrindex)
    mystim2=calc_motion_noise(maxtime/2,-1,perccoher,blurrindex)
    output=np.concatenate((mystim1,mystim2),axis=2)
    
    return output
    