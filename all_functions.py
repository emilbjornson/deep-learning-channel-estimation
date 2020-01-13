# This Python script consists of the functions needed to generate the figures in the paper:
#
#Özlem Tugfe Demir, Emil Bjornson, "Channel Estimation in Massive MIMO under Hardware
#Non-Linearities: Bayesian Methods versus Deep Learning,"
#IEEE Open Journal of the Communications Society, To appear.
#
#Download article: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8933050
#
#This is version 1.0 (Last edited: 2020-01-13)
#
#License: This code is licensed under the GPLv2 license. If you in any way
#use this code for research that results in publications, please cite our
#paper as described above.


import numpy as np
from math import factorial as fac


# This function calculates the moments in (8) and (9) in the paper
# if selection=1: Gaussian signals
# if selection=2: QPSK signaling
# if selection=3: 16-QAM signaling
def moments(selection,tildeb):
    zeta=np.zeros((1,9))
    chi=np.zeros((1,3))

    if selection==1:
        for n in np.arange(9):
            zeta[:,n]=fac(n+1)
    if selection==2:
        zeta=np.ones((1,9))
    if selection==3:
        a=np.array([[ -3, -1, 1, 3 ]])/np.sqrt(10)
        b=np.array([[3], [1], [-1], [-3]])/np.sqrt(10)
        Q=a+1j*b
        A=np.ones((16,1))
        for tt in np.arange(9):
            A=A*np.real(Q.reshape((16,1))*np.conj(Q.reshape((16,1))))
            zeta[:,tt]=np.sum(A)/16
    
    

    polyn=np.array([np.square(np.abs(tildeb[:,1])), 2*np.real(tildeb[:,0]*np.conj(tildeb[:,1])), np.square(np.abs(tildeb[:,0]))])
    chi[:,0]=np.sum(polyn.T*zeta[:,2::-1])
    polyn2=np.convolve(polyn.T[0,:],polyn.T[0,:])
    chi[:,1]=np.sum(polyn2*zeta[:,5:0:-1])
    polyn3=np.convolve(polyn2,polyn.T[0,:])
    chi[:,2]=np.sum(polyn3*zeta[:,8:1:-1])
    return zeta,chi

# This function calculates the moments in (8) and (9) in the paper and additional moments for the calculations in (70)
# if selection=2: QPSK signaling
# if selection=3: 16-QAM signaling
def moments7(selection,tildeb):
    zeta=np.zeros((1,9))
    chi=np.zeros((1,3))
    chi2=np.zeros((1,4))+1j*np.zeros((1,4))
    
   
    if selection==2:
        zeta=np.ones((1,9))
        a=np.array([-1 ,1])/np.sqrt(2)
        b=np.array([[-1], [1]])/np.sqrt(2)
        Q=a+1j*b
        Q2=tildeb[0,0]*Q+tildeb[0,1]*np.square(np.abs(Q))*Q\
        +tildeb[0,2]*np.power(np.abs(Q),4)*Q+tildeb[0,3]*np.power(np.abs(Q),6)*Q
        for rr in np.arange(3):
            Q3=np.power(np.abs(Q2),2*(rr+1))
            chi[0,rr]=np.mean(Q3.reshape(4,1))
        for r in np.arange(4):
            Q3=np.power(Q2,r+1)*np.power(np.conj(Q2),r)*np.conj(Q)
            chi2[0,r]=np.mean(Q3.reshape(4,1))
        Q3=np.power(np.conj(Q2),3)*np.conj(Q)
        chi3=np.mean(Q3.reshape(4,1))
        Q3=np.power(Q2,4)
        chi4=np.mean(Q3.reshape(4,1))
    if selection==3:
        a=np.array([-3, -1, 1, 3])/np.sqrt(10)
        b=np.array([[3], [1], [-1], [-3]])/np.sqrt(10)
        Q=a+1j*b
        A=np.ones((16,1))
        for tt in np.arange(9):
            A=A*np.real(Q.reshape((16,1))*np.conj(Q.reshape((16,1))))
            zeta[:,tt]=np.sum(A)/16
        Q2=tildeb[0,0]*Q+tildeb[0,1]*np.square(np.abs(Q))*Q \
        +tildeb[0,2]*np.power(np.abs(Q),4)*Q+tildeb[0,3]*np.power(np.abs(Q),6)*Q
        for rr in np.arange(3):
            Q3=np.power(np.abs(Q2),2*(rr+1))
            chi[0,rr]=np.mean(Q3.reshape(16,1))
        for r in np.arange(4):
            Q3=np.power(Q2,r+1)*np.power(np.conj(Q2),r)*np.conj(Q)
            chi2[0,r]=np.mean(Q3.reshape(16,1))
        Q3=np.power(np.conj(Q2),3)*np.conj(Q)
        chi3=np.mean(Q3.reshape(16,1))
        Q3=np.power(Q2,4)
        chi4=np.mean(Q3.reshape(16,1))
    

    
    return zeta,chi,chi2,chi3,chi4


# This function calculates the equation (56) in the paper

def Hmxy(x,y,hmBar):
    Hm=np.matmul(np.conj(x.T),np.matmul(hmBar, np.matmul( np.conj(hmBar.T),y)))
    return Hm

# This function calculates the effective channels in (69) in the paper for seventh-order non-linearities

def effChannelseventh(M,K,tildea,G,chi,chi2,chi3,chi4):
    effChannel=np.zeros((M,K))+1j*np.zeros((M,K))
    
    for m in np.arange(M):
        AAA1=np.sum(np.square(np.abs(G[m,:])))
        AAA2=np.sum(np.power(np.abs(G[m,:]),4))
        AAA3=np.sum(np.power(np.abs(G[m,:]),6))
        AAA4=np.sum(np.power(G[m,:],4))
        for k in np.arange(K):
            BBB1=AAA1-np.square(np.abs(G[m,k]))
            BBB2=AAA2-np.power(np.abs(G[m,k]),4)
            BBB3=AAA3-np.power(np.abs(G[m,k]),6)
            BBB4=AAA4-np.power(G[m,k],4)
            effChannel[m,k]=tildea[0,0]*chi2[0,0]*G[m,k] \
            +tildea[0,1]*chi2[0,1]*np.square(np.abs(G[m,k]))*G[m,k] \
            +tildea[0,1]*2*chi2[0,0]*chi[0,0]*G[m,k]*BBB1 \
            +tildea[0,2]*chi2[0,2]*np.power(np.abs(G[m,k]),4)*G[m,k] \
            +tildea[0,2]*6*chi2[0,1]*chi[0,0]*np.square(np.abs(G[m,k]))*G[m,k]*BBB1 \
            +tildea[0,2]*3*chi2[0,0]*chi[0,1]*G[m,k]*BBB2 \
            +tildea[0,2]*6*chi2[0,0]*chi[0,0]*chi[0,0]*G[m,k]*(BBB1*BBB1-BBB2) \
            +tildea[0,3]*chi2[0,3]*np.power(np.abs(G[m,k]),6)*G[m,k] \
            +tildea[0,3]*12*chi2[0,2]*chi[0,0]*np.power(np.abs(G[m,k]),4)*G[m,k]*BBB1 \
            +tildea[0,3]*18*chi2[0,1]*chi[0,1]*np.square(np.abs(G[m,k]))*G[m,k]*BBB2 \
            +tildea[0,3]*4*chi2[0,0]*chi[0,2]*G[m,k]*BBB3 \
            +tildea[0,3]*36*chi2[0,1]*chi[0,0]*chi[0,0]*np.square(np.abs(G[m,k]))*G[m,k]*(BBB1*BBB1-BBB2)\
            +tildea[0,3]*36*chi2[0,0]*chi[0,1]*chi[0,0]*G[m,k]*(BBB2*BBB1-BBB3) \
            +tildea[0,3]*24*chi2[0,0]*chi[0,0]*chi[0,0]*chi[0,0]*G[m,k]*(BBB1*BBB1*BBB1-3*BBB2*BBB1+2*BBB3)\
            +tildea[0,3]*chi3*chi4*np.conj(G[m,k]*G[m,k]*G[m,k])*BBB4
            
    return effChannel

 
   
# This function calculates the effective channels in (20) in the paper for third-order non-linearities
def effChannelthird(M,K,tildea,tildeb,G,zeta):

    effChannel=np.zeros((M,K))+1j*np.zeros((M,K))
    # equations (18) and (19)
    B2=np.zeros((2,2))+1j*np.zeros((2,2))
    B3=np.zeros((2,2,2))+1j*np.zeros((2,2,2))
    for i1 in np.arange(2):
        for i2 in np.arange(2):
            B2[i1,i2]=tildeb[0,i1]*np.conj(tildeb[0,i2])
            for i3 in np.arange(2):
                B3[i1,i2,i3]=tildeb[0,i1]*np.conj(tildeb[0,i2])*tildeb[0,i3]
        
    for m in np.arange(M):
        for k in np.arange(K):
            effChannel[m,k]=tildea[0,0]*G[m,k]*(tildeb[0,0]+zeta[0,1]*tildeb[0,1]) \
            +tildea[0,1]*np.square(np.abs(G[m,k]))*G[m,k]*(zeta[0,4]*B3[1,1,1]+2*zeta[0,3]*B3[1,1,0] \
            +zeta[0,3]*B3[1,0,1]+2*zeta[0,2]*B3[0,0,1]+zeta[0,2]*B3[0,1,0]+zeta[0,1]*B3[0,0,0]) \
            +2*tildea[0,1]*G[m,k]*(tildeb[0,0]+zeta[0,1]*tildeb[0,1])*(zeta[0,2]*B2[1,1]+zeta[0,1]*B2[1,0] \
            +zeta[0,1]*B2[0,1]+B2[0,0])*(np.sum(np.square(np.abs(G[m,:])))-np.square(np.abs(G[m,k])))

    return effChannel

# This function calculates the equation in claim 1 of Lemma 3
def lemma31(a1,b1,hmBar):

    Em1=Hmxy(a1,b1,hmBar)+np.sum(np.conj(a1)*b1)
    return Em1


# This function calculates the equation in claim 2 of Lemma 3
def lemma32(a1,b1,a2,b2,hmBar):

    Em2=Hmxy(a1,b1,hmBar)*Hmxy(a2,b2,hmBar)+Hmxy(a1,b1,hmBar)*np.sum(np.conj(a2)*b2) \
    +Hmxy(a2,b2,hmBar)*np.sum(np.conj(a1)*b1)+Hmxy(a1,b2,hmBar)*np.sum(np.conj(a2)*b1) \
    +Hmxy(a2,b1,hmBar)*np.sum(np.conj(a1)*b2)+np.sum(np.conj(a1)*b1)*np.sum(np.conj(a2)*b2) \
    +np.sum(np.conj(a1)*b2)*np.sum(np.conj(a2)*b1)

    return Em2

# This function calculates the equation in claim 3 of Lemma 3
def lemma33(a1,b1,a2,b2,a3,b3,hmbar):

    Em3=Hmxy(a1,b1,hmbar)*Hmxy(a2,b2,hmbar)*Hmxy(a3,b3,hmbar) \
    +Hmxy(a1,b1,hmbar)*Hmxy(a2,b2,hmbar)*np.sum(np.conj(a3)*b3) \
    +Hmxy(a1,b1,hmbar)*Hmxy(a2,b3,hmbar)*np.sum(np.conj(a3)*b2) \
    +Hmxy(a1,b3,hmbar)*Hmxy(a2,b2,hmbar)*np.sum(np.conj(a3)*b1) \
    +Hmxy(a1,b1,hmbar)*Hmxy(a3,b2,hmbar)*np.sum(np.conj(a2)*b3) \
    +Hmxy(a1,b1,hmbar)*Hmxy(a3,b3,hmbar)*np.sum(np.conj(a2)*b2) \
    +Hmxy(a1,b2,hmbar)*Hmxy(a3,b3,hmbar)*np.sum(np.conj(a2)*b1) \
    +Hmxy(a2,b2,hmbar)*Hmxy(a3,b1,hmbar)*np.sum(np.conj(a1)*b3) \
    +Hmxy(a2,b1,hmbar)*Hmxy(a3,b3,hmbar)*np.sum(np.conj(a1)*b2) \
    +Hmxy(a2,b2,hmbar)*Hmxy(a3,b3,hmbar)*np.sum(np.conj(a1)*b1) \
    +Hmxy(a1,b1,hmbar)*(np.sum(np.conj(a2)*b2)*np.sum(np.conj(a3)*b3)+np.sum(np.conj(a2)*b3)*np.sum(np.conj(a3)*b2)) \
    +Hmxy(a1,b2,hmbar)*(np.sum(np.conj(a2)*b1)*np.sum(np.conj(a3)*b3)+np.sum(np.conj(a2)*b3)*np.sum(np.conj(a3)*b1)) \
    +Hmxy(a1,b3,hmbar)*(np.sum(np.conj(a2)*b2)*np.sum(np.conj(a3)*b1)+np.sum(np.conj(a2)*b1)*np.sum(np.conj(a3)*b2)) \
    +Hmxy(a2,b1,hmbar)*(np.sum(np.conj(a1)*b2)*np.sum(np.conj(a3)*b3)+np.sum(np.conj(a1)*b3)*np.sum(np.conj(a3)*b2)) \
    +Hmxy(a2,b2,hmbar)*(np.sum(np.conj(a1)*b1)*np.sum(np.conj(a3)*b3)+np.sum(np.conj(a1)*b3)*np.sum(np.conj(a3)*b1)) \
    +Hmxy(a2,b3,hmbar)*(np.sum(np.conj(a1)*b1)*np.sum(np.conj(a3)*b2)+np.sum(np.conj(a1)*b2)*np.sum(np.conj(a3)*b1)) \
    +Hmxy(a3,b1,hmbar)*(np.sum(np.conj(a1)*b3)*np.sum(np.conj(a2)*b2)+np.sum(np.conj(a1)*b2)*np.sum(np.conj(a2)*b3)) \
    +Hmxy(a3,b2,hmbar)*(np.sum(np.conj(a1)*b1)*np.sum(np.conj(a2)*b3)+np.sum(np.conj(a1)*b3)*np.sum(np.conj(a2)*b1)) \
    +Hmxy(a3,b3,hmbar)*(np.sum(np.conj(a1)*b1)*np.sum(np.conj(a2)*b2)+np.sum(np.conj(a1)*b2)*np.sum(np.conj(a2)*b1)) \
    +np.sum(np.conj(a1)*b1)*(np.sum(np.conj(a2)*b2)*np.sum(np.conj(a3)*b3)+np.sum(np.conj(a2)*b3)*np.sum(np.conj(a3)*b2)) \
    +np.sum(np.conj(a1)*b2)*(np.sum(np.conj(a2)*b1)*np.sum(np.conj(a3)*b3)+np.sum(np.conj(a2)*b3)*np.sum(np.conj(a3)*b1)) \
    +np.sum(np.conj(a1)*b3)*(np.sum(np.conj(a2)*b2)*np.sum(np.conj(a3)*b1)+np.sum(np.conj(a2)*b1)*np.sum(np.conj(a3)*b2)) 

    
    return Em3


# This function calculates the required matrices for theoretical LMMSE-based distortion-aware effective channel estimation
def lmmseTheory(tildea,tildec, gBar, beta,tildeP,eta):
    
    K=tildeP.shape[0]
    pilotSize=tildeP.shape[1]
    hbar=np.zeros((K,1))+1j*np.zeros((K,1))
    for k in np.arange(K):
        if np.sqrt(beta[k,0])>0:
            hbar[k,0]=gBar[0,k]/np.sqrt(beta[k,0])
    yBar=np.zeros((pilotSize,1))+1j*np.zeros((pilotSize,1))
    Cbar=np.zeros((K,1))+1j*np.zeros((K,1))
    CCy=np.zeros((K,pilotSize))+1j*np.zeros((K,pilotSize))
    Cyy=np.zeros((pilotSize,pilotSize))+1j*np.zeros((pilotSize,pilotSize))
    ek=np.diag(np.sqrt(beta[:,0]*eta[:,0]))

    for p3 in np.arange(pilotSize):
        yBar[p3,0]=np.sum(np.conj(tildeP[:,p3:p3+1])*hbar)*(tildea[0,0]+tildea[0,1]*np.square(np.abs(np.sum(np.conj(tildeP[:,p3:p3+1])*hbar))) \
        +2*tildea[0,1]*np.sum(np.conj(tildeP[:,p3:p3+1])*tildeP[:,p3:p3+1]))


    for k in np.arange(K):
        Cbar[k,0]=gBar[0,k]*np.sqrt(eta[k,0])*(tildec[0,0]+tildec[0,1]*np.square(np.abs(gBar[0,k]))*eta[k,0]+tildec[0,1]*2*beta[k,0]*eta[k,0] \
        +tildec[0,2]*(np.sum(beta*eta)-beta[k,0]*eta[k,0]+np.sum(np.square(np.abs(gBar.T))*eta)-np.square(np.abs(gBar[0,k]))*eta[k,0]))


    for p1 in np.arange(pilotSize):
        for k in np.arange(K):
            
            summ2=0+1j*0
            summ3=0+1j*0
            
            for k2 in np.arange(K):
                summ2=summ2+lemma32(ek[:,k2:k2+1],ek[:,k2:k2+1],ek[:,k:k+1],tildeP[:,p1:p1+1],hbar)
                summ3=summ3+lemma33(ek[:,k2:k2+1],ek[:,k2:k2+1],ek[:,k:k+1],tildeP[:,p1:p1+1],tildeP[:,p1:p1+1],tildeP[:,p1:p1+1],hbar)
    
    
            CCy[k,p1]=tildec[0,0]*np.conj(tildea[0,0])*lemma31(ek[:,k:k+1],tildeP[:,p1:p1+1],hbar) \
            +tildec[0,0]*np.conj(tildea[0,1])*lemma32(ek[:,k:k+1],tildeP[:,p1:p1+1],tildeP[:,p1:p1+1],tildeP[:,p1:p1+1],hbar) \
            +(tildec[0,1]*np.conj(tildea[0,0])-tildec[0,2]*np.conj(tildea[0,0])) \
            *lemma32(ek[:,k:k+1],ek[:,k:k+1],ek[:,k:k+1],tildeP[:,p1:p1+1],hbar)+tildec[0,2]*np.conj(tildea[0,0])*summ2 \
            +tildec[0,2]*np.conj(tildea[0,1])*summ3-Cbar[k,0]*np.conj(yBar[p1,0]) \
            +(tildec[0,1]*np.conj(tildea[0,1])-tildec[0,2]*np.conj(tildea[0,1])) \
            *lemma33(ek[:,k:k+1],ek[:,k:k+1],ek[:,k:k+1],tildeP[:,p1:p1+1],tildeP[:,p1:p1+1],tildeP[:,p1:p1+1],hbar)
    
        
        
        for p2 in np.arange(pilotSize):
            
            Cyy[p1,p2]=np.square(np.abs(tildea[0,0]))*lemma31(tildeP[:,p1:p1+1],tildeP[:,p2:p2+1],hbar)\
            +tildea[0,0]*np.conj(tildea[0,1])*lemma32(tildeP[:,p1:p1+1],tildeP[:,p2:p2+1],tildeP[:,p2:p2+1],tildeP[:,p2:p2+1],hbar)\
            +tildea[0,1]*np.conj(tildea[0,0])*lemma32(tildeP[:,p1:p1+1],tildeP[:,p1:p1+1],tildeP[:,p1:p1+1],tildeP[:,p2:p2+1],hbar)\
            +np.square(np.abs(tildea[0,1]))*lemma33(tildeP[:,p1:p1+1],tildeP[:,p1:p1+1],tildeP[:,p1:p1+1], tildeP[:,p2:p2+1],tildeP[:,p2:p2+1],tildeP[:,p2:p2+1],hbar)\
            -np.sum(yBar[p1:p1+1,:]*np.conj(yBar[p2:p2+1,:]))
 


    Cyy=Cyy+np.eye(pilotSize)



    return yBar, Cbar, CCy, Cyy




#This function generates channels according to the models in reference [26]

def channel_generator(M,K,numberr,pmax):
    # standard deviation for shadow fading
    sigma_sf_NLOS=4 #for NLOS
    sigma_sf_LOS=3   #for LOS
    # Prepare LOS components of the channels
    GMean=np.zeros((M,K))+1j*np.zeros((M,K))
    # antenna spacing in wavelength to calculate LOS components of the channels
    antennaSpacing = 1/2 #Half wavelength distance
    # Prepare UE positions (real part is x-axis and imaginary part is y-axis components)
    UEpositions = np.zeros((K,1))+1j*np.zeros((K,1))
    # Variable indicating how many users' locations are determined  
    perBS = 0
    # UEs are located in a cell of maxDistance m x maxDistance m area
    maxDistance = 250
    # minimum distance of UEs to the BS by taking into account 8.5 height difference 
    minDistance=np.sqrt(10*10-8.5*8.5)
    # Prepare channel gains in dB
    channelGaindB=np.zeros((K,1))
    # continue until all the UEs' locations are determined
    while perBS<K:
        UEremaining = K-perBS
        posX = np.random.rand(UEremaining,1)*maxDistance - maxDistance/2
        posY = np.random.rand(UEremaining,1)*maxDistance - maxDistance/2
        posXY = posX + 1j*posY
        # Keep the UE if it satisfies minimum distance criterion
        posXY=posXY[np.abs(posXY)>minDistance]
        posXY=posXY.reshape(posXY.shape[0],1)
        UEpositions[perBS:perBS+posXY.shape[0],0:] = posXY
        # Increase the number of determined UEs
        perBS = perBS+posXY.shape[0]
      
    # BS height    
    hBS=10
    # UE height
    hUT=1.5
    # effective BS and UE heights in [26]
    hBS2=hBS-1
    hUT2=hUT-1
    # breakpoint distance in [26] where carrier frequency is 2GHz
    bpdist=4*hBS2*hUT2*20/3
    # 3D distances of UEs to the BS
    distancesBS = np.sqrt(np.square(np.abs(UEpositions))+8.5*8.5)
        
    
    # Prepare probabilities of LOS for UEs
    probLOSprep=np.zeros((K,1))
    for k in np.arange(K):
        probLOSprep[k,0]=min(18/distancesBS[k],1)*(1-np.exp(-distancesBS[k]/36))+np.exp(-distancesBS[k]/36)
    probLOS=(np.random.rand(K,1)<probLOSprep).astype(int)
    ricianFactor=np.power(10,(5*np.random.randn(K,1)+9)/10) 
    for k in np.arange(K):
            
        if probLOS[k,0]==1:
            if distancesBS[k]<bpdist:
                channelGaindB[k,0]=-22*np.log10(distancesBS[k])-28-20*np.log10(2)
            else:
                channelGaindB[k,0]=-40*np.log10(distancesBS[k])-7.8+18*np.log10(hBS2)+18*np.log10(hUT2)-2*np.log10(2)
        else:
            channelGaindB[k,0]=-36.7*np.log10(distancesBS[k])-22.7-26*np.log10(2)
        
    for k in np.arange(K):
        
        if probLOS[k,0]==1:
            shadowing =sigma_sf_LOS*np.random.randn(1,1)
            channelGainShadowing = channelGaindB[k,0] + shadowing
        else:
            shadowing = sigma_sf_NLOS*np.random.randn(1,1)
            channelGainShadowing = channelGaindB[k,0] + shadowing
        
        channelGaindB[k,0] = channelGainShadowing

    
    
    for k in np.arange(K):
            
        # Angle of the UEs with respect to the BS    
        angleBS = np.angle(UEpositions[k,0])
        # Add random phase shift to the LOS components of the channels for training neural networks
        anglerandom=2*np.pi*np.random.rand(1)
        # normalized LOS vector by assuming uniform linear array
        GMean[:,k]=np.exp(1j*anglerandom+1j*2*np.pi*np.arange(M)*np.sin(angleBS)*antennaSpacing) 
    
    # bandwidth in Hz
    B = 20e6
    # noise figure in dB
    noiseFigure = 5
    noiseVariancedBm = -174 + 10*np.log10(B) + noiseFigure
    # channel gain over noise in dB
    channelGainOverNoise = channelGaindB-noiseVariancedBm+30
    
    # apply the heuristic uplink power control in reference [5, Section 7.1.2] with delta=20 dB
    betaMin = np.min(channelGainOverNoise[channelGainOverNoise>-np.inf])

    deltadB=20

    differenceSNR = channelGainOverNoise-betaMin
    backoff = differenceSNR-deltadB
    backoff[backoff<0] = 0
    # p_k in the paper
    power_coef=pmax/np.power(10,backoff/10)
    
    # prepare LOS and NLOS channel gains
    channelGain_LOS=np.zeros((K,1))
    channelGain_NLOS=np.zeros((K,1))

    for k in np.arange(K):
    
        if probLOS[k,0]==1: # The LoS Path exists, Rician Factor ~= 0
            channelGain_LOS[k,0]= (ricianFactor[k,0]/(ricianFactor[k,0] +1 ))*np.power(10,channelGainOverNoise[k,0]/10)
            channelGain_NLOS[k,0]=(1/(ricianFactor[k,0] +1 ))*np.power(10,channelGainOverNoise[k,0]/10)
        else:  # Pure NLoS case
            channelGain_LOS[k,0]= 0
            channelGain_NLOS[k,0]=np.power(10,channelGainOverNoise[k,0]/10)
   
        GMean[:,k]=np.sqrt(channelGain_LOS[k,0])*GMean[:,k]
   
    # sort the UE indices according to their channel gains for improvement in deep learning        
    indexx=np.argsort(channelGainOverNoise[:,0])
    channelGainOverNoise=channelGainOverNoise[indexx,:]
    GMean=GMean[:,indexx]
    probLOS=probLOS[indexx,:]
    channelGain_NLOS=channelGain_NLOS[indexx,:]
    power_coef=power_coef[indexx,:]

    G=np.zeros((M,K,numberr))+1j*np.zeros((M,K,numberr))
    for nnn in np.arange(numberr):

        W=np.sqrt(0.5)*(np.random.randn(M,K)+1j*np.random.randn(M,K))

        G_Rayleigh=np.matmul(W,np.sqrt(np.diag(channelGain_NLOS[:,0])))

        G[:,:,nnn]=GMean+G_Rayleigh


    return G, channelGainOverNoise, channelGain_NLOS, GMean, power_coef

# This function evaluates the expectaion in claim 1 of Lemma 2
def lemma2ExpMatrix1(A,chi):

    expMatrix=np.square(chi[0,0])*A+np.square(chi[0,0])*np.trace(A)*np.eye(A.shape[0])+(chi[0,1]-2*np.square(chi[0,0]))*np.diag(np.diag(A))

    return expMatrix

# This function evaluates the expectaion in claim 2 of Lemma 2

def lemma2ExpMatrix2(A,B,chi):

    expMatrix=np.power(chi[0,0],3)*(np.matmul(A,B)+np.matmul(B,A)+np.trace(A)*B+np.trace(B)*A \
    +(np.trace(A)*np.trace(B)+np.trace(np.matmul(A,B)))*np.eye(A.shape[0])) \
    +(chi[0,1]*chi[0,0]-2*np.power(chi[0,0],3))*(np.matmul(np.diag(np.diag(A)),B) \
    +np.matmul(np.diag(np.diag(B)),A) \
    +np.matmul(A,np.diag(np.diag(B)))+np.matmul(B,np.diag(np.diag(A))) \
    +np.diag(np.diag(np.matmul(A,B)+np.matmul(B,A))) \
    +np.trace(A)*np.diag(np.diag(B))+np.trace(B)*np.diag(np.diag(A)) \
    +np.trace(np.matmul(np.diag(np.diag(A)),np.diag(np.diag(B))))*np.eye(A.shape[0])) \
    +(chi[0,2]-9*chi[0,1]*chi[0,0]+12*np.power(chi[0,0],3))*np.matmul(np.diag(np.diag(A)),np.diag(np.diag(B)))
   

    return expMatrix


# This function calculates the elements of distortion correlation matrix in (21)
def distortionCorr(tildea,G,chi,M,effChannel):

    Corr=np.zeros((M,M))+1j*np.zeros((M,M))

    a=tildea.copy()
    GG=np.conj(G.T)

    for m in np.arange(M):
        for n in np.arange(M):
            # Calculates the expectations in (24) using Lemma 2
            M1=lemma2ExpMatrix1(np.matmul(GG[:,m:m+1],np.conj(GG[:,m:m+1].T)),chi)
            M2=lemma2ExpMatrix1(np.matmul(GG[:,n:n+1],np.conj(GG[:,n:n+1].T)),chi)
            M3=lemma2ExpMatrix2(np.matmul(GG[:,m:m+1],np.conj(GG[:,m:m+1].T)),np.matmul(GG[:,n:n+1],np.conj(GG[:,n:n+1].T)),chi)


            Corr[m:m+1,n:n+1]=a[0,0]*np.conj(a[0,0])*chi[0,0]*np.matmul(np.conj(GG[:,m:m+1].T),GG[:,n:n+1]) \
            +a[0,1]*np.conj(a[0,0])*np.matmul(np.conj(GG[:,m:m+1].T),np.matmul(M1,GG[:,n:n+1])) \
            +a[0,0]*np.conj(a[0,1])*np.matmul(np.conj(GG[:,m:m+1].T),np.matmul(M2,GG[:,n:n+1])) \
            +a[0,1]*np.conj(a[0,1])*np.matmul(np.conj(GG[:,m:m+1].T),np.matmul(M3,GG[:,n:n+1]))
    


    distCorr=Corr+np.eye(M)-np.matmul(effChannel,np.conj(effChannel.T))

    return distCorr


# This function calculates the required matrices for Monte Carlo-based LMMSE estimation 
# If sel=0, the data is assumed to be zero-mean.
def lmmseMonteCarlo(obser,estim,sel):
    
    if sel==0:
        yBar1=np.zeros((obser.shape[0],1))+1j*np.zeros((obser.shape[0],1))
        Cbar1=np.zeros((estim.shape[0],1))+1j*np.zeros((estim.shape[0],1))
    else:
        yBar1=np.mean(obser,axis=1).reshape(obser.shape[0],1)
        Cbar1=np.mean(estim,axis=1).reshape(estim.shape[0],1)
    CCy1=np.matmul((estim-Cbar1),np.conj((obser-yBar1).T))/estim.shape[1]
    Cyy1=np.matmul((obser-yBar1),np.conj((obser-yBar1).T))/estim.shape[1]
    
    
    return yBar1, Cbar1, CCy1, Cyy1

# This function calculates the instantaneous rate by using the SINR in (25)
def insRate(v,effChannel,distCorr,K):

    ratee=np.zeros((K,1))
    A=effChannel.copy()
    B=distCorr+np.matmul(A,np.conj(A.T))
    for k in np.arange(K):
        numerat=np.real(np.matmul(np.conj(v[:,k:k+1].T),np.matmul(np.matmul(A[:,k:k+1],np.conj(A[:,k:k+1].T)),v[:,k:k+1])))
        denumer=np.real(np.matmul(np.conj(v[:,k:k+1].T),np.matmul(B,v[:,k:k+1])))

        ratee[k,0]=np.log2(1+numerat/(denumer-numerat))



    return ratee
        