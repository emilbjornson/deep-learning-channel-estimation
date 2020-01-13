# This Python script is used to generate Figure 9 in the paper:
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
import matplotlib
import matplotlib.pyplot as plt
from numpy.linalg import inv
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.linalg import dft
from keras.callbacks import EarlyStopping
from all_functions import moments, effChannelthird, lmmseTheory, channel_generator
from all_functions import distortionCorr

# number of BS antennas
M=100
# number of UEs
K=20
# Backoff parameter in dB
backOff=np.power(10,0.7)
# pilot matrix
pilot=dft(K)
# training data length
trainn=int(3e6)
# validation data length
valsize=int(2e5)
# number of UE location setups
test1=100
# number of Monte Carlo trials for each setup
test2=100

# Third-order polynomial coefficients obtained by curve fitting to [25]
polyCoef=np.array([[ 0.9798-0.0075*1j, -0.2419+0.0374*1j]])

# \tilde{b}_0 and \tilde{b}_1 in (7) in the paper
tildeb=polyCoef/np.array([[1, backOff]])
# moments in (8) and (9) in the paper, the first input is 2 for QPSK 
(zeta,chi)=moments(2,tildeb)
# maximum transmission power for UE antennas in Watts
pmax=0.2
# equations (18) and (19)
B2=np.zeros((2,2))+1j*np.zeros((2,2))
B3=np.zeros((2,2,2))+1j*np.zeros((2,2,2))
for i1 in np.arange(2):
    for i2 in np.arange(2):
        B2[i1,i2]=tildeb[0,i1]*np.conj(tildeb[0,i2])
        for i3 in np.arange(2):
            B3[i1,i2,i3]=tildeb[0,i1]*np.conj(tildeb[0,i2])*tildeb[0,i3]
# distorted pilot matrix in (31) in the paper  
pilott2=tildeb[0,0]*pilot+tildeb[0,1]*np.square(np.abs(pilot))*pilot

# prepare training & validation inputs & outputs
input_deep=np.zeros((trainn,3*K))
input_val=np.zeros((valsize,3*K))
output_deep=np.zeros((trainn,2*K+1))
output_val=np.zeros((valsize,2*K+1))
# generate the training data
for trialdeep in np.arange(trainn):
    
    (G,channelGainOverNoise,betaNLOS,GMean,power_coef)=channel_generator(1,K,1,pmax)
    # \eta_k in (15) in the paper
    eta=(power_coef/chi[0,0]).reshape(K,)
    # \tilde{\eta}_k in (32) assuming pilot matrix is DFT matrix
    eta_pilot=(power_coef/np.square(np.abs(tildeb[0,0]+tildeb[0,1]))).reshape(K,)

    # convert the channel gain over noise to the linear scale
    normalizedGain=np.power(10,channelGainOverNoise/10)
    # \tilde{a}_{0} and \tilde{a}_1 in (4) in the paper (same for each BS antenna)
    tildea=polyCoef/np.array([[1, backOff]])
    tildea[:,1]=tildea[:,1]/np.sum(normalizedGain*power_coef)
    
    # Give the scaled channel coefficients in (15) as input to the following functions
    effChannel=effChannelthird(1,K,tildea,tildeb,np.matmul(G[:,:,0],np.diag(np.sqrt(eta))),zeta)
    distt=distortionCorr(tildea,np.matmul(G[:,:,0],np.diag(np.sqrt(eta))),chi,1,effChannel)
    
    # transmit the uplink signals and add Gaussian noise at the BS antennas
    U=np.matmul(pilott2,np.matmul(G[:,:,0],np.diag(np.sqrt(eta_pilot))).T)
    Z=tildea[:,0]*U+tildea[:,1]*np.square(np.abs(U))*U
    Y=Z+np.sqrt(0.5)*(np.random.randn(K,1)+1j*np.random.randn(K,1))
    # Inputs and outputs for the neural network
    inputt=np.matmul(np.conj(pilot.T),Y)
    input_deep[trialdeep:trialdeep+1,0:K]=np.real(inputt).T
    input_deep[trialdeep:trialdeep+1,K:2*K]=np.imag(inputt).T
    input_deep[trialdeep:trialdeep+1,2*K:3*K]=np.sqrt(normalizedGain.T)
    output_deep[trialdeep:trialdeep+1,0:K]=np.real(effChannel)
    output_deep[trialdeep:trialdeep+1,K:2*K]=np.imag(effChannel)
    output_deep[trialdeep:trialdeep+1,2*K]=np.log10(np.real(distt))

# generate the validation data    
for trialval in np.arange(valsize):
    (G,channelGainOverNoise,betaNLOS,GMean,power_coef)=channel_generator(1,K,1,pmax)
    eta=(power_coef/chi[0,0]).reshape(K,)
    eta_pilot=(power_coef/np.square(np.abs(tildeb[0,0]+tildeb[0,1]))).reshape(K,)
    
    normalizedGain=np.power(10,channelGainOverNoise/10)
    tildea=polyCoef/np.array([[1, backOff]])
    tildea[:,1]=tildea[:,1]/np.sum(normalizedGain*power_coef)
    
    effChannel=effChannelthird(1,K,tildea,tildeb,np.matmul(G[:,:,0],np.diag(np.sqrt(eta))),zeta)
    distt=distortionCorr(tildea,np.matmul(G[:,:,0],np.diag(np.sqrt(eta))),chi,1,effChannel)

    U=np.matmul(pilott2,np.matmul(G[:,:,0],np.diag(np.sqrt(eta_pilot))).T)
    Z=tildea[:,0]*U+tildea[:,1]*np.square(np.abs(U))*U
    Y=Z+np.sqrt(0.5)*(np.random.randn(K,1)+1j*np.random.randn(K,1))
    
    inputt=np.matmul(np.conj(pilot.T),Y)
    input_val[trialval:trialval+1,0:K]=np.real(inputt).T
    input_val[trialval:trialval+1,K:2*K]=np.imag(inputt).T
    input_val[trialval:trialval+1,2*K:3*K]=np.sqrt(normalizedGain.T)
    
    output_val[trialval:trialval+1,0:K]=np.real(effChannel)
    output_val[trialval:trialval+1,K:2*K]=np.imag(effChannel)
    output_val[trialval:trialval+1,2*K]=np.log10(np.real(distt))


# eliminate the outliers for improved learning for training data length 3e6
aaaa=np.zeros((2*K+1,1))
for kk in np.arange(K):
    aaaa[kk,0]=(np.sort(np.abs(output_deep[:,kk]).T).T)[2990000]
    aaaa[kk+K,0]=(np.sort(np.abs(output_deep[:,kk+K]).T).T)[2990000]
aaaa[2*K,0]=(np.sort(np.abs(output_deep[:,2*K]).T).T)[2990000]


output_deep0=np.zeros((output_deep.shape))
output_val0=np.zeros((output_val.shape))
input_deep0=np.zeros((input_deep.shape))
input_val0=np.zeros((input_val.shape))
iii1=0
iii2=0
for tt in np.arange(trainn):
    doorr=1
    for kk in np.arange(K):
        if np.abs(output_deep[tt,kk])>aaaa[kk,0]:
            doorr=0
        if np.abs(output_deep[tt,K+kk])>aaaa[K+kk,0]:
            doorr=0
        if np.abs(output_deep[tt,2*K])>aaaa[2*K,0]:
            doorr=0
    if doorr>0.5:
        output_deep0[iii1,:]=output_deep[tt,:]
        input_deep0[iii1,:]=input_deep[tt,:]
        iii1=iii1+1
        
for tt in np.arange(valsize):
    doorr=1
    for kk in np.arange(K):
        if np.abs(output_val[tt,kk])>aaaa[kk,0]:
            doorr=0
        if np.abs(output_val[tt,K+kk])>aaaa[K+kk,0]:
            doorr=0  
        if np.abs(output_val[tt,2*K])>aaaa[2*K,0]:
            doorr=0 
    if doorr>0.5:
        output_val0[iii2,:]=output_val[tt,:]
        input_val0[iii2,:]=input_val[tt,:]
        iii2=iii2+1
                
input_deep00=input_deep0[0:iii1,:]
output_deep00=output_deep0[0:iii1,:]
input_val00= input_val0[0:iii2,:]
output_val00= output_val0[0:iii2,:]




# apply MinMaxScaler and StandardScaler as in the paper

output_deep1=np.zeros((output_deep00.shape))
output_val1=np.zeros((output_val00.shape))

scaler22= MinMaxScaler(feature_range=(0.1,0.9))
output_deep1[:,0:2*K]=output_deep00[:,0:2*K]
output_val1[:,0:2*K]=output_val00[:,0:2*K]
output_deep1[:,2*K:]=scaler22.fit_transform(output_deep00[:,2*K:])
output_val1[:,2*K:]=scaler22.transform(output_val00[:,2*K:])

scaler= StandardScaler()
scaler2 = MinMaxScaler(feature_range=(0.1,0.9))
input_deep1=np.zeros((input_deep00.shape))
input_val1=np.zeros((input_val00.shape))
input_deep1[:,0:2*K]=scaler.fit_transform(input_deep00[:,0:2*K])
input_deep1[:,2*K:]=scaler2.fit_transform(input_deep00[:,2*K:])

input_val1[:,0:2*K]=scaler.transform(input_val00[:,0:2*K])
input_val1[:,2*K:]=scaler2.transform(input_val00[:,2*K:])

# deep learning for effective channel estimation
estimator1 = Sequential()
estimator1.add(Dense(units=30*K, kernel_initializer = 'glorot_normal'))
estimator1.add(Activation('relu'))
estimator1.add(Dense(units=30*K, kernel_initializer = 'glorot_normal'))
estimator1.add(Activation('relu'))
estimator1.add(Dense(units=2*K, kernel_initializer = 'glorot_normal', activation = 'linear'))
callbacks = [EarlyStopping(monitor='val_loss', patience=5)]
estimator1.compile(optimizer ='adam', loss = 'mean_squared_error', metrics = ['mse'])
estimator1.fit(input_deep1, 
                      output_deep1[:,0:2*K], 
                      epochs=50, 
                      callbacks=callbacks, 
                      batch_size=1000, 
                      validation_data=(input_val1, output_val1[:,0:2*K])) 

# deep learning for distortion correlation receiver
estimator2 = Sequential()
estimator2.add(Dense(units=30*K, kernel_initializer = 'glorot_normal'))
estimator2.add(Activation('relu'))
estimator2.add(Dense(units=30*K, kernel_initializer = 'glorot_normal'))
estimator2.add(Activation('relu'))
estimator2.add(Dense(units=1, kernel_initializer = 'glorot_normal', activation = 'relu'))
callbacks = [EarlyStopping(monitor='val_loss', patience=5)]
estimator2.compile(optimizer ='adam', loss = 'mean_squared_error', metrics = ['mse'])
estimator2.fit(input_deep1, 
                      output_deep1[:,2*K:], 
                      epochs=50, 
                      callbacks=callbacks, 
                      batch_size=1000, 
                      validation_data=(input_val1, output_val1[:,2*K:])) 


# prepare arrays for the BER results
berRZF_DuA=np.zeros((test1,K))
berRZF_DA_LMMSE=np.zeros((test1,K))
berRZF_DL=np.zeros((test1,K))
berEW_DL=np.zeros((test1,K))
berRZF_perfect=np.zeros((test1,K))
berEW_perfect=np.zeros((test1,K))
# go through each setup
for trialtest1 in np.arange(test1):
    # prepare arrays for each setup
    sRZF_DuA=np.zeros(K)
    sRZF_DA_LMMSE=np.zeros(K)
    sRZF_DL=np.zeros(K)
    sEW_DL=np.zeros(K)
    sRZF_perfect=np.zeros(K)
    sEW_perfect=np.zeros(K)
    
    input_test=np.zeros((1,3*K))
    (G,channelGainOverNoise,betaNLOS,GMean,power_coef)=channel_generator(M,K,test2,pmax)
    eta=(power_coef/chi[0,0]).reshape(K,)
    eta_pilot=(power_coef/np.square(np.abs(tildeb[0,0]+tildeb[0,1]))).reshape(K,)
    normalizedGain=np.power(10,channelGainOverNoise/10)
    tildea=polyCoef/np.array([[1, backOff]])
    tildea[:,1]=tildea[:,1]/np.sum(normalizedGain*power_coef)
    tildec=np.zeros((1,3))+1j*np.zeros((1,3))
    tildec[:,0]=tildea[:,0]*(tildeb[:,0]+zeta[:,1]*tildeb[:,1])
    tildec[:,1]=tildea[:,1]*(zeta[:,4]*B3[1,1,1]+2*zeta[:,3]*B3[1,1,0] \
    +zeta[:,3]*B3[1,0,1]+2*zeta[:,2]*B3[0,0,1]+zeta[:,2]*B3[0,1,0]+zeta[:,1]*B3[0,0,0])
    tildec[:,2]=2*tildea[:,1]*(tildeb[:,0]+zeta[:,1]*tildeb[:,1])*(zeta[:,2]*B2[1,1]+zeta[:,1]*B2[1,0] \
    +zeta[:,1]*B2[0,1]+B2[0,0])
    
    tildeP=np.zeros((K,K))+1j*np.zeros((K,K))
    for n in np.arange(K):
        tildeP[:,n:n+1]=np.sqrt(betaNLOS*eta_pilot.reshape(K,1))*np.conj(pilott2[n:n+1,:].T)
  

        
    (yBar1b, Cbar1b, CCy1b, Cyy1b)=lmmseTheory(tildea,tildec, GMean[0:1,:], betaNLOS,tildeP,eta.reshape(K,1))
    yBar1b_=np.zeros((yBar1b.shape[0],yBar1b.shape[1],M))+1j*np.zeros((yBar1b.shape[0],yBar1b.shape[1],M))
    Cbar1b_=np.zeros((Cbar1b.shape[0],Cbar1b.shape[1],M))+1j*np.zeros((Cbar1b.shape[0],Cbar1b.shape[1],M))
    CCy1b_=np.zeros((CCy1b.shape[0],CCy1b.shape[1],M))+1j*np.zeros((CCy1b.shape[0],CCy1b.shape[1],M))
    invCyy1b_=np.zeros((Cyy1b.shape[0],Cyy1b.shape[1],M))+1j*np.zeros((Cyy1b.shape[0],Cyy1b.shape[1],M))
    for mm in np.arange(M):
        (yBar1b, Cbar1b, CCy1b, Cyy1b)=lmmseTheory(tildea,tildec, GMean[mm:mm+1,:], betaNLOS,tildeP,eta.reshape(K,1))
        yBar1b_[:,:,mm]=yBar1b
        Cbar1b_[:,:,mm]=Cbar1b
        CCy1b_[:,:,mm]=CCy1b
        invCyy1b_[:,:,mm]=inv(Cyy1b)
    # go through each channel reallization
    for trialtest2 in np.arange(test2):
        deep_channel=np.zeros((M,K))+1j*np.zeros((M,K))
        perfect_channel=np.zeros((M,K))+1j*np.zeros((M,K))
        lmmse_channel=np.zeros((M,K))+1j*np.zeros((M,K))
        lmmse_unaware=np.zeros((M,K))+1j*np.zeros((M,K))
        y_est4=np.zeros((M,1))
        y_est4_=np.zeros((M,1))
        # g through each antenna
        for mm in np.arange(M):
            effChannel=effChannelthird(1,K,tildea,tildeb,np.matmul(G[mm:mm+1,:,trialtest2],np.diag(np.sqrt(eta))),zeta)
            distt=distortionCorr(tildea,np.matmul(G[mm:mm+1,:,trialtest2],np.diag(np.sqrt(eta))),chi,1,effChannel)

            U=np.matmul(pilott2,np.matmul(G[mm:mm+1,:,trialtest2],np.diag(np.sqrt(eta_pilot))).T)
            Z=tildea[:,0]*U+tildea[:,1]*np.square(np.abs(U))*U
            Y=Z+np.sqrt(0.5)*(np.random.randn(K,1)+1j*np.random.randn(K,1))
            inputt=np.matmul(np.conj(pilot.T),Y)
            input_test[:,0:K]=np.real(inputt).T
            input_test[:,K:2*K]=np.imag(inputt).T
            input_test[:,2*K:3*K]=np.sqrt(normalizedGain.T)

            input_test1=np.zeros((input_test.shape))

            input_test1[:,0:2*K]=scaler.transform(input_test[:,0:2*K])
            input_test1[:,2*K:]=scaler2.transform(input_test[:,2*K:])
            y_est2=estimator1.predict(input_test1)
            y_est3=estimator2.predict(input_test1)

            y_est4[mm:mm+1,:]=scaler22.inverse_transform(y_est3)
            y_est4_[mm:mm+1,:]=np.real(distt)
            deep_channel[mm:mm+1,:]=y_est2[:,0:K]+1j*y_est2[:,K:2*K]
            perfect_channel[mm:mm+1,:]=effChannel
            lmmse_channel[mm:mm+1,:]=(Cbar1b_[:,:,mm]+np.matmul(CCy1b_[:,:,mm],np.matmul(invCyy1b_[:,:,mm],Y-yBar1b_[:,:,mm]))).T

            for kk in np.arange(K):
            
                lmmse_unaware[mm:mm+1,kk]=np.sqrt(power_coef[kk,0])*GMean[mm,kk]+(power_coef[kk,0]*betaNLOS[kk,0])/(K*power_coef[kk,0]*betaNLOS[kk,0]+1) \
                *(inputt[kk,0]-np.sqrt(power_coef[kk,0])*K*GMean[mm,kk])

    
    
        SymbolInterval=10000
        symbolReal=np.sqrt(0.5)*(np.sign(np.random.rand(K,SymbolInterval)-0.5))
        symbolImag=np.sqrt(0.5)*(np.sign(np.random.rand(K,SymbolInterval)-0.5))
        symbol=symbolReal+1j*symbolImag
        distSymbol=tildeb[0,0]*symbol+tildeb[0,1]*np.square(np.abs(symbol))*symbol
        receivedU=np.matmul(np.matmul(G[:,:,trialtest2],np.diag(np.sqrt(eta))),distSymbol)
        receivedZ=np.matlib.repmat(tildea[0,0],1,SymbolInterval)*receivedU \
        +np.matlib.repmat(tildea[0,1],1,SymbolInterval)*np.square(np.abs(receivedU))*receivedU 
        receivedY=receivedZ+np.sqrt(0.5)*(np.random.randn(M,SymbolInterval)+1j*np.random.randn(M,SymbolInterval))
        vRZF_DL=np.matmul(deep_channel,inv(np.matmul(np.conj(deep_channel.T),deep_channel)+np.eye(K)))
        vRZF_DA_LMMSE=np.matmul(lmmse_channel,inv(np.matmul(np.conj(lmmse_channel.T),lmmse_channel)+np.eye(K)))
        vRZF_perfect=np.matmul(perfect_channel,inv(np.matmul(np.conj(perfect_channel.T),perfect_channel)+np.eye(K)))

        vRZF_DuA=np.matmul(lmmse_unaware,inv(np.matmul(np.conj(lmmse_unaware.T),lmmse_unaware)+np.eye(K)))

        vEW_DL=np.zeros((M,K))+1j*np.zeros((M,K))
        vEW_perfect=np.zeros((M,K))+1j*np.zeros((M,K))
        Cmumu=np.power(10,y_est4)*np.eye(M)+np.matmul(deep_channel,np.conj(deep_channel.T))
        Cmumu_=y_est4_*np.eye(M)+np.matmul(perfect_channel,np.conj(perfect_channel.T))
        for k in np.arange(K):
            vEW_DL[:,k:k+1]=np.matmul(inv(Cmumu-np.matmul(deep_channel[:,k:k+1],np.conj(deep_channel[:,k:k+1].T))),deep_channel[:,k:k+1])
            vEW_perfect[:,k:k+1]=np.matmul(inv(Cmumu_-np.matmul(perfect_channel[:,k:k+1],np.conj(perfect_channel[:,k:k+1].T))),perfect_channel[:,k:k+1])
        
        
        
        processedRZF_DL=np.matmul(np.conj(vRZF_DL.T),receivedY)
        processedRZF_DA_LMMSE=np.matmul(np.conj(vRZF_DA_LMMSE.T),receivedY)
        processedRZF_perfect=np.matmul(np.conj(vRZF_perfect.T),receivedY)
        processedRZF_DuA=np.matmul(np.conj(vRZF_DuA.T),receivedY)
        processedEW_DL=np.matmul(np.conj(vEW_DL.T),receivedY)
        processedEW_perfect=np.matmul(np.conj(vEW_perfect.T),receivedY)
        
        diffRZF1_DL=np.abs(np.sign(np.real(processedRZF_DL))-np.sign(symbolReal))
        diffRZF1_DA_LMMSE=np.abs(np.sign(np.real(processedRZF_DA_LMMSE))-np.sign(symbolReal))
        diffRZF1_perfect=np.abs(np.sign(np.real(processedRZF_perfect))-np.sign(symbolReal))
        diffRZF1_DuA=np.abs(np.sign(np.real(processedRZF_DuA))-np.sign(symbolReal))
        diffEW1_DL=np.abs(np.sign(np.real(processedEW_DL))-np.sign(symbolReal))
        diffEW1_perfect=np.abs(np.sign(np.real(processedEW_perfect))-np.sign(symbolReal))
        
        diffRZF2_DL=np.abs(np.sign(np.imag(processedRZF_DL))-np.sign(symbolImag))
        diffRZF2_DA_LMMSE=np.abs(np.sign(np.imag(processedRZF_DA_LMMSE))-np.sign(symbolImag))
        diffRZF2_perfect=np.abs(np.sign(np.imag(processedRZF_perfect))-np.sign(symbolImag))
        diffRZF2_DuA=np.abs(np.sign(np.imag(processedRZF_DuA))-np.sign(symbolImag))
        diffEW2_DL=np.abs(np.sign(np.imag(processedEW_DL))-np.sign(symbolImag))
        diffEW2_perfect=np.abs(np.sign(np.imag(processedEW_perfect))-np.sign(symbolImag))
        
        
        for k in np.arange(K):
            sRZF_DL[k]=sRZF_DL[k]+np.sum(diffRZF1_DL[k,:]>0.5)+np.sum(diffRZF2_DL[k,:]>0.5)
            sRZF_DA_LMMSE[k]=sRZF_DA_LMMSE[k]+np.sum(diffRZF1_DA_LMMSE[k,:]>0.5)+np.sum(diffRZF2_DA_LMMSE[k,:]>0.5)
            sRZF_perfect[k]=sRZF_perfect[k]+np.sum(diffRZF1_perfect[k,:]>0.5)+np.sum(diffRZF2_perfect[k,:]>0.5)
            sRZF_DuA[k]=sRZF_DuA[k]+np.sum(diffRZF1_DuA[k,:]>0.5)+np.sum(diffRZF2_DuA[k,:]>0.5)
            sEW_DL[k]=sEW_DL[k]+np.sum(diffEW1_DL[k,:]>0.5)+np.sum(diffEW2_DL[k,:]>0.5)
            sEW_perfect[k]=sEW_perfect[k]+np.sum(diffEW1_perfect[k,:]>0.5)+np.sum(diffEW2_perfect[k,:]>0.5)

    
    berRZF_DL[trialtest1:trialtest1+1,:]=sRZF_DL.reshape(1,K)/(2*test1*test2) 
    berRZF_DA_LMMSE[trialtest1:trialtest1+1,:]=sRZF_DA_LMMSE.reshape(1,K)/(2*test1*test2) 
    berRZF_perfect[trialtest1:trialtest1+1,:]=sRZF_perfect.reshape(1,K)/(2*test1*test2)  
    berRZF_DuA[trialtest1:trialtest1+1,:]=sRZF_DuA.reshape(1,K)/(2*test1*test2) 
    berEW_perfect[trialtest1:trialtest1+1,:]=sEW_perfect.reshape(1,K)/(2*test1*test2) 
    berEW_DL[trialtest1:trialtest1+1,:]=sEW_DL.reshape(1,K)/(2*test1*test2) 
    



# x axis values (user indices)
xax=np.arange(1, 8) 

A=np.mean(berRZF_DuA,axis=0)
B=np.mean(berRZF_DA_LMMSE,axis=0)
C=np.mean(berRZF_DL,axis=0)
D=np.mean(berEW_DL,axis=0)
E=np.mean(berRZF_perfect,axis=0)
F=np.mean(berEW_perfect,axis=0)

plt.figure(7)
plt.plot(xax,A[0:8],xax,B[0:8], xax,C[0:8],xax, D[0:8], xax, E[0:8], xax, F[0:8])
plt.grid()
matplotlib.rcParams.update({'font.size': 50,})
plt.rc('text', usetex=True)
plt.ylabel(r'BER',fontsize=50)
plt.xlabel(r'UE index starting from the worst',fontsize=50)
plt.yticks([ 1e-8,1e-7, 1e-6, 1e-5 ,1e-4,1e-3, 1e-2, 1e-1])
plt.tick_params(axis='both', which='major', labelsize=50)
plt.legend((r'DuA-RZF (LMMSE)', r'DA-RZF (LMMSE)', r'DA-RZF (Deep Learning)',r'EW-DA-MMSE (Deep Learning)', \
            r'DA-RZF (Perfect CSI)', r'EW-DA-MMSE (Perfect CSI)'),ncol=2)
