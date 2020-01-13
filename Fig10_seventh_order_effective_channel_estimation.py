# This Python script is used to generate Figure 10 in the paper:
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
from numpy import linalg as LA
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.linalg import dft
from keras.callbacks import EarlyStopping
from all_functions import channel_generator
from all_functions import moments7, effChannelseventh, lmmseMonteCarlo
# number of BS antennas, it is taken as 1 for estimation comparisons without loss of generality
M=1
# number of UEs
K=10
# Backoff parameter in dB
backOff=np.power(10,0.7)
# pilot matrix
pilot=dft(K)
# training data length
trainn=int(3e6)
# validation data length
valsize=int(2e5)
# testing data lengths 
# number of UE location setups
test1=1000
# number of Monte Carlo trials for averaging in the CDF plots
test2=1000
# Seventh-order polynomial coefficients obtained by curve fitting to [25]
polyCoef=np.array([[ 1.0470+0.0020*1j, -0.7512-0.0544*1j, 0.9866+0.2099*1j, -0.5533-0.1333*1j ]])

# \tilde{b}_0 and \tilde{b}_1 in (64) in the paper
tildeb=polyCoef/np.array([[1, backOff, np.square(backOff), np.power(backOff,3)]])
# moments in (8) and (9) in the paper and also addditional moments for evaluating (70), the first input is 2 for QPSK and 3 for 16-QAM
(zeta,chi,chi2,chi3,chi4)=moments7(2,tildeb)
# maximum transmission power for UE antennas in Watts
pmax=0.2

# distorted pilot matrix
pilott2=(np.sum(tildeb[0,:])*pilot)

# prepare training & validation inputs & outputs
input_deep=np.zeros((trainn,3*K))
input_val=np.zeros((valsize,3*K))
output_deep=np.zeros((trainn,2*K))
output_val=np.zeros((valsize,2*K))

# generate the training data
for trialdeep in np.arange(trainn):
    
    (G,channelGainOverNoise,betaNLOS,GMean,power_coef)=channel_generator(M,K,1,pmax)
    # \eta_k in (15) in the paper
    eta=(power_coef/chi[0,0]).reshape(K,)
    # \tilde{\eta}_k in (32) assuming pilot matrix is DFT matrix
    eta_pilot=(power_coef/np.square(np.abs(np.sum(tildeb[0,:])))).reshape(K,)
    # convert the channel gain over noise to the linear scale
    normalizedGain=np.power(10,channelGainOverNoise/10)
    
    backOff2=backOff*np.sum(normalizedGain*power_coef)
    tildea=polyCoef/np.array([[1, backOff2, np.square(backOff2), np.power(backOff2,3)]])
    
    # Give the scaled channel coefficients in (15) as input to the following function
    effChannel=effChannelseventh(M,K,tildea,np.matmul(G[:,:,0],np.diag(np.sqrt(eta))),chi,chi2,chi3,chi4)
    # transmit the uplink signals and add Gaussian noise at the BS antennas
    U=np.matmul(pilott2,np.matmul(G[:,:,0],np.diag(np.sqrt(eta_pilot))).T)
    Z=tildea[:,0]*U+tildea[:,1]*np.square(np.abs(U))*U \
    +tildea[:,2]*np.power(np.abs(U),4)*U+tildea[:,3]*np.power(np.abs(U),6)*U
    Y=Z+np.sqrt(0.5)*(np.random.randn(K,1)+1j*np.random.randn(K,1))
    # Inputs and outputs for the neural network
    inputt=np.matmul(np.conj(pilot.T),Y)
    input_deep[trialdeep:trialdeep+1,0:K]=np.real(inputt).T
    input_deep[trialdeep:trialdeep+1,K:2*K]=np.imag(inputt).T
    input_deep[trialdeep:trialdeep+1,2*K:3*K]=np.sqrt(normalizedGain.T)
    output_deep[trialdeep:trialdeep+1,0:K]=np.real(effChannel)
    output_deep[trialdeep:trialdeep+1,K:2*K]=np.imag(effChannel)
  
       
# generate the validation data
for trialval in np.arange(valsize):
    (G,channelGainOverNoise,betaNLOS,GMean,power_coef)=channel_generator(M,K,1,pmax)
    eta=(power_coef/chi[0,0]).reshape(K,)
    eta_pilot=(power_coef/np.square(np.abs(np.sum(tildeb[0,:])))).reshape(K,)
   
    normalizedGain=np.power(10,channelGainOverNoise/10)
    backOff2=backOff*np.sum(normalizedGain*power_coef)
    tildea=polyCoef/np.array([[1, backOff2, np.square(backOff2), np.power(backOff2,3)]])
    
    
    effChannel=effChannelseventh(M,K,tildea,np.matmul(G[:,:,0],np.diag(np.sqrt(eta))),chi,chi2,chi3,chi4)
    

    U=np.matmul(pilott2,np.matmul(G[:,:,0],np.diag(np.sqrt(eta_pilot))).T)
    Z=tildea[:,0]*U+tildea[:,1]*np.square(np.abs(U))*U \
    +tildea[:,2]*np.power(np.abs(U),4)*U+tildea[:,3]*np.power(np.abs(U),6)*U
    Y=Z+np.sqrt(0.5)*(np.random.randn(K,1)+1j*np.random.randn(K,1))
    inputt=np.matmul(np.conj(pilot.T),Y)
    input_val[trialval:trialval+1,0:K]=np.real(inputt).T
    input_val[trialval:trialval+1,K:2*K]=np.imag(inputt).T
    input_val[trialval:trialval+1,2*K:3*K]=np.sqrt(normalizedGain.T)
    output_val[trialval:trialval+1,0:K]=np.real(effChannel)
    output_val[trialval:trialval+1,K:2*K]=np.imag(effChannel)

# eliminate the outliers for improved learning for training data length 3e6
aaaa=np.zeros((2*K,1))
for kk in np.arange(K):
    aaaa[kk,0]=(np.sort(np.abs(output_deep[:,kk]).T).T)[2990000]
    aaaa[kk+K,0]=(np.sort(np.abs(output_deep[:,kk+K]).T).T)[2990000]



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
    if doorr>0.5:
        output_val0[iii2,:]=output_val[tt,:]
        input_val0[iii2,:]=input_val[tt,:]
        iii2=iii2+1
                
input_deep00=input_deep0[0:iii1,:]
output_deep00=output_deep0[0:iii1,:]
input_val00= input_val0[0:iii2,:]
output_val00= output_val0[0:iii2,:]

# apply MinMaxScaler and StandardScaler as in the paper
scaler= StandardScaler()
scaler2 = MinMaxScaler(feature_range=(0.1,0.9))
input_deep1=np.zeros((input_deep00.shape))
input_val1=np.zeros((input_val00.shape))
input_deep1[:,0:2*K]=scaler.fit_transform(input_deep00[:,0:2*K])
input_deep1[:,2*K:3*K]=scaler2.fit_transform(input_deep00[:,2*K:3*K])

input_val1[:,0:2*K]=scaler.transform(input_val00[:,0:2*K])
input_val1[:,2*K:3*K]=scaler2.transform(input_val00[:,2*K:3*K])


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
                      output_deep00, 
                      epochs=50, 
                      callbacks=callbacks, 
                      batch_size=1000, 
                      validation_data=(input_val1, output_val00)) 
# testing results
# For deep learning, elements of "deepp" are for effective channel estimation.
# The elements of "lmmse" are for distortion-aware LMMSE-based channel estimation.
# The elements of "lmmse2" are for distortion-unaware LMMSE-based channel estimation.

deepp=np.zeros((test1,K))
lmmse=np.zeros((test1,K))
lmmse2=np.zeros((test1,K))
# go through each setup
for trialtest1 in np.arange(test1):
    # prepare for different realizations
    input_test=np.zeros((test2,3*K))
    output_test=np.zeros((test2,K))+1j*np.zeros((test2,K))
    # generate 2*test2 realizations (the first test2 channels are for Monte Carlo based LMMSE estimations)
    (G,channelGainOverNoise,betaNLOS,GMean,power_coef)=channel_generator(M,K,2*test2,pmax)
    eta=(power_coef/chi[0,0]).reshape(K,)
    eta_pilot=(power_coef/np.square(np.abs(np.sum(tildeb[0,:])))).reshape(K,)
    
    normalizedGain=np.power(10,channelGainOverNoise/10)
    backOff2=backOff*np.sum(normalizedGain*power_coef)
    tildea=polyCoef/np.array([[1, backOff2, np.square(backOff2), np.power(backOff2,3)]])
    
    # obtain statistics for Monte Carlo based LMMSE estimations

    obser=np.zeros((K,test2))+1j*np.zeros((K,test2))
    estim=np.zeros((K,test2))+1j*np.zeros((K,test2))
    for trialtest2 in np.arange(test2):
    

        U=np.matmul(pilott2,np.matmul(G[:,:,trialtest2],np.diag(np.sqrt(eta_pilot))).T)
        Z=tildea[:,0]*U+tildea[:,1]*np.square(np.abs(U))*U \
        +tildea[:,2]*np.power(np.abs(U),4)*U+tildea[:,3]*np.power(np.abs(U),6)*U
        Y=Z+np.sqrt(0.5)*(np.random.randn(K,1)+1j*np.random.randn(K,1))
        obser[:,trialtest2:trialtest2+1]=Y
        effChannel=effChannelseventh(M,K,tildea,np.matmul(G[:,:,trialtest2],np.diag(np.sqrt(eta))),chi,chi2,chi3,chi4)
        estim[:,trialtest2:trialtest2+1]=effChannel.T
    
    # Obtain related matrices for Monte Carlo-based LMMSE channel estimation (distortion-aware)
    (yBar1, Cbar1, CCy1, Cyy1)=lmmseMonteCarlo(obser,estim,1)
    invCyy1=inv(Cyy1)

    # prepare array for Monte Carlo-based distortion-aware LMMSE channel estimation
    lmmse_channel=np.zeros((test2,K))+1j*np.zeros((test2,K))
    # prepare array for distortion-unaware LMMSE channel estimation
    lmmse_unaware=np.zeros((test2,K))+1j*np.zeros((test2,K))
    
    # testing data is generated
    for trialtest2 in np.arange(test2):
    

        U=np.matmul(pilott2,np.matmul(G[:,:,trialtest2+test2],np.diag(np.sqrt(eta_pilot))).T)
        Z=tildea[:,0]*U+tildea[:,1]*np.square(np.abs(U))*U \
        +tildea[:,2]*np.power(np.abs(U),4)*U+tildea[:,3]*np.power(np.abs(U),6)*U
        Y=Z+np.sqrt(0.5)*(np.random.randn(K,1)+1j*np.random.randn(K,1))
        inputt=np.matmul(np.conj(pilot.T),Y)
        input_test[trialtest2:trialtest2+1,0:K]=np.real(inputt).T
        input_test[trialtest2:trialtest2+1,K:2*K]=np.imag(inputt).T
        input_test[trialtest2:trialtest2+1,2*K:3*K]=np.sqrt(normalizedGain.T)
        output_test[trialtest2:trialtest2+1,:]=effChannelseventh(M,K,tildea,np.matmul(G[:,:,trialtest2+test2],np.diag(np.sqrt(eta))),chi,chi2,chi3,chi4)
        lmmse_channel[trialtest2:trialtest2+1,:]=(Cbar1+np.matmul(CCy1,np.matmul(invCyy1,Y-yBar1))).T
    
        for kk in np.arange(K):
            
            lmmse_unaware[trialtest2:trialtest2+1,kk]=np.sqrt(power_coef[kk,0])*GMean[0,kk]+(power_coef[kk,0]*betaNLOS[kk,0])/(K*power_coef[kk,0]*betaNLOS[kk,0]+1) \
            *(inputt[kk,0]-np.sqrt(power_coef[kk,0])*K*GMean[0,kk])

    # obtain deep learning-based estimates
    input_test1=np.zeros((input_test.shape))
    input_test1[:,0:2*K]=scaler.transform(input_test[:,0:2*K])
    input_test1[:,2*K:]=scaler2.transform(input_test[:,2*K:])
    y_est2=estimator1.predict(input_test1)
    deep_channel=y_est2[:,0:K]+1j*y_est2[:,K:2*K]
    
    # calculate normalized mean square errors

    for kk in np.arange(K):
        
        lmmse[trialtest1,kk]=np.square(LA.norm(lmmse_channel[:,kk]-output_test[:,kk]))/np.square(LA.norm(output_test[:,kk]))
        lmmse2[trialtest1,kk]=np.square(LA.norm(lmmse_unaware[:,kk]-output_test[:,kk]))/np.square(LA.norm(output_test[:,kk]))
        deepp[trialtest1,kk]=np.square(LA.norm(deep_channel[:,kk]-output_test[:,kk]))/np.square(LA.norm(output_test[:,kk]))


lmmseEffChan=lmmse.reshape(test1*K,1)
deeppEffChan=deepp.reshape(test1*K,1)
lmmse2EffChan=lmmse2.reshape(test1*K,1)

# y-axis values for plotting CDF
    
yax=np.arange(1, K*test1+1) / (K*test1)


# Plot the CDF curves for effective channel estimation

plt.plot(10*np.log10(np.sort(lmmse2EffChan.T).T),yax,10*np.log10(np.sort(lmmseEffChan.T).T),yax,\
          10*np.log10(np.sort(deeppEffChan.T).T),yax)
plt.grid()
matplotlib.rcParams.update({'font.size': 50,})
plt.rc('text', usetex=True)
plt.ylabel(r'CDF',fontsize=50)
plt.xlabel(r'NMSE (dB)',fontsize=50)
plt.legend((r'DuA-LMMSE', r'DA-LMMSE', r'Deep Learning' ))
plt.xticks([-40, -35, -30, -25, -20, -15, -10 , -5, 0])
plt.tick_params(axis='both', which='major', labelsize=50)