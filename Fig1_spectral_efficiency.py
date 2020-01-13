# This Python script generates Figure 1 in the paper:
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
from numpy.linalg import inv
import matplotlib
import matplotlib.pyplot as plt
from all_functions import moments, channel_generator, effChannelthird, distortionCorr, insRate

# number of BS antennas
M=100   
# number of UEs
K=10
# Backoff parameter in dB
backOff=np.power(10,0.7)
# Third-order polynomial coefficients obtained by curve fitting to [25]
polyCoef=np.array([[ 0.9798-0.0075*1j, -0.2419+0.0374*1j]])
# \tilde{b}_0 and \tilde{b}_1 in (7) in the paper
tildeb=polyCoef/np.array([[1, backOff]])
# moments in (8) and (9) in the paper
(zeta,chi)=moments(1,tildeb)
# maximum transmission power for UE antennas in Watts
pmax=0.2

# number of channel realizations for each setup
MonteCarloTrial=100
# number of Setups
trialNumber=250
# Prepare results for CDF curve
cdfMR=np.zeros((K,trialNumber))
cdfRZF=np.zeros((K,trialNumber))
cdfDA_MMSE=np.zeros((K,trialNumber))
cdfEW_DA_MMSE=np.zeros((K,trialNumber))
# Go through each setup
for trial in np.arange(trialNumber):
    # Prepare SE estimations for averaging
    rateMR=np.zeros((K,1))
    rateRZF=np.zeros((K,1))
    rateDA_MMSE=np.zeros((K,1))
    rateEW_DA_MMSE=np.zeros((K,1))
    
    (G,channelGainOverNoise,betaNLOS,GMean,power_coef)=channel_generator(M,K,MonteCarloTrial,pmax)
    # \eta_k in (15) in the paper
    eta=(power_coef/chi[0,0]).reshape(K,)
    # convert the channel gain over noise to the linear scale
    normalizedGain=np.power(10,channelGainOverNoise/10)
    # \tilde{a}_{0} and \tilde{a}_1 in (4) in the paper (same for each BS antenna)
    tildea=polyCoef/np.array([[1, backOff]])
    tildea[:,1]=tildea[:,1]/np.sum(normalizedGain*power_coef)
    
    
    for trial2 in np.arange(MonteCarloTrial):
        # Give the scaled channel coefficients in (15) as input to the following functions
        effChannel=effChannelthird(M,K,tildea,tildeb,np.matmul(G[:,:,trial2],np.diag(np.sqrt(eta))),zeta)
        distCorr=distortionCorr(tildea,np.matmul(G[:,:,trial2],np.diag(np.sqrt(eta))),chi,M,effChannel)
        # calculate the linear receiver vectors
        vMR=effChannel.copy()
        vRZF=np.matmul(effChannel,inv(np.matmul(np.conj(effChannel.T),effChannel)+np.eye(K)))
        vDA=np.zeros((M,K))+1j*np.zeros((M,K))
        vEW=np.zeros((M,K))+1j*np.zeros((M,K))
        Cyy=distCorr+np.matmul(effChannel,np.conj(effChannel.T))
        Cyydiag=np.diag(np.diag(distCorr))+np.matmul(effChannel,np.conj(effChannel.T))

        for k in np.arange(K):
            vDA[:,k:k+1]=np.matmul(inv(Cyy-np.matmul(effChannel[:,k:k+1],np.conj(effChannel[:,k:k+1].T))),effChannel[:,k:k+1])
            vEW[:,k:k+1]=np.matmul(inv(Cyydiag-np.matmul(effChannel[:,k:k+1],np.conj(effChannel[:,k:k+1].T))),effChannel[:,k:k+1])
        
        
        # add the instantaneous rates for averaging
        rateMR=rateMR+insRate(vMR,effChannel,distCorr,K)
        rateRZF=rateRZF+insRate(vRZF,effChannel,distCorr,K)
        rateDA_MMSE=rateDA_MMSE+insRate(vDA,effChannel,distCorr,K)
        rateEW_DA_MMSE=rateEW_DA_MMSE+insRate(vEW,effChannel,distCorr,K)
    
    # take averages
    cdfMR[:,trial:trial+1]=rateMR/MonteCarloTrial
    cdfRZF[:,trial:trial+1]=rateRZF/MonteCarloTrial
    cdfDA_MMSE[:,trial:trial+1]=rateDA_MMSE/MonteCarloTrial
    cdfEW_DA_MMSE[:,trial:trial+1]=rateEW_DA_MMSE/MonteCarloTrial


# y-axis values for plotting CDF
yax=np.arange(1, K*250+1) / (K*250)
A=cdfMR.reshape(K*250)
B=cdfRZF.reshape(K*250)
C=cdfEW_DA_MMSE.reshape(K*250)
D=cdfDA_MMSE.reshape(K*250)

plt.plot(np.sort(A),yax,np.sort(B),yax, np.sort(C), yax,np.sort(D),yax)
plt.grid()
matplotlib.rcParams.update({'font.size': 50,})
plt.rc('text', usetex=True)
plt.ylabel(r'CDF',fontsize=50)
plt.xlabel(r'SE (bits/s/Hz)',fontsize=50)
plt.tick_params(axis='both', which='major', labelsize=50)
plt.legend((r'DA-MRC', r'DA-RZF',r'EW-DA-MMSE', r'DA-MMSE' ))