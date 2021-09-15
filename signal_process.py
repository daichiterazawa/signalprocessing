import numpy as np
import matplotlib.pyplot as plt

L = 6
################　送信ビットを生成　##################

#prbs(２で割った場合の商と剰余を利用する)
#レジスターを10進数表記
def prbs(L):
    reg = 1495
    prbs_list = []
    for _ in range((2**15-1) * L):
        reg, amari = divmod(reg, 2)    
        reg += ((reg % 2) ^ amari) * 2**14
        prbs_list.append(amari)
        
    prbs = np.array(prbs_list)
    return prbs

##################################################

################# 信号を変調 ######################

#QPSK
def modQPSK(prbs):
    prbs = 2 * prbs - 1
    qpsk = prbs[0::2] + 1j * prbs[1::2]
    
    return qpsk

#16QAM
def modQAM_16(prbs):
    #グレイ符号に伴う（下位ビット）の反転を補正
    prbs[0::4] ^= prbs[2::4]
    prbs[1::4] ^= prbs[3::4]
    #変調
    prbs = prbs *2 - 1
    QAM_16 = 1*prbs[0::4] + 1j*prbs[1::4] + 2*prbs[2::4] + 2j*prbs[3::4]
    
    return QAM_16

#64QAM
def modQAM_64(prbs):
    #グレイ符号化に伴う（中位ビット）の反転補正
    prbs[2::6] ^= prbs[4::6]
    prbs[3::6] ^= prbs[5::6]
    #グレイ符号に伴う（下位ビット）の反転を補正
    prbs[0::6] ^= prbs[2::6]
    prbs[1::6] ^= prbs[3::6]
    #変調
    prbs = prbs*2 -1
    QAM_64 = 1*prbs[0::6] + 1j*prbs[1::6] + 2*prbs[2::6] + 2j*prbs[3::6] + 4*prbs[4::6] + 4j*prbs[5::6]
    
    return QAM_64

###################################################

###################　雑音の付与　#####################
    
def noise(SNR,N):
    noise =  np.random.normal(0, np.sqrt(1/(2*SNR)), N) + 1j*np.random.normal(0, np.sqrt(1/(2*SNR)), N)
    return noise
    
###################################################    

################### 信号の復調 #####################

#64QAM
def decQAM_64(signal_AWGN):
    list_Q = []
    list_I = []
    num_real = signal_AWGN.real
    num_imag = signal_AWGN.imag
    
    #実数成分の閾値処理
    num_real = (num_real // 2) *2 +1
    num_real[np.where(num_real > 7)] = 7
    num_real[np.where(num_real < -7)] = -7
    
    
    
        
    return format(num_real,'b')
        
        
    
    

a = modQAM_64(prbs(L))
b = noise(10,a.size) + a
c = decQAM_64(b)


fig = plt.figure(figsize = (10,10))
plt.scatter(b.real,b.imag)
plt.show()

