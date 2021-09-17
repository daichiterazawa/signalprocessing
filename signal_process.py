import numpy as np
import matplotlib.pyplot as plt


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
    bits = np.copy(prbs)
    
    #グレイ符号に伴う（下位ビット）の反転を補正
    bits[0::4] ^= bits[2::4]
    bits[1::4] ^= bits[3::4]
    #変調
    bits = bits *2 - 1
    QAM_16 = 1*bits[0::4] + 1j*bits[1::4] + 2*bits[2::4] + 2j*bits[3::4]
    
    return QAM_16

#64QAM
def modQAM_64(prbs):
    bits = np.copy(prbs)
    
    #グレイ符号化に伴う（中位ビット）の反転補正
    bits[2::6] = bits[2::6]^bits[4::6]
    bits[3::6] = bits[3::6]^bits[5::6]
    
    #グレイ符号に伴う（下位ビット）の反転を補正
    bits[0::6] = bits[0::6]^bits[2::6]
    bits[1::6] = bits[1::6]^bits[3::6]
    
    #変調
    bits = bits*2 -1
    QAM_64 = 1*bits[0::6] + 1j*bits[1::6] + 2*bits[2::6] + 2j*bits[3::6] + 4*bits[4::6] + 4j*bits[5::6]
    
    return QAM_64

###################################################

###################　雑音の付与　#####################
    
def noise(SNRdB,N,QAM_order):
    SNR = 1/QAM_order *10**(0.1 * SNRdB)
    noise =  np.random.normal(0, np.sqrt(1/(2*SNR)), N) + 1j*np.random.normal(0, np.sqrt(1/(2*SNR)), N)
    return noise
    
###################################################    

################### 信号の復調 #####################
#QPSA
def decQPSK(signal_AWGN):
    #デコードしたビットを格納する配列
    decQPSK = np.zeros(2 * signal_AWGN.size, dtype = int)
    
    #実数成分の閾値処理
    num_real = signal_AWGN.real
    num_real[np.where(num_real > 0)] = 1
    num_real[np.where(num_real < 0)] = 0
    
    #虚数成分の閾値処理
    num_imag = signal_AWGN.imag
    num_imag[np.where(num_imag > 0)] = 1
    num_imag[np.where(num_imag < 0)] = 0
    
    decQPSK[0::2] = num_real
    decQPSK[1::2] = num_imag
    
    
    return decQPSK
#16QAM
def decQAM_16(signal_AWGN):
    #デコードしたビットを格納する配列
    decQAM_16 = np.zeros(4 * signal_AWGN.size, dtype = int)
    
    #実数成分の閾値処理
    num_real = signal_AWGN.real
    num_real = (num_real // 2) *2 +1
    num_real[np.where(num_real > 3)] = 3
    num_real[np.where(num_real < -3)] = -3
    #虚数成分の閾値処理
    num_imag = signal_AWGN.imag
    num_imag = (num_imag // 2) *2 +1
    num_imag[np.where(num_imag > 3)] = 3
    num_imag[np.where(num_imag < -3)] = -3
    
    #シンボル->ビット
    #第1象限のみに縮小
    num_real = (num_real+3)/2
    num_imag = (num_imag+3)/2
    #10進数を2進数に
    decQAM_16[3::4],num_imag = divmod(num_imag,2)
    decQAM_16[2::4],num_real = divmod(num_real,2)
    decQAM_16[1::4] = num_imag
    decQAM_16[0::4] = num_real
    
    #グレイ符号のため
    decQAM_16[0::4] ^= decQAM_16[2::4]
    decQAM_16[1::4] ^= decQAM_16[3::4]
    
    return decQAM_16
#64QAM
def decQAM_64(signal_AWGN):
    #デコードしたビットを格納する配列
    decQAM_64 = np.zeros(6 * signal_AWGN.size, dtype = int)
    
    #実数成分の閾値処理
    num_real = signal_AWGN.real
    num_real = (num_real // 2) *2 +1
    num_real[np.where(num_real > 7)] = 7
    num_real[np.where(num_real < -7)] = -7
    #虚数成分の閾値処理
    num_imag = signal_AWGN.imag
    num_imag = (num_imag // 2) *2 +1
    num_imag[np.where(num_imag > 7)] = 7
    num_imag[np.where(num_imag < -7)] = -7
    
    #シンボル->ビット
    #第1象限のみに縮小
    num_real = (num_real+7)/2
    num_imag = (num_imag+7)/2
    #10進数を2進数に
    decQAM_64[5::6],num_imag = divmod(num_imag,4)
    decQAM_64[4::6],num_real = divmod(num_real,4)
    decQAM_64[3::6],num_imag = divmod(num_imag,2)
    decQAM_64[2::6],num_real = divmod(num_real,2)
    decQAM_64[1::6] = num_imag
    decQAM_64[0::6] = num_real
    
    #グレイ符号のため（下位ビット）=（中位ビット）^（下位ビット)
    decQAM_64[0::6] ^= decQAM_64[2::6]
    decQAM_64[1::6] ^= decQAM_64[3::6]
    #グレイ符号のため（中位ビット）=（上位ビット）^（中位ビット)
    decQAM_64[2::6] ^= decQAM_64[4::6]
    decQAM_64[3::6] ^= decQAM_64[5::6]
           
    return decQAM_64
        
        
if __name__ == '__main__':  
    
    
    #送信ビット
    prbs_QPSK = prbs(2 * 20)
    prbs_QAM16 = prbs(4 * 20)
    prbs_QAM64 = prbs(6 * 20)
    
    #変調    
    mod_QPSK = modQPSK(prbs_QPSK)
    mod_QAM16 = modQAM_16(prbs_QAM16)
    mod_QAM64 = modQAM_64(prbs_QAM64) 
    
    #リスト作成
    SNR_list = np.arange(0,30,1)
    BER_QPSK_list = []
    BER_16QAM_list = []
    BER_64QAM_list = []
    
    for SNR in SNR_list:
        #雑音付与
        noise_QPSK = mod_QPSK + noise(SNR, mod_QPSK.size, 2*10)
        noise_QAM16 = mod_QAM16 + noise(SNR, mod_QAM16.size, 4*10)
        noise_QAM64 = mod_QAM64 + noise(SNR, mod_QAM64.size,6*10)
        
        #デコード
        dec_QPSK = decQPSK(noise_QPSK)
        dec_QAM16 = decQAM_16(noise_QAM16)
        dec_QAM64 = decQAM_64(noise_QAM64)
        
        BER_QPSK = np.sum(prbs_QPSK^dec_QPSK)/prbs_QPSK.size
        BER_QAM16 = np.sum(prbs_QAM16^dec_QAM16)/prbs_QAM16.size
        BER_QAM64 = np.sum(prbs_QAM64^dec_QAM64)/prbs_QAM64.size
        
        BER_QPSK_list.append(BER_QPSK)
        BER_16QAM_list.append(BER_QAM16)
        BER_64QAM_list.append(BER_QAM64)
        
    fig = plt.figure(figsize = (10,10))
    plt.plot(SNR_list, BER_QPSK_list,marker='.')
    plt.plot(SNR_list, BER_16QAM_list,marker='.')
    plt.plot(SNR_list, BER_64QAM_list,marker='.')
    plt.yscale('log')
    fig.savefig("img.png")
        
        



