
# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd
import os

dir_file =  os.listdir(os.getcwd())

print(dir_file)

stockdata = []
for i in dir_file:
    if i.find('.csv') >=0: 
        stockdata.append(pd.read_csv(i))
stockdata = pd.concat(stockdata)



import matplotlib
import matplotlib.pyplot as plt
import datetime

stockdata.head()

hsbc = stockdata[stockdata['Stock_Code'] == "0005.HK" ] 

hsbc["DateTime"] = hsbc["Date"].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y %H:%M:%S"))  

hsbc.head()

plt.plot( hsbc.DateTime, hsbc.Open)

plt.show()

n = hsbc["Open"].shape[0]
ffthsbc = np.fft.fft(hsbc["Open"], n )

def extractData( startpoint, endpoint, data):
    return data[startpoint: endpoint]


def shiftData( startpoint, data): 
    return data[startpoint+1:]


def convertinput(windowsize = 10): 
    return 0, windowsize-1

def checkend(windowsize, data): 
    if data.shape[0] < windowsize:
        return True
    else:
        return False
    


def convertToPolar(npData): 
    return np.absolute(ffthsbc), np.angle(ffthsbc)

def sortedPolar( data): 
    zipped = sorted(zip(data[0], data[1]) , reverse=True) 
    abslist = []
    anglelist = []
    for abscomp , anglecomp in zipped: 
        abslist.append(abscomp)
        anglelist.append(anglecomp)
    return abslist, anglelist

def convertTimeSeriesToFFT(data): 
    n = data.shape[0]
    return sortedPolar( convertToPolar( np.fft.fft( data, n )))

def FFTExtractedData( data, windowsize = 10): 
    startpoint , endpoint = convertinput(windowsize)
    extractdata = extractData( startpoint, endpoint, data)
    return convertTimeSeriesToFFT(extractdata) , shiftData(endpoint, data)

def rollingFFT(data, windowsize = 10): 
    if not checkend(windowsize, data): 
        fftdata, shiftdata = FFTExtractedData(data , windowsize)
        return rollingFFT(shiftdata, windowsize).insert(fftdata,0)
    else:
        return []
    



fullFFTData = rollingFFT(hsbc.Close)

print(b[0][0])
print(b[1][0])

