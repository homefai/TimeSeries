import pandas as pd
import numpy as np
import StockDataConstant as stc
import sys

class StockData():
    def __init__(self, df):
        self.data = df
    
    def colDiff(self, columns_name=[], new_col_name=None ,percent=False ):
        if len(columns_name) == 2 and new_col_name != None: 
            try:
                self.data[new_col_name] = self.data.apply(lambda x: x[columns_name[1]] - x[columns_name[0]] ,axis = 1 )
            except:
                print('Input columns Error')
            if percent: 
                self.data[new_col_name] = self.data.apply(lambda x: x[new_col_name]/x[columns_name[0]], axis =1)
        else:
            print('Input collDiff Error')

    def colSum(self, columns_name=[], new_col_name=None ,percent=False ):
        if len(columns_name) == 2 and new_col_name != None: 
            try:
                self.data[new_col_name] = self.data.apply(lambda x: x[columns_name[1]] + x[columns_name[0]] ,axis = 1 )
            except:
                print('Input columns Error')
            if percent: 
                self.data[new_col_name] = self.data.apply(lambda x: x[new_col_name]/x[columns_name[0]], axis =1)
        else:
            print('Input collSum Error')

    def colMul(self, columns_name=[], new_col_name=None, norm=False):
        if len(columns_name) == 2 and new_col_name != None: 
            try:
                self.data[new_col_name] = self.data.apply(lambda x: x[columns_name[1]] * x[columns_name[0]] ,axis = 1 )
            except:
                print('Input columns Error')
            if norm: 
                import math
                self.data[new_col_name] = self.data[new_col_name].apply(lambda x: math.sqrt(x))
        else:
            print('Input collMal Error')
    
    def normalizeSeries(self, df_segment, normfunc=None):
        if normfunc == None:
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            return scaler.fit_transform(df_segment)
        else:
            return normfunc(df_segment)

    def movingFunc(self , window=[10], fields= ["Open"], func='mean'):
        
        funcdict = {
                    'mean' : np.mean, 
                    'sum' : np.sum,  
                }

        if not isinstance(window, list): 
            print("Input is not a list")
            window = [window] 
        if not isinstance(fields, list):
            print("Input us not a list")
            fields = [fields]
        fields = [ x for x in fields if x in self.data.columns]
        for field in fields:
            for size in window: 
                self.data["%s_%s_%d" % (field, func ,size)] = self.data[field].rolling(window=size).apply(func=funcdict[func])
    
    def plotSingleSeries(self, pltfieldname): 
        import matplotlib
        import matplotlib.pyplot as plt
        plt.plot(self.data[pltfieldname] )
        plt.show()
        return True



class SeriesFilter():
    def __init__(self): 
        pass
    
    def filterfunc(self):
        raise NotImplementedError

    def applyfilter(self):
        raise NotImplementedError

class WaveletFilter(SeriesFilter):
    def __init__(self,half_period=2):
        self.filter = self.filterfunc(half_period)
        self.half_period = half_period

    def filterfunc(self, half_period=2):
        return np.array([1]*half_period + [-1]*half_period)

    def applyfilter(self, input_df):
        return input_df.rolling(self.half_period*2).apply(lambda x: np.matmul(self.filter, x))

    def changeHalfPeriod(self, half_period):
        try:
            tmp = self.filterfunc(half_period)
            self.filter = tmp
            self.half_period = half_period
        except:
            print('Initialize the filter matrix error: WaveletFilter')


class FFTFilter(SeriesFilter):
    def __init__(self, half_period=2): 
        self.filter = self.filterfunc()
        self.half_period = half_period
    
    def filterfunc(self):
        import numpy.fft as fft
        return fft

    def applyfilter(self, input_df):
        fftresult = input_df.rolling(self.half_period).apply(lambda x: self.filter.fft(x))
        length_of_data = self.half_period*2
        result_of_fft = {}
        for j in range(length_of_data):
            result_of_fft["%d_%d" % (length_of_data, j) ] = []
            for i in range(len(fftresult)): 
                result_of_fft["%d_%d" % (length_of_data, j)].append(np.abs(i[j]))
        return result_of_fft
    
def getFullPath(filename, basepath = stc.STOCK_DATA_BASEPATH, relativepath= stc.STOCK_DATA_RELATIVEPATH):
    basepath = basepath
    relativepath = relativepath
    data_files_full_path = "%s%s/%s"%(basepath, relativepath, filename)   
    return data_files_full_path

def importData(filename, basepath = stc.STOCK_DATA_BASEPATH, relativepath= stc.STOCK_DATA_RELATIVEPATH): 
    try:
        print getFullPath(filename=filename, basepath=basepath, relativepath=relativepath)
        with open( getFullPath(filename=filename, basepath=basepath, relativepath=relativepath), 'rb' ) as f:
            data_files = pd.read_csv(f)
    except:
        print('Import data Error:ImporData')
        return None
    return data_files

def selectStock(stock_code, df):
    import re
    stock_code_pattern = re.compile( '[0-9]{4}\.HK' )
    if isinstance(stock_code, str): 
        r = stock_code_pattern.findall(stock_code)
        print r
    else:
        print("Input Code pattern invalid:StockData.selectStock")
        return None
    try: 
        data_files = df[df["Stock_Code"] == stock_code]
        return data_files
    except:
        print("Import Data Error:getAllStock")
        return None


def runFullClass(filename, code):
    raw_data = selectStock(code , importData(filename, basepath = stc.STOCK_DATA_BASEPATH, relativepath= stc.STOCK_DATA_RELATIVEPATH))
    a= StockData(raw_data)
    a.colDiff(columns_name=['Open' , 'Close'], new_col_name='diff_close_open', percent=True)
    a.colMul(columns_name=['Open' , 'Close'], new_col_name='openclose_mul', norm=True)
    return a


def main():
    if len(sys.argv) < 2:
        raw_data = importData(filename='stockCollectionUpdate-checkoutpage_6_1_2017.csv')
        data_files =  StockData(selectStock('0005.HK', raw_data))
    else:
        print(importData(filename=sys.argv[1]))

if __name__ == '__main__':
    main()