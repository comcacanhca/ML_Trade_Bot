import builtins
import datetime

import pandas as pd

from data_handler.download_data_mt5 import *
from objects.ClassObjects import MAU_NEN, RAU_DUOI, RAU_TREN, THAN_NEN, UPPER, LOWER, DELMAS
from plot_chart.draw_cha import *
def round_value(data, col):
    def _round(row):
        if abs(row[col]) > 100:
            return round(row[col], -2)
        elif abs(row[col]) > 10:
            return round(row[col], -1)
        else:
            return round(row[col], 0)
    data[col] = data.apply(_round, axis=1)

    return data

data = pd.read_csv(r'E:\filter_tradeBot\main\thong_Ke\phanvung_move.csv')
print(data.columns)
CHECK_COLS = ['mau_nen', 'rau_tren', 'rau_duoi', 'max_vol',
       'DelMAs200', 'DelMAs1x200', 'DelMAsUpper',
       'DelMAsLower', 'DelMAs1xUpper200', 'DelMAs1xLower200', 'ULDIS', 'SUL16',
       'S_DelMAs_16', 'min', 'max']
data = data[CHECK_COLS]
data.dropna()


for col in CHECK_COLS:
    data = round_value(data, col)

def filter(df, col, value, type=1):
    print(f'Filter for {col} and {value}')
    if value!=None and isinstance(value, list)==False:
        if type==1:
            filter_data = df[df[col] >= value]
        elif type==-1:
            filter_data = df[df[col] <= value]
        else:
            raise 'Not support type'
    elif isinstance(value, list):
        filter_data = df[value[0] <= df[col]]
        filter_data = filter_data[filter_data[col] <= value[1]]
    else:
        print('No value to filter')
        filter_data= df
    print(f'AFTER FILTER: {len(filter_data)}')
    return filter_data

def check(BINS = [0, 3, 45, 55, 97, 100]):
    MEANS = data.describe([i*0.01 for i in BINS])
    print(MEANS)
    boundaries = {
        'mau_nen': MEANS[MAU_NEN],
        RAU_TREN: MEANS[RAU_TREN],
        RAU_DUOI: MEANS[RAU_DUOI],
        'DelMAs200': [[MEANS['DelMAs200']['0%'],MEANS['DelMAs200'][f'{BINS[1]}%']],[MEANS['DelMAs200'][f'{BINS[2]}%'],MEANS['DelMAs200'][f'{BINS[3]}%']],[MEANS['DelMAs200'][f'{BINS[4]}%'],MEANS['DelMAs200']['100%']]],
        'DelMAsUpper': [[MEANS['DelMAsUpper'][i],MEANS['DelMAsUpper'][i+1]] for i in [0,2,4]],
        'DelMAsLower': [[MEANS['DelMAsLower'][i],MEANS['DelMAsLower'][i+1]] for i in [0,2,4]],

        'DelMAs1x200': [[MEANS['DelMAs1x200']['0%'],MEANS['DelMAs1x200'][f'{BINS[1]}%']],[MEANS['DelMAs1x200'][f'{BINS[2]}%'],MEANS['DelMAs1x200'][f'{BINS[3]}%']],[MEANS['DelMAs1x200'][f'{BINS[4]}%'],MEANS['DelMAs1x200']['100%']]],
        'DelMAs1xUpper200':[0],
        'DelMAs1xLower200':[0],
        'ULDIS': [[MEANS['ULDIS']['0%'],MEANS['ULDIS'][f'{BINS[1]}%']],[MEANS['ULDIS'][f'{BINS[2]}%'],MEANS['ULDIS'][f'{BINS[3]}%']],[MEANS['ULDIS'][f'{BINS[4]}%'],MEANS['ULDIS']['100%']]],
        'SUL16': [[MEANS['SUL16']['0%'],MEANS['SUL16'][f'{BINS[1]}%']],[MEANS['SUL16'][f'{BINS[2]}%'],MEANS['SUL16'][f'{BINS[3]}%']],[MEANS['SUL16'][f'{BINS[4]}%'],MEANS['SUL16']['100%']]],
    }

    for va1 in [None]+boundaries['DelMAs1x200']:
        filter_data = filter(data.copy(), 'DelMAs1x200', value=va1, type=1)
        for va2 in [None] + boundaries['ULDIS']:
            filter_data1 = filter(filter_data, 'ULDIS', value=va2, type=1)
            for va3 in [None] + boundaries['SUL16']:
                filter_data2 = filter(filter_data1, 'SUL16', value=va3, type=1)
                for va4 in [None] + boundaries['DelMAs1xLower200']:
                    filter_data3 = filter(filter_data2, 'DelMAs1xLower200', value=va4, type=-1)
                    for va5 in [None] + boundaries['DelMAs1xUpper200']:
                        filter_data4 = filter(filter_data3, 'DelMAs1xUpper200', value=va5, type=1)

                        if 50000> len(filter_data)>0 and (max(filter_data['min'])<=-200 or min(filter_data['max'])>=200):
                            print('Thoa man dieu kien tai')
                            filter_data4.to_csv(fr'E:\filter_tradeBot\data\data_analyzing/{va1}_{va2}_{va3}_{va4}_{va5}.csv', index=False)

for x1 in [i for i in range(2, 10)]:
    for x2 in [45, 46, 47, 48]:
        for x3 in [52,53,54,55]:
            for x4 in [i for i in range(95, 100)]:
                check([0, x1, x2, x3, x4, 100])