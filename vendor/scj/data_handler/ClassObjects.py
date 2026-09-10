import statistics
from itertools import combinations
from . import *
import numpy as np
import pandas as pd
from log.MyLogger import MyLogger
from plot_chart.draw_cha import draw_lan_can, draw_MA

logger = MyLogger.get_logger()

from data_handler.download_data_mt5 import *


#Phan loai doi tuong nen
def is_weekend(day):
    if day.weekday() in [5, 6]:
        return True
    else:
        return False

class Candle:
    def __init__(self, data, tick='GBPUSD'):
        if isinstance(data, pd.DataFrame):
            logger.info(f'ClassObjects:DataFrame input > get old data_handler from csv')
            self.data = data
        elif isinstance(data, str):
            logger.info(f'ClassObjects:str input > get old data_handler from csv')
            self.data = pd.read_csv(data)
        else:
            logger.info(f'ClassObjects:no data_handler input > get old data_handler from csv')
            self.data = pd.read_csv(r'E:\filter_tradeBot\main\data\data_resource\Candles.csv')
        self.BBPer = None

        if tick in ['GBPUSD', 'EURUSD', 'AUDUSD', 'USDCHF', 'GBPAUD','USDJPY','NZDUSD', 'USDCAD']:
            self._x = 100000
        elif tick in ['XAUUSD', 'DXY']:
            self._x = 100
        else:
            raise f'Not support {tick}'

    def get_x(self):
        return self._x

    def _get_datetime_series(self):
        dates = pd.to_datetime(self.data[DATE], errors='coerce')
        if dates.isna().any():
            dates = pd.to_datetime(self.data[DATE], format='%Y-%m-%d %H:%M:%S%z', errors='coerce').fillna(dates)
        return dates

    """BASICCAL ATTS HANDLE"""
    def generate_basic_attribute(self):
        logger.info(f'ClassObjects:Start generate basic atts')
        data = self.data
        logger.info(len(data))
        open_series = data[OPEN]
        close_series = data[CLOSE]
        high_series = data[HIGH]
        low_series = data[LOW]
        volume_series = data[CLOSE]-data[OPEN]

        self.data.loc[:, MAU_NEN] = (volume_series * self._x)
        self.data.loc[:, THAN_NEN] = (volume_series.abs() * self._x)
        self.data.loc[:, RAU_TREN] = ((high_series - pd.concat([close_series, open_series], axis=1).max(axis=1)).abs() * self._x)
        self.data.loc[:, RAU_DUOI] = ((pd.concat([close_series, open_series], axis=1).min(axis=1) - low_series).abs() * self._x)
        self.data.loc[:, MAX_VOL] = ((high_series - low_series) * self._x)
        return self.data

    def determine_session(self, hour):
        if 0 <= hour < 9:
            return "Phiên Á"
        elif 8 <= hour < 17:
            return "Phiên Âu"
        elif 13 <= hour < 22:
            return "Phiên Mỹ"
        else:
            return "Ngoài phiên"

    def checkPhien(self):
        dates = self._get_datetime_series()
        hours = dates.dt.hour
        minutes = dates.dt.minute
        phien = pd.Series("Ngoài phiên", index=self.data.index)
        phien[((0 <= hours) & (hours < 9)) | ((hours == 9) & (minutes == 0))] = "Phiên Á"
        phien[((8 <= hours) & (hours < 17)) | ((hours == 17) & (minutes == 0))] = "Phiên Âu"
        phien[((13 <= hours) & (hours < 22)) | ((hours == 22) & (minutes == 0))] = "Phiên Mỹ"
        self.data.loc[:, 'phien'] = phien
        return self.data

    def checkH4(self):
        dates = self._get_datetime_series()
        hours = dates.dt.hour
        values = pd.Series(-10, index=self.data.index)
        values[(0 <= hours) & (hours < 3)] = 10
        values[(3 <= hours) & (hours < 6)] = 100
        values[(6 <= hours) & (hours < 9)] = 1000
        values[(9 <= hours) & (hours < 12)] = 10000
        values[(12 <= hours) & (hours < 15)] = -10000
        values[(15 <= hours) & (hours < 18)] = -1000
        values[(18 <= hours) & (hours < 21)] = -100
        self.data.loc[:, '4hours'] = values
        return self.data

    def checkH1(self):
        dates = self._get_datetime_series()
        chia = dates.dt.hour // 2
        values = pd.Series(index=self.data.index, dtype='object')
        mask_up = (3 <= chia) & (chia < 9)
        mask_down = chia >= 9
        values[mask_up] = [10 ** (val - 3) for val in chia[mask_up]]
        values[mask_down] = [-1 * 10 ** (val - 9) for val in chia[mask_down]]
        values[~(mask_up | mask_down)] = [-1 * 10 ** (val + 3) for val in chia[~(mask_up | mask_down)]]
        self.data.loc[:, '1hours'] = values
        return self.data

    def calc_candles_group(self, period):
        logger.info(f'ClassObjects:Start generate candles_group of{period} candles')
        data = self.data
        self.data.loc[:,THAN_NEN+f'{period}'] = [abs(data[CLOSE][candle] - data[OPEN][candle - period]) * self._x if candle>=period else 0 for candle in range(len(data.index))]
        self.data.loc[:,MAU_NEN+f'{period}'] = [(data[CLOSE][candle] - data[OPEN][candle - period]) * self._x if candle>=period else 0 for candle in range(len(data.index))]
        self.data.loc[:,RAU_TREN+f'{period}'] = [abs(data[HIGH][candle] - max(data[CLOSE][candle], data[OPEN][candle - period])) * self._x if candle>=period else 0 for candle in range(len(data.index))]
        self.data.loc[:,RAU_DUOI+f'{period}'] = [abs(min(data[CLOSE][candle], data[OPEN][candle - period]) - data[LOW][candle]) * self._x if candle>=period else 0 for candle in range(len(data.index))]

    def classify_reversal_candle_pairs(self):
        logger.info('ClassObjects: Start classifying reversal candle pairs')
        reversal_types = [None for i in range(len(self.data.index))]

        for i in range(0, len(self.data.index)):
            open_prev, close_prev, high_prev, low_prev = self.data.iloc[i-1][[OPEN, CLOSE, HIGH, LOW]]
            open_curr, close_curr, high_curr, low_curr = self.data.iloc[i][[OPEN, CLOSE, HIGH, LOW]]

            if i > 2:
                open_2prev, close_2prev, high_2prev, low_2prev = self.data.iloc[i-2][[OPEN, CLOSE, HIGH, LOW]]
                if open_prev > close_prev and open_curr < close_curr and high_2prev < open_prev and close_2prev > close_curr:
                    reversal_types[i]=100000
                elif open_prev < close_prev and open_curr > close_curr and low_2prev > open_prev and close_2prev < close_curr:
                    reversal_types[i] = -100000
            if open_prev > close_prev and open_curr < close_curr and close_curr > open_prev:
                reversal_types[i] = 10000
            elif open_prev < close_prev and open_curr > close_curr and close_curr < open_prev:
                reversal_types[i]= -10000
            elif open_prev > close_prev and open_curr < close_curr and high_curr < open_prev:
                reversal_types[i]=100
            elif open_prev < close_prev and open_curr > close_curr and low_curr > close_prev:
                reversal_types[i] = -100

        self.data['reversal_type'] = reversal_types
        return self.data

    """INDICATORS HANDLE"""
    def calc_MA(self, period: int, type=CLOSE, exp=True, avg=5):
        """
        Chú ý độ dài của data_handler có the ảnh hưởng đến giá trị của EMA tại 1 thời điểm xác định
        EMA thường có giá trị ổn định tại thời điểm t cách 0 1 khoảng thời gian = 10*period
        """
        period = int(period)
        logger.info(f'ClassObjects:Start calculate MA {period}')
        ma_col = f"{MA}{period}"
        del_col = DELMAS+f'{period}'

        if exp==True:
            self.data[ma_col] = self.data[type].ewm(span=period, min_periods=period).mean().round(5)
            # initial delta
            self.data[del_col] = ((self.data[ma_col] - self.data[ma_col].shift(1)) * self._x).round(5)

        elif exp==False:
            self.data[ma_col] = self.data[type].rolling(window=period).mean()
            self.data[del_col] = (self.data[ma_col] - self.data[ma_col].shift(1)) * self._x

        if avg:
            avg_col = f'MA{period}AVG'
            self.data[avg_col] = self.data[ma_col].rolling(window=avg).mean()

        return self.data

    def calc_Divisible_MA(self, period: int, type=CLOSE, exp=True, divisor=1):
        """
        Chú ý độ dài của data_handler có the ảnh hưởng đến giá trị của EMA tại 1 thời điểm xác định
        EMA thường có giá trị ổn định tại thời điểm t cách 0 1 khoảng thời gian = 10*period
        """
        period = int(period)
        logger.info(f'ClassObjects:Start calculate MA {period}')
        ma_col = f"{MA}{period}Div{divisor}"
        del_col = DELMAS+f'{period}Div{divisor}'
        df = self.data.copy()

        df['Datetime'] = pd.to_datetime(df[DATE])
        # 2. Tạo một cột tạm thời chỉ chứa giá trị Close tại các mốc chia hết cho 5
        # Những hàng không chia hết cho 5 sẽ mang giá trị NaN (trống)
        df.loc[df['Datetime'].dt.minute % divisor == 0, 'Close_5m'] = df[type]

        # 3. Tính MA trên các giá trị 5 phút này
        # Lưu ý: Ta chỉ tính MA dựa trên các hàng NOT NULL
        # ma_5m_series = df.loc[df['Close_5m'].notnull(), 'Close_5m'].rolling(window=period).mean()
        # # 4. Gán giá trị MA vừa tính được vào df gốc
        # df['MA_5m_fitted'] = ma_5m_series
        # # 5. Sử dụng ffill() để lấp đầy các ô trống
        # # Các hàng phút 6, 7, 8, 9 sẽ lấy giá trị của phút 5
        # self.data_handler[ma_col] = df['MA_5m_fitted'].ffill()

        if exp==True:
            df['MA_5m_fitted'] = df.loc[df['Close_5m'].notnull(), 'Close_5m'].ewm(span=period, min_periods=period).mean().round(5)
            self.data[ma_col] =  df['MA_5m_fitted'].ffill()
            # initial delta
            self.data[del_col] = ((self.data[ma_col] - self.data[ma_col].shift(1)) * self._x).round(5)
        elif exp==False:
            df['MA_5m_fitted'] = df.loc[df['Close_5m'].notnull(), 'Close_5m'].rolling(window=period).mean().round(5)
            self.data[ma_col] =  df['MA_5m_fitted'].ffill()
            self.data[del_col] = (self.data[ma_col] - self.data[ma_col].shift(1)) * self._x
        #print(self.data_handler[ma_col].head(20))

        return self.data



    def check_giai_doan_MA(self, MA1: pd.DataFrame(), MA2: pd.DataFrame(), candle, period):
        logger.info(f'Check giai doan MAs tai candle {candle}')
        if candle-period<0:
            return False
        S1 = abs(sum([MA1[candle - i] - MA2[candle - i] for i in range(0, int(period/2))]))
        S2 = abs(sum([MA1[candle - i] - MA2[candle - i] for i in range(int(period/2), period)]))
        # S3_do_doc_MA2 = abs(sum([MA2[candle - i] - MA2[candle] for i in range(0, int(period/2))]))
        # S4_do_doc_MA2 = abs(sum([MA2[candle - i] - MA2[candle - int(period/2)] for i in range(int(period/2), period)]))
        logger.info(f'{S1} and {S2}')
        if S1>S2:
            return 'phan_ky'
        elif S1<8*S2/10:#tỉ lệ càng bé thì hội tụ càng mạnh
            logger.info('giai doan hoi tu')
            return 'hoi_tu'

    @classmethod
    def calc_CCI_(cls, data, period=14):
        logger.info(f'ClassObjects:Start calculate CCI {period}')
        tp = (data[HIGH] + data[LOW] + data[CLOSE]) / 3
        sma = tp.rolling(window=period).mean()
        mean_deviation = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        cci = (tp - sma) / (0.015 * mean_deviation)
        data[MFI_] = cci
        return data

    def calc_CCI(self, period=14):
        logger.info(f'ClassObjects:Start calculate CCI {period}')
        tp = (self.data[HIGH] + self.data[LOW] + self.data[CLOSE]) / 3
        sma = tp.rolling(window=period).mean()
        mean_deviation = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        cci = (tp - sma) / (0.015 * mean_deviation)
        self.data[MFI_] = cci
        return self.data

    def calc_stochastic(self, period_k=14, period_d=3):
        logger.info(f'ClassObjects:Start calculate Stochastic {period_k} {period_d}')
        lowest_low = self.data[LOW].rolling(window=period_k).min()
        highest_high = self.data[HIGH].rolling(window=period_k).max()
        k = 100 * (self.data[CLOSE] - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(window=period_d).mean()
        self.data[STO] = k
        self.data[f'{STO}_D{period_d}'] = d
        return self.data

    def check_FVG(self, data, candle):
        if candle+1 >= len(data):
            return False
        logger.info(f'Check embalance tai {data[DATE][candle]}')
        if data[MAU_NEN][candle]>0 and self._x*(data[LOW][candle+1]-data[HIGH][candle-1])>0.25*data[MAU_NEN][candle]:
            pass
        elif data[MAU_NEN][candle]<0 and self._x*(data[HIGH][candle+1]-data[LOW][candle-1])>0.25*data[MAU_NEN][candle]:
            pass
        else:
            return False

        logger.info('Co FVG')
        return True

    @classmethod
    def tinhVector(cls, data, c, period=3):
        checkingData = data[c - period:c + 1].copy().reset_index(drop=True)
        checkingData = (checkingData[[OPEN, HIGH, LOW, CLOSE]] - data[OPEN][c]).round(5)
        res = []
        for A in range(period):
           res = res+list(checkingData.loc[A])
        #print(res)
        return res

    def calcDeltaMAs(self, smallMA: int, bigMA: int):
        smallMA = int(smallMA)
        bigMA = int(bigMA)
        logger.info(f'ClassObjects:Start calculate DeltaMA of {smallMA} and {bigMA}')
        DeltaMAs = [None for i in range(len(self.data.index))]
        if smallMA>1:
            for candle in range(bigMA+1, len(self.data.index)):
                DeltaMAs[candle] = (self.data[MA+f'{smallMA}'][candle]-self.data[MA+f'{bigMA}'][candle])*self._x
            self.data[DELMAS+f'{smallMA}x{bigMA}']=DeltaMAs
        elif smallMA==1:
            for candle in range(bigMA+1, len(self.data.index)):
                DeltaMAs[candle] = (self.data[CLOSE][candle] - self.data[MA + f'{bigMA}'][candle]) * self._x
            self.data[DELMAS + f'{smallMA}x{bigMA}'] = DeltaMAs
        else:
            logger.info(f'Wrong value {smallMA}')
            raise f'Wrong value {smallMA}'
        return self.data

    def ichimoku_cloud(self, tenkan_period=9, kijun_sen_period=26, senkou_span_b_period=52):
        """
        Tính toán các đường của chỉ báo Ichimoku Cloud

        Args:
          df: DataFrame chứa dữ liệu giá
          tenkan_period: Số chu kỳ cho đường Tenkan-sen
          kijun_sen_period: Số chu kỳ cho đường Kijun-sen
          senkou_span_b_period: Số chu kỳ cho đường Senkou Span B

        Returns:
          DataFrame chứa các đường của chỉ báo Ichimoku Cloud
        """
        df = self.data
        # Tính toán các đường cơ bản
        df['High'] = df[HIGH].rolling(window=tenkan_period).max()
        df['Low'] = df[LOW].rolling(window=tenkan_period).min()
        df['tenkan_sen'] = (df['High'] + df['Low']) / 2
        df['kijun_sen'] = (df[HIGH].rolling(window=kijun_sen_period).max() + df[LOW].rolling(
            window=kijun_sen_period).min()) / 2

        # Tính toán các đường Senkou Span
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(kijun_sen_period)
        df['senkou_span_b'] = ((df[HIGH].rolling(window=senkou_span_b_period).max() + df[LOW].rolling(
            window=senkou_span_b_period).min()) / 2).shift(kijun_sen_period)

        # Tính toán đường Chikou Span
        df['chikou_span'] = df[CLOSE].shift(-kijun_sen_period)
        return df

    @classmethod
    def candle_typeV1(cls, hist=pd.DataFrame, index=0):
        """
        Check xem nen dang xet co phai nen sideway ko
        :param hist:
        :param index:
        :return:
        """
        than_nen = hist[THAN_NEN][index]
        rau_tren = hist[RAU_TREN][index]
        rau_duoi = hist[RAU_DUOI][index]
        max_vol = hist[MAX_VOL][index]
        mau_nen = hist[MAU_NEN][index]
        # logger.info(f'Check type of candle than nen {than_nen}, rau tren {rau_tren} rau duoi {rau_dui} vol {max_vol}')
        if than_nen < 15:
            logger.info('ko xu huong')
            return 'ko xu huong'
        elif rau_tren>rau_duoi*2 and than_nen<25:
            logger.info('pinbar tren')
            return 'pinbar tren'
        elif rau_duoi>rau_tren*2 and than_nen<25:
            logger.info('pinbar duoi')
            return 'pinbar duoi'
        elif rau_tren>rau_duoi and mau_nen<0 and than_nen>=0.8*max_vol:
            logger.info('nhan chim do')
            return 'nhan chim do'
        elif rau_tren<rau_duoi and mau_nen>0 and than_nen>=0.8*max_vol:
            logger.info('nhan chim xanh')
            return 'nhan chim xanh'
        elif rau_tren < 0.0001 and rau_duoi > than_nen / 2 and hist[MAU_NEN][index] < 0:
            logger.info('rut chan do')
            return 'rut chan do'
        elif rau_tren>rau_duoi and mau_nen>=25:
            logger.info('rut chan xanh')
            return 'rut chan xanh'
        elif rau_tren<rau_duoi and mau_nen<=-25:
            logger.info('rut chan do')
            return 'rut chan do'
        elif than_nen == 0:
            logger.info('nen zero')
            return 'nen zero'
        else:
            logger.info(f'Chua dinh dang duoc nen {than_nen}, {max_vol}, {rau_tren}, {rau_duoi}')
            return False

    @classmethod
    def get_preH1Candle(cls, df, candle: int=None, mode: str='last'):
        """
        return: OLHC of pre H1
        """
        logger.info(f'get candle at {candle}')

        if mode=='last':
            currenttime = str(df[DATE][candle])
            df = df[candle-100:candle]
        df["dates"] = pd.to_datetime(df["dates"], format="%Y-%m-%d %H:%M:%S")
        df = df.set_index("dates")
        ohlc_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }
        h1df = df.resample('H').apply(ohlc_dict).dropna()
        h1df['dates'] = h1df.index.strftime('%Y-%m-%d %H:%M:%S')
        return h1df

    @classmethod
    def get_preH4Candle(cls, df, candle: int = None, mode: str = 'last'):
        """
        return: OLHC of pre H1
        """
        logger.info(f'get candle at {candle}')

        if mode == 'last':
            currenttime = str(df[DATE][candle])
            df = df[candle - 200:candle]
        df.loc[:, "dates"] = pd.to_datetime(df["dates"], format="%Y-%m-%d %H:%M:%S")
        df = df.set_index("dates")
        ohlc_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }
        h4df = df.resample('4H').apply(ohlc_dict).dropna()
        h4df['dates'] = h4df.index.strftime('%Y-%m-%d %H:%M:%S')
        return h4df

    def getPreM15(self, df, index=None, mode='last'):
        if mode=='last':
            df = df[index-100::]
        df["dates"] = pd.to_datetime(df["dates"], format="%Y-%m-%d %H:%M:%S")
        df = df.set_index("dates")
        ohlc_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }
        m15_df = df.resample('15T').apply(ohlc_dict).dropna()
        m15_df['dates'] = m15_df.index.strftime('%Y-%m-%d %H:%M:%S')
        return m15_df

    def calc_CucTri(self, data, minmaxs: list = [5, 15, 30]):
        CUCTRILS = [f'CUCTRI{minmaxs[0]}', f'CUCTRI{minmaxs[1]}', f'CUCTRI{minmaxs[2]}']
        for col in CUCTRILS:
            data.insert(loc=len(data.columns) - 1, column=col, value=[None for i in range(len(data))])
        for candle in range(max(minmaxs)+1, len(data.index) - max(minmaxs)-1):
            """"""
            if data[HIGH][candle] == max(data[HIGH][candle - (minmaxs[-1]):candle + minmaxs[-1]+1]):
                data[CUCTRILS[-1]][candle] = 'max'
            elif data[LOW][candle] == min(data[LOW][candle - (minmaxs[-1]):candle + minmaxs[-1]+1]):
                data[CUCTRILS[-1]][candle] = 'min'
            if data[HIGH][candle] == max(data[HIGH][candle - (minmaxs[1]):candle + minmaxs[1]+1]):
                data[CUCTRILS[1]][candle] = 'max'
            elif data[LOW][candle] == min(data[LOW][candle - (minmaxs[1]):candle + minmaxs[1]+1]):
                data[CUCTRILS[1]][candle] = 'min'
            if data[HIGH][candle] == max(data[HIGH][candle - (minmaxs[0]):candle + minmaxs[0]+1]):
                data[CUCTRILS[0]][candle] = 'max'
            elif data[LOW][candle] == min(data[LOW][candle - (minmaxs[0]):candle + minmaxs[0]+1]):
                data[CUCTRILS[0]][candle] = 'min'
        return data

    def detect_KeyForPrice(self, order, area, pos):
        data = self.data
        logger.info(f'Detect key for {area}')
        start, end = 0, 0
        for candle in range(pos-5, 0 , -1):
            if any(area[0]<=data[x][candle]<=area[1] for x in [LOW, HIGH, open, CLOSE]) and start==0:
                start = candle
                if data[THAN_NEN][candle]>1.2*data[THAN_NEN][pos-1]:
                    return candle
                break
        if start==0:
            return False

        logger.info(f'Get start: {start} to {end}')
        for candle in range(start, start-20, -1):
            if order=='sell' and any(area[0]<=x<=area[1] for x in [data[LOW][candle], min(data[[OPEN, CLOSE]].iloc[candle])]) and\
                    data[LOW][candle]==min(data[LOW][candle-9:candle+9]):
                return candle
            elif order=='buy' and any(area[0]<=x<=area[1] for x in [data[HIGH][candle], max(data[[OPEN, CLOSE]].iloc[candle])]) and\
                    data[HIGH][candle]==max(data[HIGH][candle-9:candle+9]):
                return candle

        return False

    def calc_trendOfCucTri(self, data, range_=30):
        CUCTRILS = f'CUCTRI{range_}'
        if any(CUCTRILS in col for col in data.columns):
            logger.info('Found CUCTRI in data_handler columns')
        else:
            raise "Not found cuc tri in data_handler columns"
        LAST4 = []
        trend = False
        for candle in range(len(data),0,-1):
            if data[CUCTRILS][candle] in ['min', 'max'] and len(LAST4)<=4:
                LAST4.append(data.iloc[candle])
            elif data[CUCTRILS][candle] in ['min', 'max']:
                LAST4.pop(0)
                LAST4.append(data[CLOSE][candle])
                TRENDs = [LAST4[i+1]-LAST4[i] for i in [0,1,2]]
                if (TRENDs[-1]>0 and TRENDs[-1]>abs(TRENDs[-2])) or (TRENDs[-1]<0 and abs(TRENDs[-1]) < TRENDs[-2]):
                    trend = 'buy'
                elif (TRENDs[-1]<0 and abs(TRENDs[-1]) > TRENDs[-2]) or (TRENDs[-1]>0 and TRENDs[-1]<abs(TRENDs[-2])) :
                    trend = 'sell'
                else:
                    raise 'Ko trend'
            if trend!=False:
                return trend
        return trend

    def calc_BB(self, period: int, type=CLOSE, rang_: int=1):
        logger.info(f'ClassObjects:Start calc BB {period}')
        self.BBPer = period
        self.data[f'{MA}'] = self.data[type].rolling(window=period).mean()
        STD = self.data[CLOSE].rolling(window=period).std()
        _UPPER = self.data[f'{MA}'] + (STD * 2)
        _LOWER = self.data[f'{MA}'] - (STD * 2)
        self.data[f'Upper{period}'] = _UPPER
        self.data[f'Lower{period}'] = _LOWER
        #self.DATA.(columns='STD')
        # self.data_handler.loc[:,DELMAS+UPPER+str(period)] = [self._x*(_UPPER[x]-_UPPER[x-rang_]) if x>period else 0 for x in range(len(_UPPER))]
        # self.data_handler.loc[:,DELMAS+LOWER+str(period)] = [self._x * (_LOWER[x] - _LOWER[x - rang_]) if x > period else 0 for x in range(len(_LOWER))]
        self.data.pop(MA)
        return self.data

    def calc_Divisible_BB(self, period: int, type=CLOSE, exp=True, divisor=1):
        """
        Chú ý độ dài của data_handler có the ảnh hưởng đến giá trị của EMA tại 1 thời điểm xác định
        EMA thường có giá trị ổn định tại thời điểm t cách 0 1 khoảng thời gian = 10*period
        """
        period = int(period)
        logger.info(f'ClassObjects:Start calculate MA {period}')
        ma_col = f"{MA}{period}Div{divisor}"
        del_col = DELMAS+f'{period}Div{divisor}'
        df = self.data.copy()
        df['Datetime'] = pd.to_datetime(df[DATE])
        df.loc[df['Datetime'].dt.minute % divisor == 0, 'Close_5m'] = df[type]
        if exp==True:
            df['MA_5m_fitted'] = df.loc[df['Close_5m'].notnull(), 'Close_5m'].ewm(span=period, min_periods=period).mean()
            df['std'] = df.loc[df['Close_5m'].notnull(), 'Close_5m'].ewm(span=period, min_periods=period).std()
            self.data[ma_col] =  df['MA_5m_fitted'].ffill()
            self.data[del_col] = ((self.data[ma_col] - self.data[ma_col].shift(1)) * self._x)
        elif exp==False:
            df['MA_5m_fitted'] = df.loc[df['Close_5m'].notnull(), 'Close_5m'].rolling(window=period).mean()
            df['std'] =  df.loc[df['Close_5m'].notnull(), 'Close_5m'].rolling(window=period).std()
            self.data[ma_col] =  df['MA_5m_fitted'].ffill()
            self.data[del_col] = (self.data[ma_col] - self.data[ma_col].shift(1)) * self._x

        STD = df['std'].ffill()
        _UPPER = self.data[ma_col] + (STD * 2)
        _LOWER = self.data[ma_col] - (STD * 2)
        self.data[f'Upper{period}'] = _UPPER
        self.data[f'Lower{period}'] = _LOWER

        return self.data

    def calc_DeltaBB(self, type='close', line: str='all', period: int=None):
        logger.info('ClassObjects: Start calc DeltaBB')
        if period == None:
            period = self.BBPer
        if line=='all':
            lines = [UPPER+f'{period}', LOWER+f'{period}']
        else:
            lines = line+f'{period}'
        if len(lines)>1:
            DeltaBBs = [None for i in range(len(self.data.index))]
            for line in lines:
                DeltaBBs = (self.data[CLOSE] - self.data[line]) * self._x
                self.data[DELMAS + f'1x{line}'] = DeltaBBs
        else:
            DeltaBBs = (self.data[CLOSE] - self.data[lines]) * self._x
            self.data[DELMAS + f'1x{line}'] = DeltaBBs

        return self.data

    def check_giai_doan_BB(self, BB: int, period: int, candle: int):
        logger.info(f'Check giai doan BB tai {self.data[DATE][candle]}')
        Upper = self.data[f'Upper{BB}']
        Lower = self.data[f'Lower{BB}']
        # Close = self.data_handler[CLOSE][candle]
        S1 = sum([Upper[candle-i]-Lower[candle-i] for i in range(0,int(period/2))])
        S2 = sum([Upper[candle-i]-Lower[candle-i] for i in range(int(period/2),period+1)])
        S_TRI_Upper = abs(sum([Upper[candle-i]-Upper[candle-period] for i in range(0,period)]))
        S_TRI_Lower = abs(sum([Lower[candle-period]-Lower[candle-i] for i in range(0,period)]))

        if S1>S2*1.1:
            return 'phan_ky'
        elif S2>S1:
            return 'hoi_tu'
        elif 9/10<S1/S2<10/9 and S_TRI_Lower<0.3*period and S_TRI_Upper<0.3*period and S1<15*period/2:
            return 'tich_luy'
        elif 9/10<S1/S2<10/9:
            return 'tang_giam_on_dinh'
        else:
            return 'ko_on_dinh'

    def check_giai_doan_EMA(self, short_period: int, long_period: int, candle: int, lookback: int = 5):
        """
        Kiểm tra giai đoạn của 2 đường EMA: Phân kỳ, Hội tụ, Tích lũy
        """
        logger.info(f'Check giai doan EMA {short_period}-{long_period} tai {candle}')
        
        ma_short = f"{MA}{short_period}"
        ma_long = f"{MA}{long_period}"
        
        if ma_short not in self.data.columns or ma_long not in self.data.columns:
            logger.error(f"Columns {ma_short} or {ma_long} not found. Please run calc_MA first.")
            return False

        # Lấy dữ liệu khoảng cách (spread) giữa 2 đường
        spread = (self.data[ma_short] - self.data[ma_long]).abs()
        
        # 1. Kiểm tra Tích lũy (Accumulation)
        # Đặc điểm: Khoảng cách nhỏ và biến động thấp
        current_price = self.data[CLOSE][candle]
        recent_spreads = spread.iloc[candle-lookback:candle+1]
        avg_spread = recent_spreads.mean()
        
        # Ngưỡng tích lũy: Khoảng cách trung bình nhỏ hơn 0.05% giá (có thể điều chỉnh tùy cặp tiền)
        # Hoặc so sánh với lịch sử biến động
        threshold_accumulation = current_price * 0.0005
        
        if avg_spread < threshold_accumulation:
            return 'tich_luy'
            
        # 2. Kiểm tra Phân kỳ (Divergence - Mở rộng) và Hội tụ (Convergence - Thu hẹp)
        # So sánh khoảng cách hiện tại với quá khứ
        current_spread = spread[candle]
        past_spread = spread[candle - lookback]
        
        # Tính độ dốc của spread (Slope)
        # Nếu spread tăng dần -> Phân kỳ
        # Nếu spread giảm dần -> Hội tụ
        
        if current_spread > past_spread:
            # Kiểm tra xem có phải tăng đáng kể không
            if current_spread > past_spread * 1.02: 
                return 'phan_ky'
            else:
                return 'sideway' # Tăng nhẹ, chưa rõ ràng
        elif current_spread < past_spread:
             # Kiểm tra xem có phải giảm đáng kể không
            if current_spread < past_spread * 0.98:
                return 'hoi_tu'
            else:
                return 'sideway' # Giảm nhẹ
        
        return 'sideway'

    def calc_MACD(self, period12: int=12, period26: int=26, signal: int=9):
        logger.info(f'ClassObjects: Start calc MACD {period12} {period26} {signal}')
        self.data[f'{MA}12'] = self.data[CLOSE].ewm(span=period12-1, adjust=False).mean()
        self.data[f'{MA}26'] = self.data[CLOSE].ewm(span=period26-1, adjust=False).mean()
        self.data[MACD] = self.data[f'{MA}12'] - self.data[f'{MA}26']
        self.data[Signal] = self.data[MACD].ewm(span=signal-1, adjust=False).mean()
        self.data.pop(f'{MA}12')
        self.data.pop(f'{MA}26')
        return self.data

    def dem_song_Elliot(self, pos, order, cutPos=0):
        logger.info(f"Detect song Elliot cho {pos} va cut tai {cutPos}")
        if pos<30:
             return False
        #print(f"{pos}, {cutPos}")
        data = self.data
        if cutPos==0:
            cutPos = pos
        start = False
        if order==False:
            return 6
        for candle in range(cutPos, 30, -1):
            if order=='buy' and data[LOW][candle]==min(data[LOW][candle-15:candle+15]):
                start = candle
                break
            elif order=='sell' and data[HIGH][candle]==max(data[HIGH][candle-15:candle+15]):
                start = candle
                break
        if start==False:
            logger.info('Ko tim thay start cua song')
            return False
        logger.info(f'Song start tai {data[DATE][start]}')
        count_song = []
        for candle in range(start, pos-5):
            if order=='buy' and data[HIGH][candle]==max(data[HIGH][candle-3:candle+3]) and\
                    any(data[HIGH][c]<max(data[CLOSE][candle-3:candle+3]) for c in range(candle+1,candle+4)):
                if len(count_song)==0 or (len(count_song)>0 and any(data[LOW][c]>data[HIGH][count_song[-1]] for c in range(candle-2, candle+3))):
                    count_song.append(candle)
                    logger.info(f'Got dd tai {data[DATE][candle]}')
            elif order=='sell' and data[LOW][candle]==min(data[LOW][candle-3:candle+3]) and \
                    any(data[LOW][c]>min(data[CLOSE][candle-3:candle+3]) for c in range(candle+1,candle+4)):
                if len(count_song)==0 or (len(count_song)>0 and any(data[HIGH][c]<data[LOW][count_song[-1]] for c in range(candle-2, candle+3))):
                    count_song.append(candle)
                    logger.info(f'Got dd tai {data[DATE][candle]}')
        return len(count_song)

    def calculate_wto(self, period):
        self.data['SMA'] = self.data['close'].rolling(window=period).mean()
        self.data['SMA1'] = self.data['close'].rolling(window=2*period).mean()
        self.data[WTO+str(period)] = (self.data['close'] - self.data['SMA']) / self.data['SMA'] * 100
        return self.data

    def calc_parabolic_sar(self, start_acceleration=0.02, acceleration_increment=0.02, max_acceleration=0.2):
        logger.info('ClassObjects: Start calc Parabolic SAR')
        data = self.data
        sar = [data[LOW][0]]
        af = start_acceleration
        ep = data[HIGH][0]
        uptrend = True

        for i in range(1, len(data)):
            if uptrend:
                sar.append(sar[-1] + af * (ep - sar[-1]))
                if data[LOW][i] < sar[-1]:
                    uptrend = False
                    sar[-1] = ep
                    af = start_acceleration
                    ep = data[LOW][i]
            else:
                sar.append(sar[-1] + af * (ep - sar[-1]))
                if data[HIGH][i] > sar[-1]:
                    uptrend = True
                    sar[-1] = ep
                    af = start_acceleration
                    ep = data[HIGH][i]

            if uptrend and data[HIGH][i] > ep:
                ep = data[HIGH][i]
                af = min(af + acceleration_increment, max_acceleration)
            elif not uptrend and data[LOW][i] < ep:
                ep = data[LOW][i]
                af = min(af + acceleration_increment, max_acceleration)

        self.data['ParabolicSAR'] = sar
        return self.data

    def detect_mo_hinh_gia(self, type=CLOSE):
        MO_HINH = ['Nope' for i in range(len(self.data))]
        data = self.data
        for x in range(len(self.data)):
            if (data[type][x]-data[type][x-1])*(data[type][x]-data[type][x-2])<0:
                if (data[type][x]-data[type][x-1])>0:
                    MO_HINH[x]='day'
                elif (data[type][x]-data[type][x-1])<0:
                    MO_HINH[x]='dinh'
        self.data[MOHINH] = MO_HINH
        return self.data

    def get_current_mohinhgia(self, candle, type='close'):
        if MOHINH not in self.data.columns:
            self.detect_mo_hinh_gia(type)
        #print(self.data_handler.columns)
        DINH, DAY = [], []
        for x in range(candle, 0, -1):
            if self.data[MOHINH][x]=='dinh':
                DINH.append(x)
            elif self.data[MOHINH][x]=='day':
                DAY.append(x)

            if len(DINH)>=2 and len(DAY)>=2:
                break
        if len(DINH) < 2 or len(DAY) < 2:
            return False

        if self.data[type][DINH[-1]]>self.data[type][DINH[-2]] and self.data[type][DAY[-1]]>=self.data[type][DAY[-2]]:
            return 'tang'
        if self.data[type][DINH[-1]] <= self.data[type][DINH[-2]] and self.data[type][DAY[-1]] < self.data[type][
            DAY[-2]]:
            return 'giam'
        if self.data[type][DINH[-1]] > self.data[type][DINH[-2]] and self.data[type][DAY[-1]] < self.data[type][
            DAY[-2]]:
            return 'loe'
        if self.data[type][DINH[-1]] < self.data[type][DINH[-2]] and self.data[type][DAY[-1]] > self.data[type][
            DAY[-2]]:
            return 'nen'

        return False

    def calculate_RSI(self, period=14):
        """
        Calculate the Relative Strength Index (RSI) for a DataFrame.

        Parameters:
        - df (pd.DataFrame): DataFrame containing the stock data_handler.
        - column (str): The column to calculate RSI from (default is 'Close').
        - period (int): The period for calculating RSI (default is 14).

        Returns:
        - pd.Series: A series containing the RSI values.
        """
        # Calculate price differences
        df = self.data
        delta = df[CLOSE].diff()

        # Separate positive gains (up) and negative gains (down)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Calculate the rolling average of the gains and losses
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()

        # Calculate the Relative Strength (RS)
        rs = avg_gain / avg_loss

        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        self.data["RSI" + f'{period}'] = rsi
        return rsi

    def calc_MFI(self, period=15):
        logger.info(f'ClassObjects:Calc MFI {period} candles')
        self.MFI_period = period
        typical_price = (self.data['high'] + self.data['low'] + self.data['close']) / 3
        raw_money_flow = typical_price * self.data[VOLUME]
        positive_flow = []
        negative_flow = []
        for i in range(1, len(typical_price)):
            if typical_price[i] > typical_price[i - 1]:
                positive_flow.append(raw_money_flow[i])
                negative_flow.append(0)
            elif typical_price[i] < typical_price[i - 1]:
                negative_flow.append(raw_money_flow[i])
                positive_flow.append(0)
            else:
                positive_flow.append(0)
                negative_flow.append(0)
        positive_flow = pd.Series(positive_flow)
        negative_flow = pd.Series(negative_flow)
        money_ratio = positive_flow.rolling(window=period).sum() / negative_flow.rolling(window=period).sum()
        mfi = 100 - (100 / (1 + money_ratio))
        lastmfi = len(mfi) - 1
        lenMFI = len(self.data.index) - 1
        do_lech = lenMFI - lastmfi
        MFIs = [None for i in range(len(self.data.index))]
        for candle in range(len(self.data.index) - 1, 0, -1):
            MFIs[candle] = mfi[candle - lenMFI + lastmfi]

        self.data["MFI" + f'{period}'] = MFIs
        return self.data

    #Handle DATA stream
    def generate_csv(self, filePath: str or None):
        try:
            if filePath:
                self.data.to_csv(filePath, index=False)
            else:
                self.data.to_csv(r'thongke.csv', index=False)
        except:
            print('Fail when generate file csv')
            logger.debug('Fail when generate file csv')
            return False

    def get_data(self):
        return self.data

    def calc_S(self, period: int):
        for col in self.data.columns:
            if DELMAS in col:
                self.data[f'S{period}_'+col] = self.data[col].rolling(window=period).mean()
        return self.data

    def add_deals(self, deals: list):
        """
        Add list deals vao hist
        :param hist:
        :param deals:
        :return:
        """
        logger.info(f'ClassObjects:ADD DEALS*****************')
        #hist.insert(loc=len(hist.columns), column='DEALS', value=None)
        col = [None for i in range(len(self.data.index))]
        for deal in deals:
            if deal['start_index']>=len(self.data.index):
                break
            col[deal['start_index']] = deal['sl']
        self.data.loc[:,'DEALS'] = col
        return self.data

    def calc_ZigZag(self, deviation=5.0, depth=12, backstep=3):
        """
        Tính chỉ báo ZigZag để xác định các đỉnh và đáy quan trọng

        Parameters:
        - deviation: Phần trăm thay đổi tối thiểu để xác nhận một đỉnh/đáy mới (mặc định 5.0%)
        - depth: Số nến tối thiểu giữa các đỉnh/đáy (mặc định 12)
        - backstep: Số nến quay lại để tìm đỉnh/đáy cục bộ (mặc định 3)

        Returns:
        - DataFrame với cột 'ZigZag' chứa giá trị tại các điểm đảo chiều
        """
        logger.info(f'ClassObjects: Start calculate ZigZag with deviation={deviation}%, depth={depth}, backstep={backstep}')

        data = self.data
        n = len(data)

        # Khởi tạo mảng kết quả
        zigzag = [None for _ in range(n)]

        # Tìm các đỉnh và đáy cục bộ
        highs = [None for _ in range(n)]
        lows = [None for _ in range(n)]

        # Xác định các đỉnh và đáy cục bộ dựa trên depth
        for i in range(depth, n - depth):
            # Kiểm tra đỉnh cục bộ
            is_high = True
            for j in range(i - depth, i + depth + 1):
                if j != i and data[HIGH][j] >= data[HIGH][i]:
                    is_high = False
                    break
            if is_high:
                highs[i] = data[HIGH][i]

            # Kiểm tra đáy cục bộ
            is_low = True
            for j in range(i - depth, i + depth + 1):
                if j != i and data[LOW][j] <= data[LOW][i]:
                    is_low = False
                    break
            if is_low:
                lows[i] = data[LOW][i]

        # Xây dựng ZigZag dựa trên deviation
        last_pivot_type = None  # 'high' hoặc 'low'
        last_pivot_index = None
        last_pivot_value = None

        for i in range(n):
            if highs[i] is not None or lows[i] is not None:
                current_high = highs[i]
                current_low = lows[i]

                if last_pivot_index is None:
                    # Điểm đầu tiên
                    if current_high is not None:
                        last_pivot_type = 'high'
                        last_pivot_value = current_high
                        last_pivot_index = i
                        zigzag[i] = current_high
                    elif current_low is not None:
                        last_pivot_type = 'low'
                        last_pivot_value = current_low
                        last_pivot_index = i
                        zigzag[i] = current_low
                else:
                    # Kiểm tra điều kiện deviation
                    if last_pivot_type == 'high':
                        # Đang ở đỉnh, tìm đáy tiếp theo
                        if current_low is not None:
                            change_percent = abs((current_low - last_pivot_value) / last_pivot_value * 100)
                            if change_percent >= deviation and (i - last_pivot_index) >= backstep:
                                zigzag[i] = current_low
                                last_pivot_type = 'low'
                                last_pivot_value = current_low
                                last_pivot_index = i
                        # Cập nhật đỉnh cao hơn nếu có
                        elif current_high is not None and current_high > last_pivot_value:
                            zigzag[last_pivot_index] = None
                            zigzag[i] = current_high
                            last_pivot_value = current_high
                            last_pivot_index = i

                    elif last_pivot_type == 'low':
                        # Đang ở đáy, tìm đỉnh tiếp theo
                        if current_high is not None:
                            change_percent = abs((current_high - last_pivot_value) / last_pivot_value * 100)
                            if change_percent >= deviation and (i - last_pivot_index) >= backstep:
                                zigzag[i] = current_high
                                last_pivot_type = 'high'
                                last_pivot_value = current_high
                                last_pivot_index = i
                        # Cập nhật đáy thấp hơn nếu có
                        elif current_low is not None and current_low < last_pivot_value:
                            zigzag[last_pivot_index] = None
                            zigzag[i] = current_low
                            last_pivot_value = current_low
                            last_pivot_index = i

        # Thêm cột ZigZag vào DataFrame
        self.data['ZigZag'] = zigzag

        # Tạo thêm cột ZigZag_Type để đánh dấu đỉnh/đáy
        zigzag_type = [None for _ in range(n)]
        for i in range(n):
            if zigzag[i] is not None:
                if i > 0:
                    # Tìm điểm ZigZag trước đó
                    prev_idx = None
                    for j in range(i - 1, -1, -1):
                        if zigzag[j] is not None:
                            prev_idx = j
                            break

                    if prev_idx is not None:
                        if zigzag[i] > zigzag[prev_idx]:
                            zigzag_type[i] = 'high'
                        else:
                            zigzag_type[i] = 'low'

        self.data['ZigZag_Type'] = zigzag_type

        logger.info(f'ClassObjects: ZigZag calculation completed')
        return self.data


def calculate_ama(
        df: pd.DataFrame,
        price_col: str = "close",
        er_period: int = 10,
        fast_period: int = 2,
        slow_period: int = 30,
        ama_col: str = "AMA"
) -> pd.DataFrame:
    """
    Tính Kaufman's Adaptive Moving Average (AMA) cho một DataFrame pandas.

    Tham số:
    - df: DataFrame có cột giá (price_col).
    - price_col: tên cột giá (mặc định "close").
    - er_period: số nửa-khoảng dùng để tính Efficiency Ratio (ER). (mặc định 10)
    - fast_period: chu kỳ nhanh để tính smoothing constant (mặc định 2).
    - slow_period: chu kỳ chậm để tính smoothing constant (mặc định 30).
    - ama_col: tên cột kết quả AMA được thêm vào DataFrame.

    Trả về:
    - DataFrame (bản sao) có thêm cột ama_col với giá trị AMA. Không thay đổi df gốc.

    Ghi chú:
    - Công thức:
        ER = abs(price[i] - price[i-n]) / sum(abs(price[k] - price[k-1]) for k in i-n+1..i)
        fastSC = 2 / (fast_period + 1)
        slowSC = 2 / (slow_period + 1)
        SC = (ER * (fastSC - slowSC) + slowSC) ** 2
        AMA[i] = AMA[i-1] + SC * (price[i] - AMA[i-1])
    - Giá khởi tạo AMA tại index = er_period là trung bình của các giá đầu tiên (0..er_period).
    - Nếu có missing values trong vùng tính ER, kết quả tương ứng là NaN.
    """
    if price_col not in df.columns:
        raise KeyError(f"Column '{price_col}' not found in DataFrame")

    # Copy để không sửa df gốc
    out = df.copy()

    price = out[price_col].astype(float).reset_index(drop=True)
    n = len(price)
    ama = np.full(n, np.nan, dtype=float)

    if n == 0:
        out[ama_col] = ama
        return out

    # chuẩn bị smoothing constants cơ bản
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)

    # nếu dữ liệu không đủ để có ER, trả về AMA công bằng NaN (hoặc khởi tạo sớm)
    start = max(er_period, 1)

    # khởi tạo AMA tại vị trí 'start' bằng trung bình của window [0..start]
    # (Một lựa chọn an toàn khi không có giá trị trước đó)
    init_window_end = start
    if init_window_end < n:
        window_vals = price.iloc[: init_window_end + 1]
        if window_vals.isna().any():
            # nếu trong vùng khởi tạo có NaN thì không thể khởi tạo AMA
            out[ama_col] = ama
            return out
        ama[start] = window_vals.mean()
    else:
        # không đủ dữ liệu để khởi tạo
        out[ama_col] = ama
        return out

    # tính AMA dần cho từng điểm từ start+1 trở đi
    for i in range(start + 1, n):
        # phải có đủ dữ liệu để tính ER: từ i-er_period đến i
        if i - er_period < 0:
            ama[i] = np.nan
            continue

        window = price.iloc[i - er_period: i + 1]  # độ dài er_period+1
        if window.isna().any():
            ama[i] = np.nan
            continue

        change = abs(price.iat[i] - price.iat[i - er_period])
        volatility = np.abs(np.diff(window.values)).sum()

        er = 0.0 if volatility == 0 else (change / volatility)

        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

        prev = ama[i - 1]
        # nếu prev NaN (ví dụ vùng trước có NaN), cố gắng dùng giá hiện tại làm prev
        if np.isnan(prev):
            prev = price.iat[i - 1]

        ama[i] = prev + sc * (price.iat[i] - prev)

    # đặt cột kết quả (khôi phục index gốc nếu cần)
    # nếu out có index khác so với reset index ở trên, align bằng vị trí
    out = out.reset_index(drop=True)
    out[ama_col] = ama
    return out

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

def is_cut(MA1, MA2, candle):
    """
    Kiem tra xem co phai la cut ko
    :param MA1:
    :param MA2:
    :param candle:
    :return:
    """
    if MA1[candle-1]<MA2[candle-1] and MA1[candle]>=MA2[candle]:
        return 1
    elif MA1[candle-1]>MA2[candle-1] and MA1[candle]<=MA2[candle]:
        return -1
    else:
        return False

def chuanHoa(df, pos, quan_of_Candle, MAs):
    logger.info(f'Chuan hoa {pos}')
    Tensor = []
    BBS = [UPPER + '200', LOWER + '200']
    EMAS = [f'MA{ma}' for ma in MAs]+BBS
    DELMASS = [DELMAS+f'{ma}' for ma in MAs]+[DELMAS+x for x in BBS]

    #for loop in range(quan_of_Candle-1):
    checkingTensor = []
    checkingData = df[EMAS+DELMASS+[CLOSE]].iloc[pos-quan_of_Candle+1:pos+1].copy()
    #print(checkingData)

    # Tính distance của các gốc lines
    bienDo = []
    for i in list(combinations(EMAS, 2)):
        bienDo = bienDo+[DELMAS+i[0]+i[1]]
        checkingData[DELMAS+i[0]+i[1]] = checkingData[i[0]].sub(checkingData[i[1]], axis=0)

    for y in bienDo:#
        for row in range(1):
            checkingTensor.append(checkingData[y][row])
            checkingTensor.append(checkingData[y][-1])

    biendoMax = [abs(x)/2 for x in checkingTensor]#Lấy giá trị chuẩn hóa

    #Lay vector close minmax
    min_close, max_close = checkingData[CLOSE].min().min(), checkingData[CLOSE].max().max()
    closeTensor = [(min_close-checkingData[CLOSE][-1])*100, (max_close-checkingData[CLOSE][-1])**100]
    closeTensor = [(x, statistics.stdev(closeTensor)) for x in closeTensor]
    #Lay vetor tuc thoi cua cac lines
    for y in DELMASS:
        LINES = []
        for row in range(quan_of_Candle):
            LINES.append(checkingData[y][row])
        LINES = [round((x)/max(biendoMax), 2) for x in LINES]#chuẩn hóa Delta
        LINES = [(x, statistics.stdev(LINES)) for x in LINES]#Tinhs độ lệch chuẩn
        Tensor += LINES

    checkingTensor = [round(100*x/max(biendoMax), 2) for x in checkingTensor]

    #tinh std
    checkingTensor = closeTensor+[(x, statistics.stdev(checkingTensor)) for x in checkingTensor]+Tensor
    #print(f'Checking: {Tensor}')

    return [(abs(x[0]), x[1]) for x in checkingTensor], [1 if x[0]>=0 else -1 for x in checkingTensor]

def chuanHoaBB20(df, pos, quan_of_Candle, MAs):
    logger.info(f'Chuan hoa {pos}')
    Tensor = []
    BBS = [UPPER + '200', LOWER + '200', UPPER + '20', LOWER + '20']
    EMAS = [f'MA{ma}' for ma in MAs]+BBS
    DELMASS = [DELMAS+f'{ma}' for ma in MAs]+[DELMAS+x for x in BBS]

    #for loop in range(quan_of_Candle-1):
    checkingTensor = []
    checkingData = df[EMAS+DELMASS+[CLOSE]].iloc[pos-quan_of_Candle+1:pos+1].copy()
    #print(checkingData)

    # Tính distance của các gốc lines
    bienDo = []
    for i in list(combinations(EMAS, 2)):
        bienDo = bienDo+[DELMAS+i[0]+i[1]]
        checkingData[DELMAS+i[0]+i[1]] = checkingData[i[0]].sub(checkingData[i[1]], axis=0)

    for y in bienDo:#
        for row in range(1):
            checkingTensor.append(checkingData[y][row])
            checkingTensor.append(checkingData[y][-1])

    biendoMax = [abs(x)/2 for x in checkingTensor]#Lấy giá trị chuẩn hóa

    #Lay vector close minmax
    # min_close, max_close = checkingData[CLOSE].min().min(), checkingData[CLOSE].max().max()
    # closeTensor = [(min_close-checkingData[CLOSE][-1])*100, (max_close-checkingData[CLOSE][-1])**100]
    # closeTensor = [(x, statistics.stdev(closeTensor)) for x in closeTensor]
    #Lay vetor tuc thoi cua cac lines
    for y in DELMASS:
        LINES = []
        for row in range(quan_of_Candle):
            LINES.append(checkingData[y][row])
        LINES = [round((x)/max(biendoMax), 2) for x in LINES]#chuẩn hóa Delta
        LINES = [(x, statistics.stdev(LINES)) for x in LINES]#Tinhs độ lệch chuẩn
        Tensor += LINES

    checkingTensor = [round(100*x/max(biendoMax), 2) for x in checkingTensor]

    #tinh std
    checkingTensor =[(x, statistics.stdev(checkingTensor)) for x in checkingTensor]+Tensor
    #print(f'Checking: {Tensor}')

    return [(abs(x[0]), x[1]) for x in checkingTensor], [1 if x[0]>=0 else -1 for x in checkingTensor]

def check_huong(MA, canlde, period):
    if MA[canlde]>MA[canlde-period]:
        logger.info(f'Huong ma tang')
        return 1
    elif MA[canlde]<MA[canlde-period]:
        logger.info(f'Huong ma giam')
        return -1
    else:
        return 0

def TrichXuat(data, candle, n):
    logger.info('----------------------------------------------------------------------------------------')
    x = chuanHoa(data, candle, n)
    print(f"Check {candle}: {x}")
    draw_lan_can(data, postion=candle, dis=[n - 1, 15])

def tinh_khoangCach(biendo0, biendo, huong0, huong, distancesPos, samelevel=(0.9, 0.9)):
    # logger.info(f'Biendo: {biendo}, {huong}')
    # logger.info(f'Biendo: {biendo0}, {huong0}')
    """
    distance: n values của khoảng cách các lines, n values của DeltaMas các EMA
    """
    dau = [True if huong0[i]==huong[i] else False for i in range(len(huong0))]
    distance = [abs((biendo0[i][0] - biendo[i][0])) for i in range(len(biendo0))]

    distance = [round(x, 2) for x in distance]
    # print('--------------------------------')
    # print(biendo)
    # print(biendo0)
    # print(distance)

    biendoPass = [x < (biendo0[i][1]+biendo[i][1])*0.2 if i<distancesPos
                  else x < (biendo0[i][1]+biendo[i][1])/2 for i,x in enumerate(distance)]
    #logger.info(f'Biendo: {biendoPass}, {dau}')
    if dau.count(True) / len(dau) >= samelevel[0] and all(dau[i] == True for i in range((distancesPos)))\
            and biendoPass.count(True) / len(biendoPass) >= samelevel[1] and \
            all(biendoPass[i] == True for i in range((distancesPos))):
        return True
    else:
        return False

