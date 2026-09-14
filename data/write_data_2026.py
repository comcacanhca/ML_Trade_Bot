import os
import sys
sys.path.append(os.getcwd())
from datetime import timedelta

from vendor.scj.data_handler.download_data_mt5 import *

extra_2026 = r'D:\ML_Trade_Bot\data\raw\1M\XAUUSDm_2026.csv'
data = pd.read_csv(extra_2026)

last_day = datetime.datetime.strptime(str(data['dates'].values[-1]).split()[0], '%Y-%m-%d')
download_from = last_day + timedelta(days=1)
download_from = str(download_from).split(' ')[0]
today = str(datetime.datetime.today().date())
print(download_from, ' to ', today)

try:
    live_data = download_data_from_to(tick='XAUUSD', time_frame=mt5.TIMEFRAME_M1, from_=download_from, to_=today)
    data = pd.concat([data, live_data], ignore_index=False)
    data = data.drop_duplicates(subset=['dates'], keep='first')

except Exception as exc:
    print(f'Skip live MT5 download: {exc}')

path_outputs = [r'D:\SCJ999\data\1M\XAUUSDm_2026.csv', extra_2026]
for out_path in path_outputs:
    data[[DATE, OPEN, LOW, HIGH, CLOSE, VOLUME]].to_csv(out_path, index=False)

