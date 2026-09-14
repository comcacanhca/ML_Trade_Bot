import os
import time
import sys

from vendor.scj.log.MyLogger import MyLogger
logger = MyLogger.get_logger(__name__)

from data_handler.ClassObjects import *
pd.options.mode.chained_assignment = None

class MoPhongDeals:
    def __init__(self, data, expired, no_res_skip=False, tinh_tien=None or dict):
        self.data = data
        self.deals = []
        self.results = []
        self.results_of_predicted_deals = []
        self.running_deals = []
        self.pending_deals = []
        self.expired = expired
        self.no_res_skip = no_res_skip
        self.predicted = True
        self.force_close_time = " 28:30"

        if tinh_tien:
            OUT_COLUMNS = ['start','Date_fill','Win','Lose','No res','Profit','Loss','Fund','Add','Rut','Com','Lot', 'winrate_range']
            self.start_fund = tinh_tien['start_fund']
            self.rut = tinh_tien['rut']
            self.density = tinh_tien['density']
            self.time_add_fund = tinh_tien['time_add_fund'] if 'time_add_fund' in tinh_tien else ['17:00']
            self.tinh_tien = True
            self.current_fund = self.start_fund
            self.add_fund = 0
            self.max_fund = 0
            self.rut_fund = 0
            self.RES_TINH_TIEN = pd.DataFrame(data[DATE]).copy()
            self.RES_TINH_TIEN = self.RES_TINH_TIEN.assign(**{col: 0 for col in OUT_COLUMNS})
            self.margin_call = False
            self._X = 100#for Gold
            self.commision = tinh_tien['commision']
        else:
            self.tinh_tien = None

    def deposit(self, candle):
        self.add_fund = self.add_fund + self.start_fund - self.current_fund
        self.current_fund = self.start_fund
        self.margin_call = False
        self.RES_TINH_TIEN['Add'][candle] = self.add_fund
        self.RES_TINH_TIEN['Fund'][candle] = self.current_fund

    def add_new_deal(self, deal, min_deal_gap: int):
        logger.info('Add new deal')
        if min_deal_gap>0 and len(self.running_deals)>=1:
            last_deal = self.running_deals[-1]
            last_entry_index = last_deal['entry_index']
            start_index = deal['start_index']
            if start_index - last_entry_index <= min_deal_gap:
                logger.info('Skip by min gap')
                return

        if self.tinh_tien:
            lot = (self.current_fund * self.density)/(abs(deal['entry']-deal['sl'])*self._X)
            lot = math.floor(lot * 100) / 100
            if lot < 0.01:
                lot = 0.01

            commision = round(lot * self.commision, 2)
            deal['reward'] = round(lot * (abs(deal['entry']-deal['tp'])*self._X), 2)
            deal['risk'] = round(lot * (abs(deal['entry']-deal['sl'])*self._X), 2)
            deal['commision'] = commision
            deal['lot'] = lot
            self.RES_TINH_TIEN['start'][deal['start_index'] - 1] = deal['risk']


        if 'lot' not in deal:
            deal['lot'] = 1

        self.deals.append(deal)
        self.pending_deals.append(deal)

    def get_deals(self):
        return self.deals

    def get_pending_deals(self):
        return self.pending_deals

    def get_results(self):
        return self.results

    def get_predicted_results(self):
        return self.results_of_predicted_deals

    def set_predicted(self, predicted: bool):
        self.predicted = predicted

    def set_margin_call(self):
        logger.info('-------------------------------------Margin call')
        self.margin_call = True
        self.current_fund = 0

    def add_new_result(self, deal):
        logger.info('Add new result')
        res = deal['res']
        if self.no_res_skip and res=='No res':
            pass
        else:
            self.results.append(deal)
            if self.predicted:
                self.results_of_predicted_deals.append(deal)
                if self.tinh_tien:
                    self.update_fund_of_deal(deal)

            elif self.tinh_tien:
                self.update_fund_of_deal(deal)

    def update_fund_of_deal(self, deal):
        res = deal['res']
        if res == 'No res':
            return

        margin_call = deal['margin_call'] if 'margin_call' in deal else False
        candle = deal['end_index']
        if margin_call==True:
            logger.info(f'Deal {deal} is margin call')
            risk = 0
            reward = 0
        else:
            risk = deal['risk']+deal['commision']
            reward = deal['reward']-deal['commision']

        #print(f"Deal {deal['end_date']}: res {res}, risk: {deal['risk']}, reward: {deal['reward']}, com: {deal['commision']}, lot: {deal['lot']}, current fund: {self.current_fund}")
        if res=='Win':
            self.current_fund += reward
            self.RES_TINH_TIEN['Win'][candle] += 1
            self.RES_TINH_TIEN['Profit'][candle] += reward
        elif res=='Lose':
            self.current_fund -= risk
            self.RES_TINH_TIEN['Lose'][candle] += 1
            self.RES_TINH_TIEN['Loss'][candle] += risk

        #self.check_winrate_in_range(candle, 6881)

        if self.current_fund > self.max_fund:
            self.max_fund = self.current_fund

        if not isinstance(self.RES_TINH_TIEN['Date_fill'][candle], str):
            self.RES_TINH_TIEN['Date_fill'][candle] = deal['entry_date']
        else:
            self.RES_TINH_TIEN['Date_fill'][candle]+=deal['entry_date']

        self.RES_TINH_TIEN['Fund'][candle] = self.current_fund
        self.RES_TINH_TIEN['Com'][candle] = deal['commision']
        self.RES_TINH_TIEN['Lot'][candle] = deal['lot']

    def get_fund(self):
        projectPath = os.getcwd().split('SCJ999')[0] + 'SCJ999'
        output_dir = projectPath + fr'\results\{datetime.datetime.now().date()}'
        os.makedirs(output_dir, exist_ok=True)
        self.RES_TINH_TIEN.to_excel(
            output_dir + fr'\res_{time.localtime().tm_hour}_{time.localtime().tm_min}_{time.localtime().tm_sec}.xlsx',
            index=False)

        return self.current_fund, self.max_fund, self.add_fund, self.rut_fund

    def check_winrate_in_range(self, candle, range_check):
        if len(self.results) < 1:
            logger.info(f'Not enough deals')
            return None

        _group_deals = []
        for deal in self.results[::-1]:
            if deal['res'] != 'No res' and 0<candle - deal['start_index'] <= range_check:
                _group_deals.append(deal['res'])
            else:
                break

        if len(_group_deals) > 1:
            winrate = round(_group_deals.count('Win')*100/len(_group_deals), 0)
        else:
            winrate = -999

        if self.tinh_tien:
            self.RES_TINH_TIEN['winrate_range'][candle] = winrate

            logger.info(f'Winrate of range {range_check} candle: {winrate}')
            return len(_group_deals), winrate

        return None, None

    def check_deals(self, candle, max_running_deal = None):
        """
        deal: 'start_index':index + 1 + shift, 'entry':round(entry, 5),
               'sl':round(sl, 5), 'tp':round(tp, 5), 'type':'limit', 'order': order

        result: 'entry_index': entry_index, 'entry_date': data_handler[DATE][entry_index], 'end_date': data_handler[DATE][end_index],
                'res': deal_res, 'order': order,'commision': commision_,
                'end_index': end_index, "RR": RR,
                'R1': R1, 'entry': entry
        """
        logger.info(f'Check deals at {self.data[DATE][candle]}')
        logger.info(f'In running deals: {len(self.running_deals)}')
        logger.info(f"In pending deals: {len(self.pending_deals)}")
        def set_deal_res(deal, res):
            logger.info(f'Set deal result {deal}: {res}')
            if res == 'Win':
                deal['res'] = 'Win'
                deal['end_index'] = candle
                deal['RR'] = abs(entry - tp) / abs(entry - sl)
                deal['R1'] = abs(entry - sl)
                deal['end_date'] = self.data[DATE][candle]
            elif res == 'Lose':
                deal['res'] = 'Lose'
                deal['end_index'] = candle
                deal['RR'] = abs(entry - tp) / abs(entry - sl)
                deal['R1'] = abs(entry - sl)
                deal['end_date'] = self.data[DATE][candle]
            else:
                deal['res'] = 'No res'
                deal['end_index'] = candle
                deal['RR'] = abs(entry - tp) / abs(entry - sl)
                deal['R1'] = abs(entry - sl)
                deal['end_date'] = self.data[DATE][candle]

            return deal

        def set_entry_values(deal):
            deal['entry_index'] = candle
            deal['entry_date'] = self.data[DATE][candle]

            return deal

        #Check margin call
        if self.tinh_tien:
            if self.margin_call:
                for idx, deal in enumerate(self.running_deals):
                    self.running_deals[idx]['margin_call'] = True

            if any(marker in str(self.data['dates'].iloc[candle]) for marker in self.time_add_fund):
                if self.current_fund<=0:
                    self.current_fund = 0

                if self.current_fund < self.start_fund * 0.75:
                    logger.info(f"---------------------------Them tien at {self.data['dates'].iloc[candle]}")
                    self.deposit(candle)

            if self.rut!=None and self.current_fund*self.rut[1] >= self.rut[0] and " 01:00" in str(self.data['dates'].iloc[candle]):
                self.rut_fund += round(self.current_fund*self.rut[1])
                self.current_fund -= round(self.current_fund*self.rut[1])

            if not self.margin_call:
                buys, sells = [deal for deal in self.running_deals if deal['order']=='buy'], [deal for deal in self.running_deals if deal['order']=='sell']
                if len(self.running_deals)>0:
                    total_buy_lots, total_sell_lots = sum(deal['lot'] for deal in buys), sum(deal['lot'] for deal in sells)
                    avg_buy_entry= sum(float(deal['entry']) * deal['lot'] for deal in buys) / total_buy_lots if total_buy_lots>0 else 0
                    avg_sell_entry = sum(float(deal['entry']) * deal['lot'] for deal in sells) / total_sell_lots if total_sell_lots>0 else 0

                    avg_sl = ((-self.current_fund/self._X)+total_buy_lots*avg_buy_entry - total_sell_lots*avg_sell_entry)/(total_buy_lots-total_sell_lots) if (total_buy_lots-total_sell_lots) != 0 else 0
                    logger.info(f'Avg sl: {avg_sl} when {total_buy_lots}, {total_sell_lots}, {avg_buy_entry}, {avg_sell_entry}, {self.current_fund}')
                    if self.data[LOW][candle]<= avg_sl <= self.data[HIGH][candle]:
                        logger.info(f'Touch margin call at candle {self.data[DATE][candle]}')
                        self.set_margin_call()


        _running_deals = []
        #Check các deals cũ có case end không
        for _idx, _deal in enumerate(self.running_deals):
            sl = _deal['sl']
            tp = _deal['tp']
            entry = _deal['entry']
            order = _deal['order']
            if 1==2 and self.force_close_time in self.data[DATE][candle]:
                logger.info('Force close running deals')
                _deal['R1'] = abs(entry - sl)
                _distance = (self.data[OPEN][candle]-_deal['entry'])
                if order=='buy':
                    _equity = round(_deal['lot']*(_distance/_deal['R1']), 2)
                elif order=='sell':
                    _equity = round(_deal['lot']*(-_distance/_deal['R1']), 2)

                if _equity > 0:
                    if self.tinh_tien:
                        _deal['reward'] = _equity*self._X

                    _deal = set_deal_res(_deal, 'Win')
                else:
                    if self.tinh_tien:
                        _deal['risk'] = _equity*self._X

                    _deal = set_deal_res(_deal, 'Lose')
                self.add_new_result(_deal)

            elif all(self.data[LOW][candle]<=X<=self.data[HIGH][candle] for X in [sl, tp]):
                logger.info('Cut sl and tp')
                if (order=='buy' and self.data[MAU_NEN][candle]<0) or (order=='sell' and self.data[MAU_NEN][candle]>0):
                    _deal = set_deal_res(_deal, 'Win')
                else:
                    _deal = set_deal_res(_deal, 'Lose')
                self.add_new_result(_deal)

            elif self.data[LOW][candle]<=tp<=self.data[HIGH][candle]:
                logger.info(f"deal: {_deal} hit tp at {self.data[DATE][candle]}")
                _deal = set_deal_res(_deal, 'Win')
                self.add_new_result(_deal)

            elif self.data[LOW][candle]<=sl<=self.data[HIGH][candle]:
                logger.info(f"deal: {_deal} hit sl at {self.data[DATE][candle]}")
                _deal = set_deal_res(_deal, 'Lose')
                self.add_new_result(_deal)

            elif min([self.data[OPEN][candle], entry]) <=sl<= max([self.data[OPEN][candle], entry]):
                logger.info(f"deal: {_deal} hit sl at {self.data[DATE][candle]} by a Gap")
                if self.tinh_tien:
                    _deal['R1'] = abs(entry - sl)
                    _distance = abs(self.data[OPEN][candle] - entry)
                    _equity = round(_deal['lot']*(_distance), 2)
                    _deal['risk'] = _equity*self._X
                    #round(lot * (abs(deal['entry'] - deal['sl']) * self._X), 2)
                    #print(f'{_distance}, {_equity}, {_deal["risk"]}')

                _deal = set_deal_res(_deal, 'Lose')
                self.add_new_result(_deal)

            elif min([self.data[OPEN][candle], entry]) <= tp <= max([self.data[OPEN][candle], entry]):
                logger.info(f"deal: {_deal} hit tp at {self.data[DATE][candle]} by a Gap")
                if self.tinh_tien:
                    _deal['R1'] = abs(entry - sl)
                    _distance = abs(self.data[OPEN][candle] - entry)
                    _equity = round(_deal['lot'] * (_distance), 2)
                    _deal['reward'] = _equity * self._X

                _deal = set_deal_res(_deal, 'Win')
                self.add_new_result(_deal)

            else:
                _running_deals.append(_deal)

        ended_deals = len(self.running_deals)-len(_running_deals)
        """
        Các deal end trong candle này thì vẫn phải tính vào số lượng max running
        """
        if max_running_deal:
            max_running_deal = max_running_deal-ended_deals
        #logger.info(f'Now running deals: {len(_running_deals)}')
        #Check các deal pending xem có case nào entry không > lây list các deal
        entry_list, _pending_list = [], []
        for _idx, _deal in enumerate(self.pending_deals):
            entry = _deal['entry']
            sl = _deal['sl']
            tp = _deal['tp']
            order = _deal['order']
            start_index = _deal['start_index']
            if candle - start_index > self.expired:#Hết hạn
                _deal = set_entry_values(_deal)
                _deal = set_deal_res(_deal, 'No res')
                self.add_new_result(_deal)

            elif self.data[LOW][candle]<=entry<=self.data[HIGH][candle]:
                logger.info(f'{_deal} touched entry')
                if len(_running_deals)>=max_running_deal:
                    logger.info('In running deal is reached limited')
                    _deal = set_entry_values(_deal)
                    _deal = set_deal_res(_deal, 'No res')
                    self.add_new_result(_deal)
                    continue

                if all(self.data[LOW][candle]<=X<=self.data[HIGH][candle] for X in [sl, tp]):
                    logger.info('Hit both of sl and tp')
                    _deal = set_entry_values(_deal)
                    _deal = set_deal_res(_deal, 'No res')
                    self.add_new_result(_deal)

                elif (self.data[LOW][candle]<=tp<=self.data[HIGH][candle]
                      and ((order=='buy' and self.data[MAU_NEN][candle]>0)
                      or (order=='sell' and self.data[MAU_NEN][candle]<0))):
                    logger.info(f'{_deal} touched tp')
                    _deal = set_entry_values(_deal)
                    _deal = set_deal_res(_deal, 'Win')
                    self.add_new_result(_deal)

                elif (self.data[LOW][candle]<=sl<=self.data[HIGH][candle]
                      and ((order == 'buy' and self.data[MAU_NEN][candle] < 0)
                      or (order == 'sell' and self.data[MAU_NEN][candle] > 0))):
                    logger.info(f'{_deal} touched sl')
                    _deal = set_entry_values(_deal)
                    _deal = set_deal_res(_deal, 'Lose')
                    self.add_new_result(_deal)

                else:
                    self.pending_deals[_idx]['entry_index'] = candle
                    self.pending_deals[_idx]['entry_date'] = self.data[DATE][candle]
                    entry_list.append(_deal)

            else:
                _pending_list.append(_deal)

        logger.info(f'Got {len(entry_list)} deals started')
        self.pending_deals = _pending_list
        if max_running_deal:
            if len(entry_list)+len(_running_deals) <= max_running_deal:
                self.running_deals = _running_deals + entry_list

            elif len(_running_deals)>=max_running_deal:
                logger.info('Reached maximum running deals')
                for _deal in entry_list:
                    _deal = set_entry_values(_deal)
                    _deal = set_deal_res(_deal, 'No res')
                    self.add_new_result(_deal)

            else:
                if len(entry_list)>1:
                    logger.info(f'Reach running deals limit: {len(_running_deals)} and {len(entry_list)}')
                    entry_price_list = [float(_x['entry']) for _x in entry_list]
                    entry_price_list.sort()
                    con_lai = max_running_deal - len(_running_deals)
                    apply_max_list = []
                    # buys, sells = ([deal for deal in entry_list  if deal['order']=='buy'],
                    #                [deal for deal in entry_list if deal['order']=='sell' ])
                    if self.data[OPEN][candle] <= min(entry_price_list):#cắt tử dưới lên
                        for _idx, _start_deal in enumerate(entry_list):
                            entry = _start_deal['entry']
                            if entry <= entry_price_list[con_lai-1]:
                                apply_max_list.append(_start_deal)

                    elif self.data[OPEN][candle] >= max(entry_price_list):
                        for _idx, _start_deal in enumerate(entry_list):
                            entry = _start_deal['entry']
                            if entry>=entry_price_list[-con_lai]:
                                apply_max_list.append(_start_deal)

                    else:
                        logger.info('Case Open nằm giữa các started deals')
                        if self.data[MAU_NEN][candle]>0:
                            for _idx, _start_deal in enumerate(entry_list):#Lấy các deal có entry lớn hơn
                                entry = _start_deal['entry']
                                if entry >= entry_price_list[-con_lai]:
                                    apply_max_list.append(_start_deal)
                        else:
                            for _idx, _start_deal in enumerate(entry_list):
                                entry = _start_deal['entry']
                                if entry <= entry_price_list[con_lai - 1]:
                                    apply_max_list.append(_start_deal)
                    for _deal in entry_list:
                        if _deal not in apply_max_list:
                            _deal = set_deal_res(_deal, 'No res')
                            self.add_new_result(_deal)

                    self.running_deals = _running_deals + apply_max_list
                else:
                    self.running_deals = _running_deals + entry_list
        else:
            self.running_deals = _running_deals + entry_list

        if self.tinh_tien:
            self.RES_TINH_TIEN['Fund'].iloc[candle] = self.current_fund
            self.RES_TINH_TIEN['Add'].iloc[candle] = self.add_fund
            self.RES_TINH_TIEN['Rut'].iloc[candle] = self.rut_fund