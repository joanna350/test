import numpy as np
import pandas as pd
import random, math
from collections import defaultdict
import postcodes_io_api

#randomize strategy at the onset (doesnt affect)
#NE: choose another strategy until change of 1 player does not benefit themselves

#we can use the following formula from prediction to incorporate 1.7 mil postcodes' centres
#utility: price (of a loc)
# w = [[-2.96098207e-01],
#  [-3.30688736e+00],
#  [-1.44171662e+00],
#  [-4.47074471e-04],
#  [-2.03437357e+01],
#  [-3.84492187e+01],
#  [-6.15046051e+00],
#  [-4.21910602e+00],
#  [ 6.40737716e+00]]
#
# b = [39.69286281]

# x = cos(lat) * cos(lon)
# y = cos(lat) * sin(lon),
# z = sin(lat)

# orders = "data/orders.csv"
# order_ls = pd.read_csv(orders, sep=',', encoding='utf-8')

#dont really need these
# order_ls['x'] = np.cos(order_ls['lat']) * np.cos(order_ls['lng'])
# order_ls['y'] = np.cos(order_ls['lat']) * np.sin(order_ls['lng'])
# order_ls['z'] = np.sin(order_ls['lat'])

#preprocessed data and added util column for convenience
# tcond = order_ls['order_start_day'] >= 24
# focond = order_ls['order_start_day'] <= 31
# game = order_ls[tcond & focond]
#
# for idx, row in game.iterrows():
#     hr = float(row['order_end_time'][:2])
#     min = float(row['order_end_time'][3:5])
#     sec = float(row['order_end_time'][6::])
#
#     end_time = hr + min / 60 + sec / 3600
#
#     hr = float(row['order_start_time'][:2])
#     min = float(row['order_start_time'][3:5])
#     sec = float(row['order_start_time'][6::])
#     start_time = hr + min / 60 + sec / 3600
#
#     if row['order_end_day'] >= row['order_start_day']:
#         util = row['hourly_price'] * ((row['order_end_day'] - row['order_start_day']) * 24 + end_time - start_time)
#     else:
#         util = row['hourly_price'] * ((31 - row['order_start_day'] + row['order_end_day']) * 24 + end_time - start_time)
#
#     util = math.floor(util)
#
#     game.loc[idx, 'util'] = util
#     print(game.loc[idx])
#
# game = game.drop(columns=['booking_day','booking_time'])
# #
# print(game.head(5))
# print(game.tail(5))
# game.to_csv('data/lstweek_order_util.csv')
#print(game.groupby(['lat','lng']).size().rename(columns='coords')) #2434
#print(game.groupby(['postal_code']).size().rename(columns='coords')) #2068 - ok so its not just the centre
#print(game) #13837

# but to strategize among the existing postcodes which would be << 1.7 mil and save memory error
# 10 player - NE in congestion game (it will be fight for a slot, not a customer nec.)
# they offer the pool to choose from with the assumption
# that a selection of 20 suffice for ~2000 given unique centres
# ..By Rosenthal theorem of 1973. at least one pure Nash equilibrium is guaranteed..

game = pd.read_csv('data/lstweek_order_util.csv')
game = game.drop(columns=['Unnamed: 0'])
game = game.rename(columns={"order_start_day": "rent_start_day", "order_start_time": "rent_start_time",
                     "order_end_day": "rent_end_day", "order_end_time": "rent_end_time"})

game = game.sort_values(by='util', ascending=False)
#game = game.head(500)

#api  = postcodes_io_api.Api(debug_http=True)


def util_(pl, ps_dict):
   utils = 0
   for p_id, p_row in pl[0].iterrows():
       ps = p_row['postal_code']
       if ps_dict[ps] > 1:
           utils += p_row['util']/ps_dict[ps]
       utils += p_row['util']
   return utils

def rm_ps(pl, ps_dict):
    for ps in pl[1]:
        ps_dict[ps] -= 1
    return ps_dict

def add_ps(pl, ps_dict):
    for idx, plc in pl[0].iterrows():
        ps_dict[plc['postal_code']] += 1
    return ps_dict

#print(len(game.head(500)['postal_code'].unique())
#      )
#import sys
#sys.exit()

def equal_test(pls):

    tr = [0] * 9
    for i in range(9):
        if pls['pl'+ str(i)][0].equals(pls['pl' + str(i+1)][0]):
            tr[i] = 1

    if tr == [True] * 9:
        return True
    else:
        return False


pls = {}
ps_dict = defaultdict(lambda: 0)
# initialize
for i in range(10):
    # the choice of postcodes and the time frame..
    assign_pl = game.sample(560)
    ps_ls = assign_pl['postal_code'].to_numpy()
    pls["pl{0}".format(i)] = [assign_pl, ps_ls]
    for psc in ps_ls:
        ps_dict[psc] += 1
    print(assign_pl)

print('check the dict',sum(ps_dict.values()))

for i in range(0, 100000):
    deltas = np.ones(10) * np.inf
    delta = -1
    pl_i = 0
    for key, pl in pls.items():
       #pl has 2 itemse.. 0: pandas df, 1: list of postcodes uinque
    #order_start_day, order_start_time, order_end_day, order_end_time, hourly_price
    #every other pl's strategy mst b known to b considered in util
    #if postcode and rent time overlap, divide the hourly px  -> later
        while delta < 0:
            cur_u = util_(pl, ps_dict)

            for g_id, g_row in game.iterrows(): # 500 -> 14,000
                g_ps= g_row['postal_code']
                if ps_dict[g_ps] == 0:
                    game.loc[g_id, 'util_div'] = g_row['util']
                else:
                    game.loc[g_id, 'util_div'] = g_row['util']/(ps_dict[g_ps])

            for p_id, p_row in pl[0].iterrows(): # 20 -> 560
                pl[0].loc[p_id, 'util_div'] = p_row['util']/ps_dict[p_row['postal_code']]

            # greedy
            game = game.sort_values(by='util_div', ascending=False)
            pl[0] = pl[0].sort_values(by='util_div', ascending=False)

            cnt = 0
            for p_id, p_row in pl[0].iterrows():
                base = game.iloc[cnt]
                base_ps = base['postal_code']
                p_row_bs = p_row['postal_code']
                if base_ps == p_row_bs:
                    if base['util']/ps_dict[base_ps] > p_row['util']/ps_dict[p_row_bs]:
                        pl[0].loc[p_id] = base
                        cnt += 1
                else:
                    if base['util']/(ps_dict[base_ps] + 1) > p_row['util']/ps_dict[p_row_bs]:
                        ps_dict[p_row_bs] -= 1
                        pl[0].loc[p_id] = base
                        ps_dict[base_ps] += 1
                        cnt += 1
            #   else: yield
            #  if the entry to compare with was smaller, use it for the next

            game.drop(columns = 'util_div')
            pl[0].drop(columns = 'util_div')

            nex_u = util_(pl, ps_dict)
            delta = nex_u - cur_u

        pl[1] = pl[0]['postal_code'].to_numpy()
        print('interim', pl)
        deltas[pl_i] = delta
        pl_i += 1


    if (delta == 0 for delta in deltas) and equal_test(pls):
        print('NASH EQUILIBRIUm...')
        with open('result/NE.csv', 'a') as file:
            for pli, item in pls.items():
                print(item[0], '\n', item[0]['util'].sum())
                item[0].to_csv(file)
        break

    #if #nash equilibrium (no player benefits from moving) then break
