import numpy as np
import scipy as sp
import random, time
import pandas as pd
from datetime import datetime
import postcodes_io_api

# not a good idea for 1 mil entries
searches = "data/search.csv"
search_ls = pd.read_csv(searches, sep=',', encoding='utf-8')
print(search_ls.head(5))
#filter the unique comb (search of 1 dropped)
search_ls['count']= search_ls.groupby(['lat','lng']).count()
print(search_ls.head(200))

import sys
sys.exit()
#search_lst.to_csv('data/search_with_postcodes.csv')

#lst = pd.read_csv('data/search_with_postcodes.csv', names = ['lat', 'lng', 'count'])
#ls = lst[lst['count'] > 1]
#print(ls)
#ls.to_csv('data/uniques_postcodes_in_search.csv')

# use API call (heavy..) to retrieve the relevant postcodes. some should not be there
#search_ls = pd.read_csv('data/uniques_postcodes_in_search.csv')

# api  = postcodes_io_api.Api(debug_http=True)
#
# for idx, row in search_ls.iterrows():
#
#     post = api.get_nearest_postcodes_for_coordinates(latitude=row['lat'], longitude=row['lng'])
#    # print(type(post))
#    # print(post)
#     try:
#         get = post['result'][0]['postcode']
#         search_ls[idx, 'postcode'] = get
#     except Exception:
#         print(Exception)
#         pass

search_ls.to_csv('data/uniques_postcodes_in_search.csv')