import numpy as np
import scipy as sp
import random, time
import pandas as pd
from datetime import datetime
import postcodes_io_api


searches = "data/search.csv"
search_ls = pd.read_csv(searches, sep=',', encoding='utf-8')

api  = postcodes_io_api.Api(debug_http=True)

for idx, row in search_ls.iterrows():

    post = api.get_nearest_postcodes_for_coordinates(latitude=row['lat'],
                                                                longitude=row['lng'])
   # print(type(post))
   # print(post)
    try:
        get = post['result'][0]['postcode']
        search_ls[idx, 'postcode'] = get
    except Exception:
        print(Exception)
        pass

search_ls.to_csv('data/search_with_postcodes.csv')

print(search_ls.postcode.value_counts())

print(search_ls)

