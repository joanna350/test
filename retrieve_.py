import pandas as pd
import numpy as np
import random, math
import postcodes_io_api


getthe = pd.read_csv('result/NE.csv', sep=',', encoding='utf-8')

api  = postcodes_io_api.Api(debug_http=True)

for idx, row in getthe.iterrows():

    post = api.get_nearest_postcodes_for_coordinates(latitude=row['lat'], longitude=row['lng'])
   # print(type(post))
   # print(post)
    try:
        get = post['result'][0]['postcode']
        search_ls[idx, 'postcode'] = get
    except Exception:
        print(Exception)
        pass