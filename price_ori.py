import numpy as np
from scipy.optimize import minimize
import random, time
import pandas as pd
from datetime import datetime
import postcodes_io_api


def get_linreg_pred(X, w, b):
    pred = np.dot(X, w) + b
    return pred

def fit_lin_regression(X, y, alpha):

    X = np.hstack([X, np.ones((X.shape[0], 1))])

    regularI = np.sqrt(alpha) * np.identity(X.shape[1], int)

    # skip the last row so as not to regularize bias
    reg_A = np.vstack([X, regularI[:-1]])

    y = np.vstack((y, np.zeros(X.shape[1]-1)[:, None]))

    w = np.linalg.lstsq(reg_A, y, rcond=1)[0]

    return w[:-1], w[-1]


def linreg_cost(params, X, y, alpha):
    w, b = params
    f = np.dot(X, w) + b
    res = f - y
    E = np.dot(res, res) + alpha * np.dot(w, w)

    f_b = 2*res
    b_b = np.sum(f_b)
    w_b = np.dot(X.T, f_b) + 2 * alpha * w

    return E, [w_b, b_b]

def fit_linreg_gradopt(X, y, alpha):

    D = X.shape[1]
    args = (X, y, alpha)
    init = (np.zeros(D), np.array(0))
    w, b = minim_ls(linreg_cost, init, args)
    return w, b

def minim_ls(cost, init_ls, args):
    opt = {'maxiter': 500, 'disp': False}
    init, unwrap = params_wrap(init_ls)
    def wrap_cost(vec, *args):
        E, params_b = cost(unwrap(vec), *args)
        vec_b, _ = params_wrap(params_b)
        return E, vec_b

    res = minimize(wrap_cost, init, args, 'L-BFGS-B', jac=True, options=opt)
    return unwrap(res.x)

def params_unwrap(param_v, shapes, sizes):
    args = []
    pos = 0
    for i in range(len(shapes)):
        sz = sizes[i]
        args.append(param_v[pos:pos+sz].reshape(shapes[i]))
        pos += sz
    return args

def params_wrap(param_ls):
    param_lst = [np.array(x) for x in param_ls]
    shapes = [x.shape for x in param_ls]
    sizes = [x.size for x in param_ls]
    param_v = np.zeros(sum(sizes))
    pos = 0
    for param in param_ls:
        sz = param.size
        param_v[pos:pos+sz] = param.ravel()
        pos += sz
    unwrap = lambda pvec: params_unwrap(pvec,shapes, sizes)
    return param_v, unwrap

def RMSE(pred, true):
    pred = pred.reshape(-1, 1)
    k = (pred- true)**2
    E = np.sqrt(k.mean())

    return E

def run(X_train, X_test, y_train, y_test):
    toc = time.time()
    print('training------')
    w, b = fit_lin_regression(X_train, y_train, 10)
    pred = get_linreg_pred(X_train, w, b)
    mse_t = RMSE(pred, y_train)

    #reusing the param..
    pred = get_linreg_pred(X_test, w, b)
    mse_v = RMSE(pred, y_test)
    print('RMSE for training vs. test set', mse_t, mse_v)

    tic = time.time()
    print('time elapsed: ', round(tic-toc, 3))

    toc = time.time()
    print('train for gradopt---')
    w, b= fit_linreg_gradopt(X_train, y_train, 10)
    mse_t = RMSE(X_train, y_train, w, b)
    mse_v = RMSE(X_test, y_test, w, b)
    print('RMSE for train vs. test set' , mse_t, mse_v)
    tic = time.time()
    print('time passed: ', round(tic-toc, 3))


def prediction_(order_ls):

    fcond = order_ls['order_start_day'] >= 1
    scond = order_ls['order_start_day'] <= 21
    train_chunk = order_ls[fcond & scond]

    tcond = order_ls['order_start_day'] >= 22
    focond = order_ls['order_start_day'] <= 31
    test_chunk = order_ls[tcond & focond]

    X_train = train_chunk[['x','y','z']]
    y_train = train_chunk[['hourly_price']]
    
    X_test = test_chunk[['x','y','z']]
    y_test = test_chunk[['hourly_price']]

    X_train= X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train= y_train.to_numpy()
    y_test = y_test.to_numpy()

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    run(X_train, X_test, y_train, y_test)


def preprocess():
    orders = "data/orders.csv"
    order_ls = pd.read_csv(orders, sep=',', encoding='utf-8')
    order_ls['book_req'] = order_ls['booking_day'].astype(str) + ' ' + order_ls['booking_time'].astype(str)
    order_ls['order_start'] = order_ls['order_start_day'].astype(str) + ' ' + order_ls['order_start_time'].astype(str)
    order_ls['order_end'] = order_ls['order_end_day'].astype(str) + ' ' + order_ls['order_end_time'].astype(str)
    order_ls = order_ls.drop(columns = ['booking_day','booking_time','order_start_time','order_end_day','order_end_time'])

    # linear regression
    # input: 'lat' ' lng'
    # x = cos(lat) * cos(lon)
    # y = cos(lat) * sin(lon),
    # z = sin(lat)
    # output: 'hourly_price'

    print(order_ls)

    order_ls['x'] = np.cos(order_ls['lat']) * np.cos(order_ls['lng'])
    order_ls['y'] = np.cos(order_ls['lat']) * np.sin(order_ls['lng'])
    order_ls['z'] = np.sin(order_ls['lat'])

    return order_ls


if __name__ == '__main__':
    order_ls = preprocess()

    prediction_(order_ls)

'''
print('search list')

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

'''
