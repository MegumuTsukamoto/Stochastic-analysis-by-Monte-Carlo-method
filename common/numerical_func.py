import math
import numpy as np
from scipy import stats

# バイナリーサーチ
# 微分係数が求まる場合は、Newton法の方が早い
# f(a)f(b)<0の場合、f(x)=0の解はa<x<bに存在する
# 解の探索範囲を半分に分割していく

def binary_search(a, b, F, s_lower, s_upper, eps):
    c = 10000 # 計算上限
    not_finish = True
    count = 0

    while not_finish and (count <= c):
        bin = (s_lower + s_upper) / 2
        y = F(bin)
        if np.abs(y) <= eps:
            x = bin # 解発見
            not_finish = False
        elif F(s_lower)*y < 0:
            s_upper = bin
        else:
            s_lower = bin
        count = count + 1
    if not_finish:
        x = "Nan"
    return x

