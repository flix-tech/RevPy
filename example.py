import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath('..'))

from PyRM import fare_transformation as ft
from PyRM import pyrm as rm
from PyRM.optimizers import EMSRb


# test fare transformation
capacity = 10
fares = np.array([ 69.5,  59.5,  48.5,  37.5,  29. ])
demands = np.array([3, 1, 0, 0, 10])
Q = demands.cumsum()
TR = Q*fares
ccs = list(range(len(fares)))

adjusted_fares, adjusted_demand, Q_eff, TR_eff, original_fares = \
    ft.fare_transformation(fares, demands, capacity, return_all=True)


# test protection level function
sigmas = np.zeros(demands.shape)  # deterministic demand
p = rm.protection_levels(fares, demands, sigmas, capacity, 'EMSRb_MR')
p
data = np.vstack((fares, demands, Q, TR, p))
col_names = ['fare', 'demand',  'Q', 'TR', 'protection_level']

res = pd.DataFrame(data.transpose(), columns=col_names)

# iterate through all possible capacities (remaining seats)
lowest_ccs = []
for cap in range(1, 40):
    p = rm.protection_levels(fares, demands, sigmas, 'EMSRb_MR', cap)
    lowest_cc = max(np.where(pd.notnull(p))[0])
    lowest_ccs.append(lowest_cc)



cc_allocation = np.zeros(len(ccs))
for cc in ccs:
    cc_allocation[cc] = lowest_ccs.count(cc)

cc_allocation



# test esmrb
EMSRb(fares, demands, np.ones(demands.shape)*0.001)

# test efficient strategies
from PyRM.fare_transformation import efficient_strategies
efficient_strategies(Q, TR, fares)


# test stepwise
p = rm.protection_levels(fares, demands, sigmas, capacity, 'EMSRb_MR')
p

p2 = rm.protection_levels(fares, demands, sigmas, capacity, 'EMSRb_MR_step')
p2
