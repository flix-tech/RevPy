from revpy.optimizers import calc_EMSRb
from revpy.fare_transformation import fare_trafo_decorator


@fare_trafo_decorator
def calc_EMSRb_MR(fares, demands, sigmas=None, cap=None):
    return calc_EMSRb(fares, demands, sigmas)
