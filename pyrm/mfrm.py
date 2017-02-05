import numpy as np
from scipy.linalg import solve


def estimate_host_level(observed, utilities, availability, market_share):
    """ Estimate demand, spill and recapture using multi-flight recapture
    method (MFRM)

    Parameters
    ----------
    observed: dict
        Observed demand for each product
    utilities: dict
        Utilities of products based on choice model parameters
    availability: dict
        Availability of demand open during period considered
    market_share: float
        Host's market share

    Returns
    -------
    tuple
        Estimated demand, spill and recapture for H
    """

    demand = spill = recapture = 0

    if observed and utilities and availability:
        probs, nofly_prob = selection_probs(utilities, market_share)

        # Probability of selecting an open element from market set M
        prob_market_open = nofly_prob + sum([pr * availability.get(p, 0)
                                             for p, pr in probs.items()])

        recapture_rate = (prob_market_open - nofly_prob) / prob_market_open

        # Probability of selecting a closed element from host set H
        prob_host_closed = (1 - prob_market_open) / (1 - nofly_prob)

        demand, spill, recapture = demand_mass_balance(sum(observed.values()),
                                                       prob_host_closed,
                                                       recapture_rate)
    return demand, spill, recapture


def selection_probs(utilities, market_share):
    """Customer selection probability for all products and 'do not fly'

    Parameters
    ----------
    utilities: dict
        Utilities of products based on choice model parameters
    market_share: float
        Host's market share

    Returns
    -------
    tuple
        Selection probs for each product and 'do not fly' prob
    """

    exp_sum = sum([np.exp(u) for u in utilities.values()])
    exp_nofly_utility = exp_sum * (1 - market_share) / market_share
    exp_sum += exp_nofly_utility

    # customer selection probability for all products
    probs = {p: (np.exp(u) / exp_sum) for p, u in utilities.items()}

    # customer selection probability for ‘do not fly’
    nofly_prob = exp_nofly_utility / exp_sum

    return probs, nofly_prob


def demand_mass_balance(observed_demand, close_prob, recapture_rate):
    """Solve Demand Mass Balance equation

    Parameters
    ----------
    observed_demand: int
        Observerd demand
    close_prob: float
        Probability of selecting a closed element from host set H
    recapture_rate: float
        Estimated recapture rate

    Returns
    -------
    tuple
        Estimated demand, spill and recapture
    """

    A = np.array([[1, -1, 1], [-close_prob, 1, 0], [0, -recapture_rate, 1]])
    B = np.array([observed_demand, 0, 0])

    demand, spill, recapture = solve(A, B)

    return demand, spill, recapture


if __name__ == '__main__':

    estimate({'product1': 10, 'product2': 6},
                                    {'product1': 0.3, 'product2': 1.3},
                                    {'product1': 0.9, 'product2': 0.65}, 0.9)