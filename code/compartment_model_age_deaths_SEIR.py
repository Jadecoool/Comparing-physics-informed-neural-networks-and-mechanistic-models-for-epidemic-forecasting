from numba import jit
import pandas as pd
import numpy as np
from typing import List
from utils import get_beta, apply_seasonality, get_gamma_params
from datetime import datetime


def SEIR(inf_t0: int,
         rec_t0: int,
         Nk: List[float],
         T: int,
         R0: float,
         eps: float,
         mu: float,
         ifr: List[float],
         Delta: float,
         C: List[List[float]],
         detection_rate: float,
         dates: List[datetime],
         hemisphere: int,
         seasonality_min: float,
         deaths_delay: str = "fixed",
         daily_steps: float = 12,
         Delta_std: float = 1,
         seed=None) -> dict:
    """
    Plain age-structured SEIR -- four compartments, no behavioural reaction.

    This is the null model for compartment_model_age_deaths.py: same seeding, same
    seasonality, same death-delay handling, but the S_B compartment is gone
    entirely, so susceptibles only ever move to E. With nothing to suppress
    transmission the epidemic turns over purely through susceptible depletion and
    R0 is constant throughout.

    The signature is deliberately UNCHANGED so the two models can be swapped
    without touching the caller. These arguments are accepted and IGNORED:

        beta_B, mu_B, r, gamma, behavioral_mechanism

    The returned dict has no 'SB' key -- that compartment no longer exists.

    Parameters
    ----------
        @param inf_t0 (int): initial number of infected
        @param rec_t0 (int): initial number of recovered
        @param Nk (List[float]): number of individuals in different age groups
        @param T (int): simulation steps
        @param R0 (float): basic reproductive number
        @param eps (float): inverse of latent period
        @param mu (float): inverse of infectious period
        @param beta_B (float): IGNORED (behavioural model only)
        @param mu_B (float): IGNORED (behavioural model only)
        @param r (float): IGNORED (behavioural model only)
        @param ifr (List[float]): infection fatality rate by age groups
        @param Delta (float): delay in deaths (mean)
        @param C (List[List[float]]): contact matrix
        @param detection_rate (float): fraction of deaths that are reported
        @param dates (List[datetime]): list of simulation dates
        @param hemisphere (int): hemisphere (0: north, 1: tropical, 2: south)
        @param seasonality_min (float): seasonality parameter
        @param gamma (float): IGNORED (behavioural model only)
        @param deaths_delay (str): method to calculate deaths delay. Defaults to "fixed" (alternative is "gamma")
        @param daily_steps (int): simulation time step. Defaults to 12
        @param Delta_std (float): delay in deaths (std). Defaults to 1
        @param behavioral_mechanism (str): IGNORED (behavioural model only)
        @param seed (int): random seed
    Return
    ------
        @return: dictionary of compartments and deaths
    """

    if seed is not None:
        np.random.seed(int(seed))

    # number of age groups
    n_age = len(Nk)

    # compute beta
    beta = get_beta(R0, mu, Nk, C)

    # compute deaths delay parameters
    if deaths_delay == "gamma":
        shape, scale = get_gamma_params(Delta, Delta_std)
    else:
        shape, scale = None, None

    # initialize compartments and set initial conditions (S: 0, E: 1, I: 2, R: 3)
    compartments, deaths = np.zeros((4, n_age, T)), np.zeros((n_age, T))
    daily_new_I = np.zeros((n_age, T))
    # S -> E per day and age. This is the INCIDENCE OF INFECTION, which is not
    # the same thing as daily_new_I (E -> I, a latent period later) nor as
    # deaths (later still, and scaled by the IFR).
    infections = np.zeros((n_age, T))

    # compute delta t
    dt = 1. / daily_steps

    # distribute intial infected and recovered among age groups
    for age in range(n_age):
        # I
        inf_t0_age = int(inf_t0 * Nk[age] / np.sum(Nk))
        compartments[2, age, 0] = int(int(inf_t0_age) * (1 / mu) / ((1 / mu) + (1 / eps)))
        # E
        compartments[1, age, 0] = inf_t0_age - compartments[2, age, 0]
        # R
        compartments[3, age, 0] = int(rec_t0 * Nk[age] / np.sum(Nk))
        # S
        compartments[0, age, 0] = Nk[age] - (
                    compartments[1, age, 0] + compartments[2, age, 0] + compartments[3, age, 0])

    # simulate
    for t in np.arange(1, T, 1):

        # get seasonal forcing
        st = apply_seasonality(day=dates[t - 1], seasonality_min=seasonality_min, basin_hemispheres=hemisphere)

        # iterate over daily steps
        compartments_next_day = compartments[:, :, t - 1].copy()
        new_R_day = np.zeros(n_age, dtype=int)
        new_I_day = np.zeros(n_age, dtype=int)
        new_E_day = np.zeros(n_age, dtype=int)
        for _ in range(int(daily_steps)):
            # S -> E is the only way out of S
            prob_S_to_I = np.sum(st * beta * C * compartments_next_day[2, :] / Nk, axis=1)
            new_E = np.random.binomial(compartments_next_day[0, :].astype(int),
                                       1 - np.exp(-prob_S_to_I * dt))

            new_I = np.random.binomial(compartments_next_day[1, :].astype(int), 1 - np.exp(-eps * dt))
            new_R = np.random.binomial(compartments_next_day[2, :].astype(int), 1 - np.exp(-mu * dt))

            # update compartments_next_day
            # S
            compartments_next_day[0, :] = compartments_next_day[0, :] - new_E
            # E
            compartments_next_day[1, :] = compartments_next_day[1, :] + new_E - new_I
            # I
            compartments_next_day[2, :] = compartments_next_day[2, :] + new_I - new_R
            # R
            compartments_next_day[3, :] = compartments_next_day[3, :] + new_R

            # store new_R for deaths computation
            new_R_day += new_R
            new_I_day += new_I
            new_E_day += new_E

        # update compartments
        compartments[:, :, t] = compartments_next_day
        daily_new_I[:, t] = new_I_day
        # Index t-1, not t: these are the people who left S BETWEEN t-1 and t,
        # so infections[:, a:b].sum() telescopes to S[a] - S[b] exactly, and a
        # weekly block sums to the drop in S over that week.
        infections[:, t - 1] += new_E_day

        # compute deaths
        if deaths_delay == "fixed":
            if (t - 1) + Delta < deaths.shape[1]:
                deaths[:, (t - 1) + int(Delta)] = (detection_rate * np.random.binomial(new_R_day, ifr)).astype(int)

        elif deaths_delay == "gamma":
            new_deaths = np.random.binomial(new_R_day, ifr)
            delays = np.random.gamma(shape=shape, scale=scale, size=np.sum(new_deaths))
            count = 0
            for i, d in enumerate(new_deaths):
                for _ in range(d):
                    # sample delay
                    delay = int(delays[count])
                    if t + delay < deaths.shape[1]:
                        deaths[i, t + delay] += 1
                    count += 1

    # apply deaths undereporting
    deaths_sum = (deaths.sum(axis=0)).astype(int)

    return {'compartments': compartments,
            'S': compartments[0].sum(axis=0),
            'E': compartments[1].sum(axis=0),
            'I': compartments[2].sum(axis=0),
            'R': compartments[3].sum(axis=0),
            'deaths_age': deaths,
            'infections_age': infections,
            'infections': infections.sum(axis=0),      # S -> E per day, all ages
            # 'deaths_actual': ((1. / detection_rate) * deaths.sum(axis=0)).astype(int),
            'deaths': deaths_sum,
            'new_I': daily_new_I.sum(axis=0)}
