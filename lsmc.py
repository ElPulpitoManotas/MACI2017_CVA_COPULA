## @Date: November 12, 2016

import pandas as pd
import numpy as np


## ----------------------------------------------------------------------------
def LSMC_american(
            mc, 
            T,
            r = 0.06,
            strike = 1.10,
            degree = 2,
            option = 'call'):
    """
    Input: a pandas.DataFrame with the Monte Carlo simulation n_paths
    Output: the Option cash flow matrix, and the american option path
    """

    n_paths = len(mc)
    n_T = len(mc.columns)
    dt = float(T)/float(n_T-1.)

    positive_exposure = pd.DataFrame(0, index=mc.index, columns=mc.columns)
    negative_exposure = pd.DataFrame(0, index=mc.index, columns=mc.columns)
    ## 
    ## This matrix is overriden in each iteration
    cashflows = pd.DataFrame(0, index=mc.index, columns=mc.columns)
    ## State vector with either the backward discounted cashflows
    ## or the new cashflow in cashflow
    cashflows_state_vector = pd.Series(0, index=mc.index)

    ## Discount factor
    df = np.exp(-r*dt)

    for i in range(n_T-1):

        t = n_T - i - 1

        ## Price of immediate exercise a time 't' for american call
        if option == 'call':
            exercise = np.maximum(mc[t] - strike, 0)
        elif option == 'put':
            exercise = np.maximum(strike - mc[t], 0)

        ## Is in-the-money if exercise price is positive
        ITM = (exercise > 0)

        if t == n_T-1:
            ## First step in backward iteration, there is no continuation

            ## Value of continuation is 0 at maturity
            continuation = pd.Series(0, index=exercise.index)
            ## Keep only in the money paths
            mask = ITM[ITM].index

        elif t > 0:
            # X: stock prices at time t
            # Y: cashflows discounted one step back
            X = mc[t]
            Y = cashflows_state_vector * df

            if any(ITM):
                # Use a polynomial fit; use only paths that are in the money
                p = np.polyfit(X.loc[ITM], Y.loc[ITM], degree)

                # "The value of continuation is given by substituting X into
                # the conditional expectation function"
                continuation = pd.Series(np.polyval(p, X.values))

            else:
                # What to do when there is no enough data to fit
                continuation = pd.Series(0, index=X.index)

            # Paths that are early exercized
            mask = (exercise.loc[ITM] > continuation.loc[ITM])
            mask = mask[mask].index

            ## The exposures
            exposure = np.maximum(exercise, continuation)
            positive_exposure[t] = np.maximum(exposure, 0)
            negative_exposure[t] = np.minimum(exposure, 0)

        ## Cashflows
        cashflows.loc[:, t] = 0
        cashflows.loc[mask, t] = exercise[mask]
        ## Override future cashflows, if it's early exercized
        if (t+1 < n_T):
            columns = list(range(t+1, n_T))
            cashflows.loc[mask, columns] = 0
            ## If there is a cashflow, then the option worths zero afterwards, and consequently no exposure
            positive_exposure.loc[mask, columns] = 0
            negative_exposure.loc[mask, columns] = 0

        cond = (cashflows[t]==0)
        cashflows_state_vector.loc[~(cond)] = cashflows.loc[~(cond), t]
        if (t+1 < n_T):
            cashflows_state_vector.loc[cond] = cashflows_state_vector.loc[cond]*df

    ## Calculate the price
    discounted_cashflows = cashflows.copy(deep=True)
    for i in range(len(cashflows.columns)):
        ## Discounted cashflows, each column is a different time
        discounted_cashflows[i] = cashflows[i]*np.exp(-i*dt*r)

    price = discounted_cashflows.sum().sum() / n_paths

    return price, cashflows, positive_exposure, negative_exposure



## ----------------------------------------------------------------------------
if __name__ == "__main__":


    #mc = [[60,  66.480219,  75.272031,  77.973084],
          #[60,  63.236123,  61.303759,  63.153156],
          #[60,  58.505768,  53.213640,  54.468330],
          #[60,  61.645109,  63.313288,  65.384128],
          #[60,  60.703218,  60.942227,  55.215432],
          #[60,  60.608018,  54.212286,  50.437040],
          #[60,  57.444254,  54.228036,  57.294473],
          #[60,  55.897615,  55.311930,  59.866106]]
    #mc = pd.DataFrame(mc)
    #T = 0.5

    #price, \
        #cashflows, \
        #positive_exposure, \
        #negative_exposure = LSMC_american(mc, T, strike=62, option='put')
    #print(price)
    #print(cashflows)


    mc = [[1.00, 1.09, 1.08, 1.34],
          [1.00, 1.16, 1.26, 1.54],
          [1.00, 1.22, 1.07, 1.03],
          [1.00, 0.93, 0.97, 0.92],
          [1.00, 1.11, 1.56, 1.52],
          [1.00, 0.76, 0.77, 0.90],
          [1.00, 0.92, 0.84, 1.01],
          [1.00, 0.88, 1.22, 1.34]]
    mc = pd.DataFrame(mc)
    T = 3.

    price, \
        cashflows, \
        positive_exposure, \
        negative_exposure = LSMC_american(mc, T, strike=1.10, option='put')
    print(price)
    print(cashflows)

    #price, \
        #cashflows, \
        #positive_exposure, \
        #negative_exposure = LSMC_american(mc, T, r=0.02, strike=.10, option='call')
    #print(price)
    #print(cashflows)
    #print(positive_exposure)
    #print(negative_exposure)

## Decision incluyendo PD a cada tiempo

