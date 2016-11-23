## @Date: November 12, 2016

import pandas as pd
import numpy as np


## ----------------------------------------------------------------------------
def LSMC_american(
            mc, 
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

    positive_exposure = pd.DataFrame(0, index=mc.index, columns=mc.columns)
    negative_exposure = pd.DataFrame(0, index=mc.index, columns=mc.columns)
    cashflows = pd.DataFrame(0, index=mc.index, columns=mc.columns)

    ## Discount factor
    df = np.exp(-r)

    for i in range(n_T):

        t = n_T - i - 1

        ## Price of immediate exercise a time 't' for american call
        if option == 'call':
            exercise = np.maximum(mc[t] - strike, 0)
        elif option == 'put':
            exercise = np.maximum(strike - mc[t], 0)

        ## Is in-the-money if exercise price is positive
        ITM = (exercise > 0)

        if i == 0:
            ## Value of continuation is 0 at maturity
            continuation = pd.Series(0, index=exercise.index)
            ## Keep only in the money paths
            mask = ITM[ITM].index

        else:

            ## X: stock prices at time t
            ## Y: denote the discounted cashflows discounted one step back
            X = mc[t]
            Y = cashflows[t+1] * df

            if any(ITM):
                ## Use a polynomial fit
                ## Use only paths that are in the money
                p = np.polyfit(X.loc[ITM], Y.loc[ITM], degree)

                ## "The value of continuation is given by substituting X into
                ## the conditional expectation function"
                continuation = pd.Series(np.polyval(p, X))
                #print(continuation)

            else:
                ## What to do when there are no enough data to fit
                continuation = pd.Series(0, index=X.index)

            ## Paths that are early exercized
            mask = (exercise.loc[ITM] > continuation.loc[ITM])
            mask = mask[mask].index


        ## The exposures
        exposure = np.maximum(exercise, continuation)
        positive_exposure[t] = np.maximum(exposure, 0)
        negative_exposure[t] = np.minimum(exposure, 0)

        ## Cashflows
        ## Override future cashflows, if it's early exercized
        cashflows.loc[mask, t] = exercise[mask]
        for ti in range(t+1, n_T):
            cashflows.loc[mask, ti] = 0
            ## If there is a cashflow, then the option worths zero afterwards
            ## Therefore, no exposure
            positive_exposure.loc[mask, ti] = 0
            negative_exposure.loc[mask, ti] = 0

        #print('cashflows at t={:d}'.format(t))
        #print(cashflows)

    ## Calculate the price
    discounted_cashflows = cashflows.copy(deep=True)
    for i in range(len(cashflows.columns)):
        ## Discount factor
        df = np.exp(-i*r)
        discounted_cashflows[i] = cashflows[i]*df

    price = discounted_cashflows.sum().sum() / n_paths

    return price, cashflows, positive_exposure, negative_exposure



## ----------------------------------------------------------------------------
if __name__ == "__main__":

    mc = [[1.00, 1.09, 1.08, 1.34],
          [1.00, 1.16, 1.26, 1.54],
          [1.00, 1.22, 1.07, 1.03],
          [1.00, 0.93, 0.97, 0.92],
          [1.00, 1.11, 1.56, 1.52],
          [1.00, 0.76, 0.77, 0.90],
          [1.00, 0.92, 0.84, 1.01],
          [1.00, 0.88, 1.22, 1.34]]
    mc = pd.DataFrame(mc)

    price, cashflows, positive_exposure, negative_exposure = LSMC_american(mc)

    print(price)
    print(cashflows)
    #print(positive_exposure)
    #print(negative_exposure)

## Decision incluyendo PD a cada tiempo

