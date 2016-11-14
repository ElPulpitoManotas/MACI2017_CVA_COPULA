## @Date: November 12, 2016

import pandas as pd
import numpy as np


## ----------------------------------------------------------------------------
def LSMC_american(
            mc, 
            r = 0.06,
            strike = 1.10,
            degree = 2,
            option = 'put'):
    """
    Input: a pandas.DataFrame with the Monte Carlo simulation n_paths
    Output: the Option cash flow matrix, and the american option path
    """

    n_paths = len(mc)
    n_T = len(mc.columns)

    cashflows = pd.DataFrame(0, index=mc.index, columns=mc.columns)

    ## Discount factor
    df = np.exp(-r)

    for i in xrange(n_T):

        t = n_T - i - 1

        ## Price of immediate exercise a time 't' for american call
        if option == 'call':
            exercise = np.maximum(mc[t] - strike, 0)
        elif option == 'put':
            exercise = np.maximum(strike - mc[t], 0)

        ## Is in-the-money if exercise price is positive
        ITM = (exercise > 0)

        if i == 0:
            mask = ITM[ITM].index

        else:
            ## X: stock prices at time t, only if they are in the money
            ## Y: denote the discounted cashflows discounted one step back
            X = mc.loc[ITM, t]
            Y = cashflows.loc[X.index, t+1]*df
            p = np.polyfit(X, Y, degree)

            ## "The value of continuation is given by substituting X into
            ## the conditional expectation function"
            continuation = pd.Series(np.polyval(p, X), index=X.index)

            ## Paths that are early exeercized
            mask = (exercise.loc[ITM] > continuation)
            mask = mask[mask].index

        ## Cashflows
        ## Override future cashflows, if it's early exercized
        cashflows.loc[mask, t] = exercise[mask]
        for ti in xrange(t+1, n_T):
            cashflows.loc[mask, ti] = 0
        #print('cashflows at t={:d}'.format(t))
        #print(cashflows)

    ## Calculate the price
    discount_cashflows = cashflows.copy(deep=True)
    for i in xrange(len(cashflows.columns)):
        ## Discount factor
        df = np.exp(-i*r)
        discount_cashflows[i] = cashflows[i]*df

    price = discount_cashflows.sum().sum() / n_paths

    return cashflows, price



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

    cashflows, price = LSMC_american(mc)

    print(cashflows)
    print(price)

## Como calcular el exposure a partir de los cashflows?
## Decision incluyendo PD a cada tiempo

