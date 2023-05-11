import numpy as np
import datetime as dt
import scipy.optimize as sc
import pandas as pd
import plotly.graph_objects as go
from pandas_datareader import data as pdr


stockList = []
tradingDays = 0


def GetData(stocks, start, end):
    stockData = pdr.get_data_yahoo(stocks, start=start, end=end)
    stockData = stockData['Close']

    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()

    return meanReturns, covMatrix


def portfolioPerformance(weights, meanReturns, covMatrix):
    returns = np.sum(meanReturns * weights) * tradingDays
    std = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights))) * np.sqrt(tradingDays)
    return returns, std


def negativeSR(weights, meanReturns, covMatrix, riskFreeRate=0):
    pReturns, pStd = portfolioPerformance(weights, meanReturns, covMatrix)
    return -(pReturns - riskFreeRate) / pStd


def maxSR(meanReturns, covMatrix, riskFreeRate=0, constrainSet=(0, 1)):
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, riskFreeRate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constrainSet
    bounds = tuple(bound for asset in range(numAssets))
    result = sc.minimize(negativeSR, numAssets * [1. / numAssets], args=args, method='SLSQP',
                         bounds=bounds, constraints=constraints)
    return result


def portfolioVariance(weights, meanReturns, covMatrix):
    return portfolioPerformance(weights, meanReturns, covMatrix)[1]


def minimizeVariance(meanReturns, covMatrix, constrainSet=(0, 1)):
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constrainSet
    bounds = tuple(bound for asset in range(numAssets))
    result = sc.minimize(portfolioVariance, numAssets * [1. / numAssets], args=args, method='SLSQP',
                         bounds=bounds, constraints=constraints)
    return result


def calculateResults(meanReturns, covMatrix, riskFreeRate=0, constrainSet=(0, 1)):
    maxSR_Portfolio = maxSR(meanReturns, covMatrix, riskFreeRate, constrainSet)
    maxSR_returns, maxSR_std = portfolioPerformance(maxSR_Portfolio['x'], meanReturns, covMatrix)
    maxSR_allocation = pd.DataFrame(maxSR_Portfolio['x'], index=meanReturns.index, columns=['allocation'])
    maxSR_allocation.allocation = [round(i * 100, 2) for i in maxSR_allocation.allocation]

    minVol_Portfolio = minimizeVariance(meanReturns, covMatrix, constrainSet)
    minVol_returns, minVol_std = portfolioPerformance(minVol_Portfolio['x'], meanReturns, covMatrix)
    minVol_allocation = pd.DataFrame(minVol_Portfolio['x'], index=meanReturns.index, columns=['allocation'])
    minVol_allocation.allocation = [round(i * 100, 2) for i in minVol_allocation.allocation]

    efficientList = []
    efficientListWeights = [[]]
    targetReturns = np.linspace(minVol_returns, maxSR_returns, 30)
    for target in targetReturns:
        efficientList.append(efficientOpt(meanReturns,covMatrix,target)['fun'])
        efficientListWeights.append([target,efficientOpt(meanReturns,covMatrix,target)['x']])
        # efficientList.append(efficientOpt(meanReturns, covMatrix, target)['x'])

    minVol_returns, minVol_std = round(minVol_returns * 100, 2), round(minVol_std * 100, 2)
    maxSR_returns, maxSR_std = round(maxSR_returns * 100, 2), round(maxSR_std * 100, 2)

    return maxSR_returns, maxSR_std, maxSR_allocation, minVol_returns, minVol_std, minVol_allocation, efficientList, targetReturns, efficientListWeights

def portfolioReturn(weights, meanReturns, covMatrix):
    return portfolioPerformance(weights, meanReturns, covMatrix)[0]

def efficientOpt(meanReturns, covMatrix, returnTarget, constrainSet=(0, 1)):
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)

    constraints = ({'type':'eq', 'fun': lambda x: portfolioReturn(x,meanReturns,covMatrix)-returnTarget},
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constrainSet
    bounds = tuple(bound for assets in range(numAssets))
    effOpt = sc.minimize(portfolioVariance, numAssets * [1. / numAssets], args=args, method='SLSQP',
                         bounds=bounds, constraints=constraints)
    return effOpt

def displayWeights(array):
    print("return ",*stockList, sep=" ")
    for i in range(1,30):
        print(array[i][0]," ",*array[i][1],sep=" ")
        print("       ", *stockList, sep=" ")
    return

def EF_graph(meanReturns, covMatrix):
    maxSR_returns, maxSR_std, maxSR_allocation, minVol_returns, minVol_std, minVol_allocation, efficientList, targetReturns, efficientListWeights = calculateResults(meanReturns, covMatrix)

    MaxSharpeRatio = go.Scatter(
        name='Maximum Sharpe Ratio',
        mode='markers',
        x=[maxSR_std],
        y=[maxSR_returns],
        marker = dict(color='red', size=14, line=dict(width=3,color='black'))
    )

    MinVol = go.Scatter(
        name='Minimum Volatility',
        mode='markers',
        x=[minVol_std],
        y=[minVol_returns],
        marker=dict(color='green', size=14, line=dict(width=3, color='black'))
    )

    EF_curve = go.Scatter(
        name='Efficient Frontier',
        mode='lines',
        x=[round(ef_std*100,2) for ef_std in efficientList],
        y=[round(target*100,2) for target in targetReturns],
        marker=dict(color='green', size=14, line=dict(width=3, color='black'))
    )

    data = [MaxSharpeRatio, MinVol, EF_curve]

    layout = go.Layout(
        yaxis = dict(title="Annualised Return (%)"),
        xaxis = dict(title="Annualised Volatility (%)"),
        width = 800,
        height = 600)

    fig = go.Figure(data=data, layout=layout)

    r_efficientListWeights = efficientListWeights
    for i in range(1,30):
        r_efficientListWeights[i][0] = np.round(r_efficientListWeights[i][0]*100,2)
        r_efficientListWeights[i][1] = [round(j*100,2) for j in r_efficientListWeights[i][1]]
    displayWeights(r_efficientListWeights)

    return fig.show()


def main():
    with open("tickers.txt") as f:
        stockList = f.read().splitlines()

    stockList = sorted(stockList, key=str.lower)

    years = 3
    endDate = dt.datetime.now()
    startDate = endDate - dt.timedelta(days=365 * years)
    tradingDays = 252 * years

    weights = np.array([1/len(stockList) for i in range(0,len(stockList))])

    meanReturns, covMatrix = GetData(stockList, start=startDate, end=endDate)
    returns, std = portfolioPerformance(weights, meanReturns, covMatrix)

    # result = maxSR(meanReturns, covMatrix, riskFreeRate = 0, constrainSet=(0,1))
    # maxSR, maxWeights = result['fun'], result['x']
    # print(maxSR, maxWeights)
    # result = minimizeVariance(meanReturns, covMatrix, constrainSet=(0,1))
    # minVar, minVarWeights = result['fun'], result['x']
    # print(minVar,minVarWeights)
    # print(calculateResults(meanReturns, covMatrix))

    EF_graph(meanReturns, covMatrix)


if __name__ == "__main__":
    main()