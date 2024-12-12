import pandas as pd
import numpy as np
import QuantLib as ql
import matplotlib.pyplot as plt
# from jupyterthemes import jtplot
# jtplot.style()
from scipy.integrate import quad
from dateutil.relativedelta import relativedelta
import datetime

from scipy.interpolate import interp1d
from CDSBootstrapper import CDSBootstrapper

import matplotlib.pyplot as plt

def fmT(t, PM):
    """Calculate the instantaneous forward rate at time t."""

    delta = 2 / 365  # 2 days as a fraction of a year
    return -(np.log(PM(t + delta)) - np.log(PM(t))) / (delta * 365) * 365


PM = lambda t: (
    np.array([zeroCurve.discount(x, True) for x in t]) if isinstance(t, np.ndarray) 
    else zeroCurve.discount(t, True)
)

vectorizedPM = np.vectorize(PM)

def genCFDates(maturity, legReset):
    floatDates = generateCashFlowDates(startDate - datetime.timedelta(365), maturity, legReset)
    cashFlowDates = np.array([(date - startDate).days/365 for date in floatDates][1:])

    lastResetDate = np.full(len(simTimes), cashFlowDates[0])
    cfIndex = 1
    for i, t in enumerate(simTimes):
        if cfIndex < len(cashFlowDates):
            if simTimes[i] < cashFlowDates[cfIndex]:
                lastResetDate[i] = cashFlowDates[cfIndex-1]
            else:
                cfIndex += 1
                lastResetDate[i] = cashFlowDates[cfIndex-1]
        else:
            lastResetDate[i] = lastResetDate[i-1]

    return cashFlowDates, lastResetDate

def generateCashFlowDates(settlementDate, maturityDate, frequency):
    """
    Generate cash flow dates similar to MATLAB's cfdates function.
    
    Parameters:
    - settlementDate (datetime): The start date of the cash flows.
    - maturityDate (datetime): The end date or maturity date.
    - frequency (int): The number of payments per year (1 for annual, 2 for semi-annual, 4 for quarterly, etc.).
    
    Returns:
    - list of datetime: List of cash flow dates from settlement to maturity.
    """
    # Calculate the interval in months between payments
    months = 12 // frequency

    # Generate cash flow dates by moving backwards from maturity to settlement
    dates = []
    currentDate = maturityDate
    while currentDate > settlementDate:
        dates.append(currentDate)
        currentDate -= relativedelta(months=months)

    dates.append(settlementDate)  # Include the settlement date as the first date if needed
    dates = sorted(dates)  # Sort dates in ascending order
    
    return dates


def mtlbToPy(matlabDate):
    pyDate =  baseDate + datetime.timedelta(days = matlabDate - 1)
    # qlDate = ql.Date(pyDate.day, pyDate.month, pyDate.year)
    return pyDate



def convertRateCompounding(refRates, refComp, outComp):
    if (refComp) > 0 & (outComp > 0):
    
        outRate = (1 + refRates / refComp) ** (refComp / outComp) - 1
        return outRate * outComp
    elif (refComp < 0) & (outComp > 0):
        outRate = outComp * (np.exp(refRates / outComp) - 1)
        return outRate
    
def swapFixedLeg(legReset, Notional, fixedRate, dfactors):
    fv = fixedRate/legReset * Notional * np.ones(len(dfactors))
    fv[-1] += Notional

    pv = np.zeros(dfactors.shape)
    # pv = fv[:, None] * dfactors
    pv = fv @ dfactors

    return pv#np.sum(pv, axis = 0)

def swapFloatingLeg(spread, Notional, forwards, lastFloatRate, legReset, dfactors):

    fv = np.zeros(forwards.shape)
    fv[0] = lastFloatRate * Notional/legReset

    fv[1:] = (forwards[1:] + spread) * Notional/legReset
    fv[-1] += Notional

    pv = np.zeros(dfactors.shape)

    pv = fv * dfactors

    return np.sum(pv, axis = 0)

def exposureProfiles(discountedExposures):
    ee = np.mean(discountedExposures, axis=1)
    pfe = np.quantile(discountedExposures, 0.95, axis=1)
    mpe = np.max(pfe) * np.ones(discountedExposures.shape[0])
    epe = np.mean(np.maximum(discountedExposures, 0), axis = 1)

    return ee, pfe, mpe, epe


def swapApprox(zeroRates, tenor, compounding, lastFloatRates, 
               maturity, settle, legType, Notional, legRates, 
               legReset, resetDate):
    
    # if compounding != legReset:
    convertedRates = convertRateCompounding(refRates=zeroRates, refComp=compounding, outComp=1)

    validSwaps = maturity.index

    pvReceive = np.zeros(shape=(len(validSwaps), paths))
    pvPay = np.zeros(shape=(len(validSwaps), paths))

    for idx, i in enumerate(validSwaps):

        # if i == 27:
        #     print(np.unique(resetDate[idx]) - settle)
        couponDates = swapCouponDates[i] - settle
        # swapResetDate = np.unique(resetDate[idx]) - settle
        cfLeftYears = couponDates[couponDates > 0]
        
        if cfLeftYears.shape[0] != 0:
            # continue


            cfInterpol = interp1d(x = tenor, y = convertedRates, kind = 'linear', 
                                axis = 0, bounds_error=False, fill_value=(convertedRates[0], convertedRates[-1]))
            
            cfRates = cfInterpol(cfLeftYears)
            # print(swapResetDate)
            dfactors = np.power((1+cfRates), -cfLeftYears[:, None])
            deltaCfYears = np.diff(cfLeftYears)

            cfForwards = np.zeros(dfactors.shape)
            cfForwards[0, :] = cfRates[0, :]
            cfForwards[1:] = (dfactors[:-1, :] / dfactors[1:, :] - 1) / deltaCfYears[:, None]
            # print(legType.shape)
            
            if legType.loc[i] == 1:
                # Calc fixed leg
                fixedR = legRates.LegRateReceiving.loc[i]
                pvFixedLeg = swapFixedLeg(legReset=legReset.loc[i], Notional=Notional.loc[i], 
                                          fixedRate= fixedR, dfactors = dfactors)
                pvReceive[idx, :] = pvFixedLeg
                
                floatR = legRates.LegRatePaying.loc[i]
                pvFloatingLeg = swapFloatingLeg(floatR/1e4, Notional.loc[i], cfForwards, 
                                                lastFloatRates[idx], legReset.loc[i], dfactors)
                pvPay[idx, :] = pvFloatingLeg

            elif legType.loc[i] == 0:
                # Calc floating leg
                floatR = legRates.LegRateReceiving.loc[i]
                pvFloatingLeg = swapFloatingLeg(floatR/1e4, Notional.loc[i], cfForwards,
                                                lastFloatRates[idx], legReset.loc[i], dfactors)
                pvReceive[idx, :] = pvFloatingLeg

                fixedR = legRates.LegRatePaying.loc[i]
                pvFixedLeg = swapFixedLeg(legReset=legReset.loc[i], Notional=Notional.loc[i],
                                          fixedRate= fixedR, dfactors = dfactors)
                pvPay[idx, :] = pvFixedLeg

            

    return pvReceive - pvPay






def computeIRSMtM(swaps, simTimes, zeroRates, tenors, lastResetDate):
    oneYrindex = np.where(tenors == 1)[0][0]
    oneYrRates = zeroRates[:, oneYrindex]

    latestFloatingRates = np.zeros(shape = (len(simTimes), swaps.shape[0], zeroRates.shape[2]))

    for idx, swap in swaps.iterrows():
        # at every last reset date you're interpolating the 1 year rate based on the interpolation of zero rates from simTime
        # This gives us the floating rate that was set at the reset date, which will be the first coupon of floating leg
        interpolator = interp1d(simTimes, oneYrRates, kind='linear', fill_value=swap.LatestFloatingRate, bounds_error=False, axis = 0)
        latestFloatingRates[:, idx] = interpolator(lastResetDate[idx])

    compounding = -1

    # print(f'latestFloatingRates.shape = {latestFloatingRates.shape}')
    price = np.zeros(shape = (len(simTimes), swaps.shape[0], paths))
    for i, t in enumerate(simTimes):
        curveRates = zeroRates[i, :, :]
        latestFloatingRate = latestFloatingRates[i, :, :]
        validSwaps = swaps.Maturity > np.datetime64(allSimDates[i])

        validResetDate = lastResetDate[validSwaps, :]

        refSwaps = swaps.loc[validSwaps, :]
        price[i, validSwaps, :] = swapApprox(zeroRates = curveRates, compounding = compounding, tenor = tenors,
                           lastFloatRates = latestFloatingRate[validSwaps, :], maturity = refSwaps.Maturity, 
                           settle = t, legType = refSwaps.LegType, Notional = refSwaps.Principal, 
                           legRates = refSwaps[['LegRateReceiving', 'LegRatePaying']], 
                           legReset = refSwaps.Period, resetDate = validResetDate)
        

    return price


baseDate = datetime.datetime(16,1,1)

tenor = np.array([0.25,.5,1,5,7,10,20,30])
zeroRate = np.array([0.033, 0.034, 0.035, 0.040, 0.042, 0.044, 0.048, 0.044])

# settle date (ie of the curve)
startDate = pd.Timestamp('2023-12-31')
settle = ql.Date(startDate.day, startDate.month, startDate.year)
# Define dates for quantlib zero curve
dates = []
for idx in tenor:
    yrs = pd.DateOffset(years = int(idx))
    fracDays = pd.Timedelta(days= (idx%1) * 360)
    new = startDate + yrs + fracDays

    dates.append(new)
curveDate = [ql.Date(i.date().day, i.date().month, i.date().year) for i in dates]

# Define other parameters
calendar = ql.UnitedStates(ql.UnitedStates.FederalReserve)
convention = ql.Unadjusted
day_count = ql.Thirty360(convention)

# Define the current discount factors
crntDiscounts = np.exp(-zeroRate * tenor)

# initialize the current zero curve using quantlib

# zeroCurve = ql.DiscountCurve(curveDate, crntDiscounts, ql.Actual360())
zeroCurve = ql.LogLinearZeroCurve(curveDate, zeroRate, ql.Actual360(), calendar, ql.LogLinear(), ql.Compounded, ql.Continuous)

a       = 0.2
sigma   = 0.01
paths = 20000
# f_t_0 = ratesDf.values[1:]
T = 7
dt = 1/52

zeroCurve.zeroRate(0, ql.Compounded, ql.Semiannual, True).rate()
simTimes = np.arange(0, T + dt, dt)
# allDiscounts = [PM(t) for t in simTimes]
allDiscounts = vectorizedPM(simTimes)
allSimDates = [settle + int(i * 365.25) for i in simTimes]
allSimDates = [datetime.date(t.year(), t.month(), t.dayOfMonth()) for t in allSimDates]

t = np.arange(0, T + dt, dt)[1:]


intervals = int(T/dt)
# t = ratesDf.index.values[1:]
g_t = lambda t: (sigma**2 / (2 * a**2)) * (1 - np.exp(-a * t))**2

r0 = zeroCurve.zeroRate(0, ql.Compounded, ql.Semiannual, True).rate()
# f_t_0 = np.array([fmT(x, PM) for x in t])
f_t_0 = fmT(t, vectorizedPM)

gValues = f_t_0.T+ g_t(t)

exp_factors = np.exp(-a * np.arange(1, intervals+1) * dt)


exp_decay_matrix = np.exp(-a * (t[:, None] - t[None, :]))  # Matrix of exp(-a*(t_i - t_j))
exp_decay_matrix = np.tril(exp_decay_matrix)  # Make it lower triangular
dW = np.random.normal(loc=0, scale=np.sqrt(dt), size = (paths, intervals))

stochastic = (sigma * exp_decay_matrix @ dW.T).T
drift = r0 * exp_factors + (gValues - gValues[0])

r = drift + stochastic
r = np.insert(r, 0, r0, axis = 1
              )

ratesStoch = pd.DataFrame(r.T, index= simTimes)

ratesStoch.sample(50, axis = 1).plot(figsize = (12, 5), alpha = 0.7, legend=False, lw  = 0.5)


gamma = lambda t: quad(lambda u: sigma**2 * np.exp(-2 * a * (t - u)), 0, t)[0]
gamma = np.vectorize(gamma)

A = lambda t, T: vectorizedPM(T) / vectorizedPM(t) * np.exp(-0.5 * B(t, T)**2 * gamma(t) + B(t, T) * fmT(t, vectorizedPM))
B = lambda t, T: (1 - np.exp(-a * (T - t))) / a




tT = simTimes + tenor[:, None]
AVal = (A(simTimes[1:], tT[:, 1:]))
BVal = B(simTimes[1:], tT[:, 1:])

rtT= np.zeros((len(simTimes), len(tenor), paths))
rtT[0, :, :] = zeroRate[:, np.newaxis]
ptT = AVal[None, :, :].T * np.exp((r[:, None, 1:] * -BVal[None, :, :])).T

rtT[1:, :, :] = -np.log(ptT)/tenor[None, :, None]




scen = 502
Z = rtT[:, :, scen].T


X,Y = np.meshgrid(simTimes, tenor)
fig = plt.figure()
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surface = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k', linewidth = 0.5, alpha = 0.99)

ax.set_xlabel("Simulation Times")
ax.set_ylabel("Tenors (Years)")
ax.set_zlabel("Zero Rates")

ax.set_title("Surface Plot for Hull White Term Structure")
ax.view_init(elev=20, azim=-70, roll=0)
fig.show()



xlFile = pd.ExcelFile('/Users/ravikohli/Desktop/Business:Research/Education/QFA/OwnPractice/BalconyView.xls', engine = 'xlrd')
cdsData = xlFile.parse(sheet_name='CDS Spreads', parse_dates= True, index_col=0)

swaps = xlFile.parse(sheet_name='Swap Portfolio', parse_dates= True)
swaps['Maturity'] = swaps.Maturity.apply(mtlbToPy)
refSwap = swaps.iloc[7, :]



legType = np.array([refSwap.LegType == 1,
                    refSwap.LegType != 1]
                    ).T * 1
legType = np.array([swaps.LegType == 1, 
                    swaps.LegType != 1]).T * 1


legRate = refSwap.loc[['LegRateReceiving', 'LegRatePaying']].values

swapCouponDates = {}
swapLastResetDate = np.zeros(shape = (len(swaps), len(simTimes)))
for idx, swap in swaps.iterrows():
    swapCFDates, swapLastResetDate[idx, :] = genCFDates(swap.Maturity, swap.Period)
    swapCouponDates[swap.name] = swapCFDates


values = computeIRSMtM(swaps, simTimes, rtT, tenor, swapLastResetDate)
discountedValues = values * allDiscounts[:, None, None]


portfolioExposure = 0
nbCpts = swaps.CounterpartyID.nunique()
cptyExposure = np.zeros(shape = (discountedValues.shape[0], discountedValues.shape[2], nbCpts))
cptyGrouped = swaps.groupby(['CounterpartyID'])
for cptyId, df in cptyGrouped:
    cptyIdx = df.index


    cptyNetIdx = df.loc[~df.NettingID.isna(), :].index

    cptyNonNetIdx = df.loc[df.NettingID.isna(), :].index

    nettingSetSum = np.maximum(np.sum(discountedValues[:, cptyNetIdx,:], axis = 1),0)

    nonNettingSum = np.sum(np.maximum(discountedValues[:, cptyNonNetIdx, :], 0), axis = 1)
    # cptyExposure = nettingSetSum + nonNettingSum
    exposure = nettingSetSum + nonNettingSum

    # cptyExposure[cptyId[0]] = exposure
    cptyExposure[:, :, cptyId[0] - 1] = exposure

    portfolioExposure += exposure
portEE, portPFE, portMPE, portEPE = exposureProfiles(portfolioExposure)
cptEE, cptPFE, cptMPE, cptEPE = exposureProfiles(cptyExposure)
plt.figure(figsize=(8, 5))
plt.plot(allSimDates,(np.array([portEE, portPFE, portMPE])).T)

plt.legend(['EE', 'PFE P95', 'MPE', 'EPE'])
plt.show()

recoveryRate = 0.4
zeroDates = simTimes  # Years
allZeros = -np.log(allDiscounts)/(simTimes + 1e-6)
firstZero = zeroCurve.zeroRate(0, ql.Compounded, ql.Continuous, True).rate()
allZeros[np.isnan(allZeros)] = firstZero
zeroRates = allZeros  # Interest rates

cdsExpiration = np.array([((index  + relativedelta(years = 16)) - startDate).days/365 for index in cdsData.index])

cdsSpreads = cdsData.values
# market_dates = np.array([0.5, 1, 1.5, 2])  # Maturities
marketDates = cdsExpiration


marketSpreads = cdsSpreads  # CDS spreads in basis points

defaultProbsCpty = []
for col in range(marketSpreads.shape[1]):
# Initialize and run bootstrapping
    bootstrapper = CDSBootstrapper(
        zeroRates, 
        zeroDates, 
        marketSpreads[:, col], 
        marketDates
    )

    defaultProbs = np.array(bootstrapper.bootstrap_default_probabilities())
    defaultProbsCpty.append(defaultProbs)

    # defaultProbsCpty[:, col] = defaultProbs
defaultProbsCpty = np.array(defaultProbsCpty).T


cdsinterpol = interp1d(marketDates, defaultProbsCpty, 'linear', fill_value='extrapolate', axis = 0)
defProbCDS = cdsinterpol(simTimes).T
cva = (1 - recoveryRate) * np.sum(cptEPE[1:, :] * np.diff(defProbCDS).T, axis = 0)

fig1, ax1 = plt.subplots(figsize = (9, 5))
ax1.title.set_text('CDS Default Probability')
ax1.plot(allSimDates, defProbCDS.T)
fig1.legend([f'Cpt {i}' for i in range(1, nbCpts+ 1)])
fig1.show()

fig, ax = plt.subplots(figsize = (9, 5))
ax.title.set_text('Counterparty CVA')
ax.bar(x = np.arange(1, len(cva) + 1), height=cva, width = 0.8)
fig.show()

for i, j in enumerate(cva):
    print(f'CVA for Counterparty {i + 1} is: $ {round(j, 2)}')