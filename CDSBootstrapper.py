import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d

class CDSBootstrapper:
    def __init__(self, 
                 zeroRates, 
                 zeroDates, 
                 marketSpreads, 
                 marketDates, 
                 recoveryR=0.4,
                 notional=1_000_000):
        """
        Initialize CDS Bootstrapper
        
        :param zero_rates: Zero coupon rates
        :param zero_dates: Dates corresponding to zero rates
        :param market_spreads: CDS market spreads (in basis points)
        :param market_dates: Dates for market spreads
        :param recovery_rate: Expected recovery rate in case of default
        :param notional: Notional amount of the CDS contract
        """
        self.zeroRates = zeroRates
        self.zeroDates = zeroDates
        self.marketSpreads = marketSpreads
        self.marketDates = marketDates
        self.recoveryR = recoveryR
        self.notional = notional
        
        # Interpolate zero rates
        self.zeroInterp = interp1d(zeroDates, zeroRates, 
                                    kind='linear', 
                                    fill_value='extrapolate')
    
    def discount_factor(self, date):
        """Calculate discount factor for a given date"""
        zeroRate = self.zeroInterp(date)
        return np.exp(-zeroRate * date)
    
    def expected_cds_leg(self, defaultProb, maturity):
        """
        Calculate expected value of CDS protection leg
        
        :param default_prob: Cumulative default probability
        :param maturity: Contract maturity
        :return: Expected protection leg value
        """
        # Simplified protection leg calculation
        protection_value = (1 - self.recoveryR) * defaultProb * self.notional
        discountFactor = self.discount_factor(maturity)
        return protection_value * discountFactor
    
    def expected_cds_premium(self, defaultProbs, maturities, spread):
        """
        Calculate expected CDS premium leg
        
        :param default_probs: Array of default probabilities
        :param maturities: Contract maturities
        :param spread: CDS market spread
        :return: Expected premium leg value
        """
        # Simplified premium leg calculation
        premium_leg = 0
        for i in range(len(maturities)):
            survival_prob = 1 - defaultProbs[i]
            discountFactor = self.discount_factor(maturities[i])
            premium_leg += survival_prob * spread * self.notional * discountFactor
        
        return premium_leg
    
    def bootstrap_default_probabilities(self):
        """
        Bootstrap default probabilities
        
        :return: Array of cumulative default probabilities
        """
        defaultProbs = []
        # defaultProbs = np.zeros(cdsSpreads.shape)

        if len(self.marketSpreads.shape) == 1:
        
        # for j in 
        
            for i, (maturity, spread) in enumerate(zip(self.marketDates, self.marketSpreads)):
                # Define objective function to find default probability
                def objective(defaultProb):
                    currentDefaultProbs = defaultProbs + [defaultProb]
                    
                    # Protection leg
                    protection_leg = self.expected_cds_leg(defaultProb, maturity)
                    
                    # Premium leg
                    premium_leg = self.expected_cds_premium(
                        currentDefaultProbs, 
                        self.marketDates[:i+1], 
                        spread/10000  # Convert basis points to decimal
                    )
                    
                    # The objective is to make protection leg equal to premium leg
                    return protection_leg - premium_leg
                
                # Use root-finding to solve for default probability
                try:
                    defaultProb = brentq(objective, 0, 1)
                    defaultProbs.append(defaultProb)
                    # defaultProbs[i] = defaultProb 
                except ValueError:
                    # If no root is found, use last valid or default to maximum
                    defaultProbs.append(defaultProbs[-1] if defaultProbs else 1)
        
        return defaultProbs