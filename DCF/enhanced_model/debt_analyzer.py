import pandas as pd
import numpy as np

class DebtAnalyzer:
    """
    Analyzes debt and calculates cost of debt.
    """
    
    def __init__(self, mapped_data, market_data, assumptions):
        self.mapped = mapped_data
        self.market_data = market_data
        self.assumptions = assumptions
        self.financials = None  # Will be set later
        
    def set_financials(self, financials):
        """Set the raw financials for debt analysis."""
        self.financials = financials
        
    def calculate_cost_of_debt(self, method='ema', ema_span=3):
        """Calculate historical cost of debt."""
        interest = self.mapped['interest_expense']
        
        # Get total debt from balance sheet
        if self.financials is None or 'Total Debt' not in self.financials['balance'].columns:
            return self.market_data['rf_rate'] + 0.02
        
        debt = self.financials['balance']['Total Debt']
        
        # Ensure aligned data
        combined = pd.concat([interest, debt], axis=1, keys=['interest', 'debt']).dropna()
        
        if combined.empty or (combined['debt'].abs() < 1e6).all():
            print("WARNING: Insufficient debt data, using synthetic rate")
            return self.market_data['rf_rate'] + 0.02
        
        # Calculate average debt
        debt_avg = (combined['debt'] + combined['debt'].shift(1)) / 2
        combined = pd.concat([combined['interest'], debt_avg], axis=1, keys=['interest', 'debt_avg']).dropna()
        
        # Calculate implied rates
        cost_debt_series = combined['interest'] / combined['debt_avg']
        cost_debt_series = cost_debt_series.replace([np.inf, -np.inf], np.nan).dropna()
        
        if cost_debt_series.empty:
            return self.market_data['rf_rate'] + 0.02
        
        # Apply chosen method
        if method == 'latest':
            return cost_debt_series.iloc[-1]
        elif method == 'average':
            return cost_debt_series.mean()
        else:  # EMA
            return cost_debt_series.ewm(span=ema_span, adjust=False).mean().iloc[-1]
    
    def get_total_debt(self):
        """Get latest total debt value."""
        if self.financials is not None and 'Total Debt' in self.financials['balance'].columns:
            return self.financials['balance']['Total Debt'].iloc[0]
        return 0
    
    def get_cash(self):
        """Get latest cash value."""
        if self.financials is not None:
            return self.financials['balance'].get('Cash And Cash Equivalents', 
                                                  self.financials['balance'].get('Cash', pd.Series([0]))).iloc[0]
        return 0