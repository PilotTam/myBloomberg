import numpy as np
import pandas as pd

class SensitivityAnalyzer:
    """
    Performs sensitivity analysis on DCF inputs.
    """
    
    def __init__(self, valuation_engine):
        self.valuation_engine = valuation_engine
        self.assumptions = valuation_engine.assumptions
        self.market_data = valuation_engine.market_data
        self.financials = valuation_engine.financials
        
    def analyze(self, wacc_range=None, growth_range=None, forecast_years=5):
        """Perform sensitivity analysis on WACC and terminal growth."""
        if wacc_range is None:
            base_wacc = self.assumptions.get('wacc', 0.08)
            wacc_range = np.arange(base_wacc - 0.02, base_wacc + 0.03, 0.005)
        
        if growth_range is None:
            growth_range = np.arange(0.01, 0.04, 0.005)
        
        results = pd.DataFrame(index=[f"{g*100:.1f}%" for g in growth_range],
                              columns=[f"{w*100:.1f}%" for w in wacc_range])
        
        # Store original assumptions
        original_wacc = self.assumptions.get('wacc')
        original_growth = self.assumptions.get('terminal_growth')
        
        for w in wacc_range:
            for g in growth_range:
                self.assumptions['wacc'] = w
                self.assumptions['terminal_growth'] = g
                
                # Quick valuation
                fcff_list = self.valuation_engine.calculate_fcff(forecast_years)
                pv_fcff = sum([f / ((1 + w) ** (i+1)) for i, f in enumerate(fcff_list)])
                tv = fcff_list[-1] * (1 + g) / (w - g) if w > g else fcff_list[-1] * 10
                pv_tv = tv / ((1 + w) ** forecast_years)
                ev = pv_fcff + pv_tv
                
                # Net debt and shares
                cash = 0
                debt = 0
                if self.financials is not None:
                    cash = self.financials['balance'].get('Cash And Cash Equivalents', 
                                                          pd.Series([0])).iloc[0]
                    if 'Total Debt' in self.financials['balance'].columns:
                        debt = self.financials['balance']['Total Debt'].iloc[0]
                
                net_debt = debt - cash
                equity_value = ev - net_debt
                shares = self.market_data['shares_out']
                
                results.loc[f"{g*100:.1f}%", f"{w*100:.1f}%"] = equity_value / shares
        
        # Restore original assumptions
        if original_wacc:
            self.assumptions['wacc'] = original_wacc
        if original_growth:
            self.assumptions['terminal_growth'] = original_growth
        
        return results