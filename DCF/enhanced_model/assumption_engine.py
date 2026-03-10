import numpy as np
import pandas as pd


class AssumptionEngine:
    """
    Calculates and manages DCF assumptions from historical data.
    """
    
    def __init__(self, mapped_data, market_data):
        self.mapped = mapped_data
        self.market_data = market_data
        self.assumptions = {}
        self.earnings_quality = None
        
    def analyze_earnings_quality(self):
        """Analyze earnings quality based on unusual items."""
        if self.mapped['unusual_items'].empty or self.mapped['revenue'].empty:
            self.earnings_quality = 'unknown'
            return
        
        unusual_pct = (self.mapped['unusual_items'].abs() / self.mapped['revenue']).mean()
        
        if unusual_pct > 0.10:
            self.earnings_quality = 'poor'
            print(f"Poor earnings quality: {unusual_pct:.1%} unusual items")
        elif unusual_pct > 0.03:
            self.earnings_quality = 'moderate'
            print(f"Moderate earnings quality: {unusual_pct:.1%} unusual items")
        else:
            self.earnings_quality = 'good'
            print(f"Good earnings quality: {unusual_pct:.1%} unusual items")
    
    def derive_assumptions(self, ema_span=3, forecast_years=5):
        """Derive all forecast assumptions using EMA."""
        use_normalized = self.earnings_quality in ['poor', 'moderate']
        
        # Revenue growth
        self.assumptions['revenue_growth'] = self._calculate_growth_rate(ema_span)
        
        # Operating margin
        self.assumptions['op_margin'] = self._calculate_operating_margin(use_normalized, ema_span)
        
        # Tax rate
        self.assumptions['tax_rate'] = self._calculate_tax_rate(ema_span)
        
        # D&A % of revenue
        self.assumptions['da_pct_rev'] = self._calculate_da_percent(ema_span)
        
        # Capex % of revenue
        self.assumptions['capex_pct_rev'] = self._calculate_capex_percent(ema_span)
        
        # Working capital % of revenue (simplified)
        self.assumptions['nwc_pct_rev'] = 0.05
        
        # Terminal growth
        self.assumptions['terminal_growth'] = 0.025
        
        # Cost of debt (will be calculated separately)
        self.assumptions['cost_debt'] = None
        
        return self.assumptions
    
    def _calculate_growth_rate(self, ema_span):
        """Calculate revenue growth rate using EMA."""
        revenue = self.mapped['revenue'].dropna()[::-1]
        growth_rates = revenue.pct_change().dropna()
        
        if len(growth_rates) >= 2:
            return growth_rates.ewm(span=ema_span).mean().iloc[-1]
        return 0.05
    
    def _calculate_operating_margin(self, use_normalized, ema_span):
        """Calculate operating margin using EMA."""
        if use_normalized and not self.mapped['normalized_ebitda'].empty:
            op_income = self.mapped['normalized_ebitda']
        else:
            op_income = self.mapped['ebit']
        
        revenue = self.mapped['revenue']
        aligned = pd.concat([op_income, revenue], axis=1, keys=['op_income', 'revenue']).dropna()
        
        if not aligned.empty:
            margins = aligned['op_income'] / aligned['revenue']
            margins = margins.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(margins) >= 2:
                return margins.ewm(span=ema_span).mean().iloc[-1]
        
        return 0.15
    
    def _calculate_tax_rate(self, ema_span):
        """Calculate effective tax rate using EMA."""
        if not self.mapped['tax_rate_calc'].empty:
            tax_rates = self.mapped['tax_rate_calc']
            tax_rates = tax_rates.replace([np.inf, -np.inf], np.nan).dropna()
            if len(tax_rates) >= 2:
                return tax_rates.ewm(span=ema_span).mean().iloc[-1]
        else:
            tax_calc = self.mapped['tax_provision'] / self.mapped['pretax_income']
            tax_calc = tax_calc.replace([np.inf, -np.inf], np.nan).dropna()
            if len(tax_calc) >= 2:
                return tax_calc.ewm(span=ema_span).mean().iloc[-1]
        
        return 0.21
    
    def _calculate_da_percent(self, ema_span):
        """Calculate D&A as % of revenue using EMA."""
        da = self.mapped['da_income']
        revenue = self.mapped['revenue']
        
        if not da.empty and not revenue.empty:
            aligned = pd.concat([da, revenue], axis=1, keys=['da', 'revenue']).dropna()
            if not aligned.empty:
                da_pct = aligned['da'] / aligned['revenue']
                da_pct = da_pct.replace([np.inf, -np.inf], np.nan).dropna()
                if len(da_pct) >= 2:
                    return da_pct.ewm(span=ema_span).mean().iloc[-1]
        
        return 0.03
    
    def _calculate_capex_percent(self, ema_span):
        """Calculate Capex as % of revenue using EMA."""
        capex = self.mapped['capex'].abs()
        revenue = self.mapped['revenue']
        
        if not capex.empty and not revenue.empty:
            aligned = pd.concat([capex, revenue], axis=1, keys=['capex', 'revenue']).dropna()
            if not aligned.empty:
                capex_pct = aligned['capex'] / aligned['revenue']
                capex_pct = capex_pct.replace([np.inf, -np.inf], np.nan).dropna()
                if len(capex_pct) >= 2:
                    return capex_pct.ewm(span=ema_span).mean().iloc[-1]
        
        return 0.04