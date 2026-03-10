import pandas as pd

class ValuationEngine:
    """
    Performs DCF valuation calculations.
    """
    
    def __init__(self, mapped_data, market_data, assumptions):
        self.mapped = mapped_data
        self.market_data = market_data
        self.assumptions = assumptions
        self.financials = None  # Will be set later
        
    def set_financials(self, financials):
        """Set the raw financials for valuation."""
        self.financials = financials
        
    def calculate_wacc(self, cost_debt):
        """Calculate Weighted Average Cost of Capital."""
        # Cost of equity: CAPM
        cost_equity = self.market_data['rf_rate'] + self.market_data['beta'] * self.market_data['erp']
        
        # Market values
        market_equity = self.market_data['price'] * self.market_data['shares_out']
        
        # Latest debt from balance sheet
        total_debt = 0
        if self.financials is not None and 'Total Debt' in self.financials['balance'].columns:
            total_debt = self.financials['balance']['Total Debt'].iloc[0]
        
        total_capital = market_equity + total_debt
        
        wacc = (market_equity / total_capital) * cost_equity + \
               (total_debt / total_capital) * cost_debt * (1 - self.assumptions['tax_rate'])
        
        return wacc
    
    def calculate_fcff(self, forecast_years=5):
        """Forecast Free Cash Flow to Firm."""
        if self.mapped['revenue'].empty:
            raise ValueError("No revenue data available")
        
        last_revenue = self.mapped['revenue'].iloc[0]
        fcff_list = []
        
        for year in range(1, forecast_years + 1):
            # Revenue with fade to terminal growth
            if year == 1:
                growth = self.assumptions['revenue_growth']
            else:
                fade = min(1.0, year / forecast_years)
                growth = self.assumptions['revenue_growth'] * (1 - fade) + \
                        self.assumptions['terminal_growth'] * fade
            
            revenue = last_revenue * (1 + growth)
            
            # Calculate FCFF components
            ebit = revenue * self.assumptions['op_margin']
            taxes = ebit * self.assumptions['tax_rate']
            da = revenue * self.assumptions['da_pct_rev']
            capex = revenue * self.assumptions['capex_pct_rev']
            nwc_change = revenue * self.assumptions['nwc_pct_rev'] - \
                        last_revenue * self.assumptions['nwc_pct_rev']
            
            fcff = ebit - taxes + da - capex - nwc_change
            fcff_list.append(fcff)
            
            last_revenue = revenue
        
        return fcff_list
    
    def terminal_value(self, last_fcff, wacc, method='perpetuity'):
        """Calculate terminal value."""
        if method == 'perpetuity':
            g = self.assumptions['terminal_growth']
            if wacc <= g:
                print("ERROR: WACC <= terminal growth, using exit multiple method")
                return self.terminal_value(last_fcff, wacc, method='exit_multiple')
            
            return last_fcff * (1 + g) / (wacc - g)
        
        else:  # exit multiple
            if not self.mapped['normalized_ebitda'].empty:
                ebitda = self.mapped['normalized_ebitda'].iloc[0]
            elif not self.mapped['ebitda'].empty:
                ebitda = self.mapped['ebitda'].iloc[0]
            else:
                ebit = self.mapped['ebit'].iloc[0]
                da = self.mapped['da_income'].iloc[0] if not self.mapped['da_income'].empty else 0
                ebitda = ebit + da
            
            multiple = self.assumptions.get('ebitda_multiple', 8.0)
            return ebitda * multiple
    
    def calculate_equity_value(self, ev, cost_debt):
        """Calculate equity value from enterprise value."""
        # Get cash and debt
        cash = 0
        total_debt = 0
        
        if self.financials is not None:
            cash = self.financials['balance'].get('Cash And Cash Equivalents', 
                                                  self.financials['balance'].get('Cash', pd.Series([0]))).iloc[0]
            if 'Total Debt' in self.financials['balance'].columns:
                total_debt = self.financials['balance']['Total Debt'].iloc[0]
        
        net_debt = total_debt - cash
        return ev - net_debt
    
    def calculate_price_target(self, forecast_years=5, terminal_method='perpetuity'):
        """Calculate price target using DCF."""
        # Get cost of debt from assumptions
        cost_debt = self.assumptions.get('cost_debt', self.market_data['rf_rate'] + 0.02)
        
        # Calculate WACC
        wacc = self.calculate_wacc(cost_debt)
        self.assumptions['wacc'] = wacc
        
        # Forecast FCFF
        fcff_list = self.calculate_fcff(forecast_years)
        
        # Calculate present value of forecast period
        pv_fcff = sum([f / ((1 + wacc) ** (i+1)) for i, f in enumerate(fcff_list)])
        
        # Terminal value
        tv = self.terminal_value(fcff_list[-1], wacc, method=terminal_method)
        pv_tv = tv / ((1 + wacc) ** forecast_years)
        
        # Enterprise value
        ev = pv_fcff + pv_tv
        
        # Equity value
        equity_value = self.calculate_equity_value(ev, cost_debt)
        
        # Price target
        shares = self.market_data['shares_out']
        price_target = equity_value / shares if shares and shares > 0 else 0
        
        return {
            'enterprise_value': ev,
            'equity_value': equity_value,
            'price_target': price_target,
            'wacc': wacc,
            'fcff_forecast': fcff_list,
            'pv_fcff': pv_fcff,
            'tv': tv,
            'pv_tv': pv_tv
        }