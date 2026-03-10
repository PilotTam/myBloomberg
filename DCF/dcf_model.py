import yfinance as yf
import pandas as pd
import numpy as np
from fredapi import Fred
from dotenv import load_dotenv
import os

class DCFModel:
    def __init__(self, ticker):
        self.ticker = ticker
        self.financials = None
        self.market_data = None
        self.assumptions = {}  # will hold all user/derived assumptions

    def fetch_financials(self, years=5):
        """Retrieve income statement, balance sheet, cash flow from yfinance."""
        stock = yf.Ticker(self.ticker)
        self.financials = {
            'income': stock.financials.T,
            'balance': stock.balance_sheet.T,
            'cashflow': stock.cashflow.T
        }
        for key in self.financials:
            self.financials[key].index = pd.to_datetime(self.financials[key].index)
            self.financials[key] = self.financials[key].sort_index(ascending=True)

    def fetch_market_data(self):
        """Get current price, shares, beta, risk-free rate, etc."""
        stock = yf.Ticker(self.ticker)
        self.market_data = {
            'price': stock.info.get('currentPrice'),
            'shares_out': stock.info.get('sharesOutstanding'),
            'beta': stock.info.get('beta'),
        }
        # Get 10y treasury from FRED
        self.market_data['rf_rate'] = self.get_risk_free_rate()

    def get_risk_free_rate(self):
        load_dotenv()
        fred = Fred(api_key=os.getenv('FRED_API_KEY'))
        # Get the most recent observation (last available date)
        data = fred.get_series('DGS10')
        # data is a Series with dates as index; take last non-null value
        latest = data.dropna().iloc[-1]
        return latest / 100.0

    def set_assumptions(self, user_assumptions=None):
        """
        Define forecast assumptions.
        user_assumptions: dict with keys like 'revenue_growth', 'op_margin', etc.
        If not provided, derive from historical averages or consensus.
        """
        self.derive_assumptions_from_history()
        if user_assumptions:
            self.assumptions.update(user_assumptions)

    def derive_assumptions_from_history(self):
        """Calculate historical averages for key drivers."""

        rev = self.financials['income']['Total Revenue']
        growth_rates = rev.pct_change().dropna()
        self.assumptions['revenue_growth'] = growth_rates.tail(3).ewm(span=3, adjust=False).mean().iloc[-1]

        ebit = self.financials['income']['EBIT']  # or Operating Income
        op_margin = (ebit / rev).dropna()
        self.assumptions['op_margin'] = op_margin.tail(3).ewm(span=3, adjust=False).mean().iloc[-1]
        
        # Tax rate
        taxes = self.financials['income']['Tax Provision']
        pretax_income = self.financials['income']['Pretax Income']
        tax_rate = (taxes / pretax_income).dropna()
        self.assumptions['tax_rate'] = tax_rate.tail(3).ewm(span=3, adjust=False).mean().iloc[-1]
        
        # D&A as % of revenue
        da = self.financials['cashflow']['Depreciation And Amortization']
        self.assumptions['da_pct_rev'] = (da / rev).dropna().tail(3).ewm(span=3, adjust=False).mean().iloc[-1]
        
        # CapEx as % of revenue
        capex = self.financials['cashflow']['Capital Expenditure'].abs()
        self.assumptions['capex_pct_rev'] = (capex / rev).dropna().tail(3).ewm(span=3, adjust=False).mean().iloc[-1]
        
        # NWC as % of revenue
        self.assumptions['nwc_pct_rev'] = 0.05  # or derive from balance sheet changes

        # Cost of debt
        interest = self.financials['income']['Interest Expense']
        debt = self.financials['balance']['Total Debt']
        combined = pd.concat([interest, debt], axis=1, keys=['interest', 'debt']).dropna()
        cost_debt_series = combined['interest'] / combined['debt']
        self.assumptions["cost_debt"] = cost_debt_series.ewm(span=3, adjust=False).mean().iloc[-1]
        
        # Add default assumptions if not already set
        self.assumptions.setdefault('erp', 0.0423)  # equity risk premium from Damodaran
        self.assumptions.setdefault('terminal_growth', 0.03)

    def calculate_fcff(self, forecast_years=5):
        """Forecast FCFF for each year based on assumptions."""
        # Starting point: latest revenue (TTM)
        last_rev = self.financials['income']['Total Revenue'].iloc[-1]
        fcff_forecast = []
        for year in range(1, forecast_years+1):
            rev_growth = self.assumptions.get(f'rev_growth_y{year}', self.assumptions['revenue_growth'])
            revenue = last_rev * (1 + rev_growth)
            ebit = revenue * self.assumptions['op_margin']
            tax = ebit * self.assumptions['tax_rate']
            d_a = revenue * self.assumptions['da_pct_rev']
            capex = revenue * self.assumptions['capex_pct_rev']
            nwc_change = revenue * self.assumptions['nwc_pct_rev'] - last_rev * self.assumptions['nwc_pct_rev']
            fcff = ebit - tax + d_a - capex - nwc_change
            fcff_forecast.append(fcff)
            last_rev = revenue
        return fcff_forecast

    def calculate_wacc(self):
        """Compute WACC using market data and assumptions."""
        cost_debt = self.assumptions.get('cost_debt', 0.058)
        cost_equity = self.market_data['rf_rate'] + self.market_data['beta'] * (self.assumptions['erp'] + self.assumptions.get("cds", 0))
        # get market values of debt and equity
        market_equity = self.market_data['price'] * self.market_data['shares_out']
        # total debt from latest balance sheet
        total_debt = self.financials['balance']['Total Debt'].iloc[-1]  # may need mapping
        total_capital = market_equity + total_debt
        wacc = (market_equity / total_capital) * cost_equity + \
               (total_debt / total_capital) * cost_debt * (1 - self.assumptions['tax_rate'])
        return wacc

    def terminal_value(self, last_fcff, wacc, sensitivity, method='perpetuity'):
        if method == 'perpetuity':
            if not sensitivity:
                self.assumptions['terminal_growth'] = 0.03
            g = self.assumptions['terminal_growth']
            return last_fcff * (1 + g) / (wacc - g)
        else:  # exit multiple
            multiple = self.assumptions['exit_multiple']
            return last_fcff * multiple  # simplified; usually use EBITDA

    def valuation(self, sensitivity=False):
        """Run the full DCF and return equity value per share."""
        fcff_list = self.calculate_fcff()
        self.assumptions['wacc'] = self.calculate_wacc() if not sensitivity else self.assumptions['wacc']
        wacc = self.assumptions['wacc']
        # discount forecast period
        pv_fcff = sum([f / ((1 + wacc) ** (i+1)) for i, f in enumerate(fcff_list)])
        # terminal value
        tv = self.terminal_value(fcff_list[-1], wacc, sensitivity)
        pv_tv = tv / ((1 + wacc) ** len(fcff_list))
        ev = pv_fcff + pv_tv
        # net debt
        cash = self.financials['balance']['Cash And Cash Equivalents'].iloc[-1]
        debt = self.financials['balance']['Total Debt'].iloc[-1]
        net_debt = debt - cash
        equity_value = ev - net_debt
        shares = self.market_data['shares_out']
        price_target = equity_value / shares
        return price_target

    def sensitivity_analysis(self, wacc_range, g_range):
        """Example: vary WACC and terminal growth."""
        results = pd.DataFrame(index=wacc_range, columns=g_range)
        for w in wacc_range:
            for g in g_range:
                self.assumptions['wacc'] = w  # override
                self.assumptions['terminal_growth'] = g
                results.loc[w, g] = self.valuation(sensitivity=True)
        return results