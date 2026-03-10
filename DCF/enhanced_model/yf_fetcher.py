import yfinance as yf
from fredapi import Fred
import os

class YFinanceFetcher:
    """
    Handles all data fetching from yfinance and FRED.
    """
    
    def __init__(self, ticker, fred_api_key=None):
        self.ticker = ticker.upper()
        self.fred_api_key = fred_api_key or os.getenv('FRED_API_KEY')
        self.stock = yf.Ticker(self.ticker)
        
    def fetch_financials(self):
        """Fetch all financial statements."""
        return {
            'income': self.stock.income_stmt.T,
            'balance': self.stock.balance_sheet.T,
            'cashflow': self.stock.cashflow.T
        }
    
    def fetch_market_data(self):
        """Fetch market data including info and risk-free rate."""
        info = self.stock.info
        
        market_data = {
            'price': info.get('currentPrice', info.get('regularMarketPrice')),
            'shares_out': info.get('sharesOutstanding'),
            'beta': info.get('beta', 1.0),
            'market_cap': info.get('marketCap'),
            'sector': info.get('sector'),
            'industry': info.get('industry'),
            'rf_rate': self._get_risk_free_rate(),
            'erp': 0.0423  # Default equity risk premium
        }
        
        return market_data
    
    def _get_risk_free_rate(self):
        """Fetch 10-year Treasury yield from FRED."""
        # Try fredapi first
        if self.fred_api_key:
            try:
                fred = Fred(api_key=self.fred_api_key)
                data = fred.get_series('DGS10')
                latest = data.dropna().iloc[-1]
                return latest / 100.0
            except Exception as e:
                print(f"fredapi failed: {e}")
        
        # Fallback
        print("WARNING: Using default 4.0% risk-free rate")
        return 0.04