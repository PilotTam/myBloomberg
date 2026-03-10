import pandas as pd

class YFinanceMapper:
    """
    Maps yfinance column names to standardized internal names.
    """
    
    # Column mapping dictionaries
    INCOME_MAPPINGS = {
        'revenue': ['Total Revenue', 'Operating Revenue'],
        'ebit': ['EBIT', 'Operating Income', 'Total Operating Income As Reported'],
        'ebitda': ['Normalized EBITDA', 'EBITDA'],
        'normalized_ebitda': ['Normalized EBITDA'],
        'normalized_income': ['Normalized Income'],
        'tax_provision': ['Tax Provision'],
        'pretax_income': ['Pretax Income'],
        'tax_rate_calc': ['Tax Rate For Calcs'],
        'da_income': ['Reconciled Depreciation', 'Depreciation And Amortization In Income Statement',
                      'Depreciation Income Statement'],
        'interest_expense': ['Interest Expense', 'Interest Expense Non Operating'],
        'interest_income': ['Interest Income', 'Interest Income Non Operating'],
        'diluted_shares': ['Diluted Average Shares'],
        'basic_shares': ['Basic Average Shares'],
        'diluted_eps': ['Diluted EPS'],
        'basic_eps': ['Basic EPS'],
        'unusual_items': ['Total Unusual Items', 'Special Income Charges']
    }
    
    CASHFLOW_MAPPINGS = {
        'capex': ['Capital Expenditure', 'Capital Expenditures'],
        'da_cf': ['Depreciation And Amortization', 'Depreciation & Amortization'],
        'change_in_working_capital': ['Changes In Working Capital', 'Change In Working Capital']
    }
    
    @staticmethod
    def safe_get(df, possible_names):
        """Safely get a column from dataframe with multiple possible names."""
        for name in possible_names:
            if name in df.columns:
                return df[name]
        return pd.Series(index=df.index, dtype=float)
    
    @classmethod
    def map_financials(cls, financials):
        """Map all financial statements to standardized format."""
        inc = financials['income']
        cf = financials['cashflow']
        
        mapped = {}
        
        # Map income statement items
        for key, names in cls.INCOME_MAPPINGS.items():
            mapped[key] = cls.safe_get(inc, names)
        
        # Map cash flow items
        for key, names in cls.CASHFLOW_MAPPINGS.items():
            mapped[key] = cls.safe_get(cf, names)
        
        # Fill unusual items with 0 if missing
        if 'unusual_items' in mapped and mapped['unusual_items'].empty:
            mapped['unusual_items'] = pd.Series(0, index=inc.index)
        
        # Ensure all series are numeric
        for key in mapped:
            if not mapped[key].empty:
                mapped[key] = pd.to_numeric(mapped[key], errors='coerce')
        
        # Use D&A from cash flow if income statement version is missing
        if mapped['da_income'].empty and not mapped['da_cf'].empty:
            mapped['da_income'] = mapped['da_cf']
        
        return mapped