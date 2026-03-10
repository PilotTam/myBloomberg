class ReportGenerator:
    """
    Generates valuation reports.
    """
    
    def __init__(self, model):
        self.model = model
        
    def generate(self, valuation_result):
        """Generate a comprehensive valuation report."""
        print(f"\n{'='*60}")
        print(f"DCF VALUATION REPORT: {self.model.ticker}")
        print(f"{'='*60}")
        
        print("\nCOMPANY INFORMATION")
        print(f"   Ticker: {self.model.ticker}")
        print(f"   Sector: {self.model.market_data.get('sector', 'N/A')}")
        print(f"   Industry: {self.model.market_data.get('industry', 'N/A')}")
        print(f"   Earnings Quality: {self.model.assumption_engine.earnings_quality}")
        
        print("\nKEY ASSUMPTIONS")
        for key, value in self.model.assumptions.items():
            if isinstance(value, float):
                print(f"   {key}: {value*100:.2f}%")
        
        print("\nVALUATION SUMMARY")
        print(f"   Enterprise Value: ${valuation_result['enterprise_value']/1e6:.2f}M")
        print(f"   Equity Value: ${valuation_result['equity_value']/1e6:.2f}M")
        print(f"   Implied Share Price: ${valuation_result['price_target']:.2f}")
        print(f"   WACC: {valuation_result['wacc']*100:.2f}%")
        
        current_price = self.model.market_data.get('price')
        if current_price:
            print(f"   Current Price: ${current_price:.2f}")
            diff = (valuation_result['price_target'] - current_price) / current_price
            print(f"   Upside/Downside: {diff*100:+.1f}%")
        
        return valuation_result