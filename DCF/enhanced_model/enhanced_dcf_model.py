import warnings
from .yf_fetcher import YFinanceFetcher
from .yf_mapper import YFinanceMapper
from .assumption_engine import AssumptionEngine
from .debt_analyzer import DebtAnalyzer
from .valuation_engine import ValuationEngine
from .sensitivity_analyzer import SensitivityAnalyzer
from .report_generator import ReportGenerator
warnings.filterwarnings('ignore')

class EnhancedDCFModel:

    def __init__(self, ticker, fred_api_key=None):
        """
        Initialize DCF model for a given ticker.
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        fred_api_key : str, optional
            FRED API key for risk-free rate
        """
        self.ticker = ticker.upper()
        
        # Initialize components
        self.data_fetcher = YFinanceFetcher(ticker, fred_api_key)
        self.financials = {}
        self.mapped = {}
        self.market_data = {}
        self.assumptions = {}
        
        # Engines (will be initialized after data fetch)
        self.assumption_engine = None
        self.debt_analyzer = None
        self.valuation_engine = None
        self.sensitivity_analyzer = None
        self.report_generator = ReportGenerator(self)
        
        # Fetch all data and initialize
        self.fetch_all_data()
        
    def fetch_all_data(self):
        """Fetch all required data."""
        # Fetch financial statements
        self.financials = self.data_fetcher.fetch_financials()
        
        # Check if data is available
        if self.financials['income'].empty:
            raise ValueError(f"No income statement data found for {self.ticker}")
        
        # Sort all dataframes descending (most recent first)
        for key in self.financials:
            self.financials[key] = self.financials[key].sort_index(ascending=False)
        
        # Create standardized mapping
        self.mapped = YFinanceMapper.map_financials(self.financials)
        
        # Fetch market data
        self.market_data = self.data_fetcher.fetch_market_data()
        
        # Initialize engines
        self.assumption_engine = AssumptionEngine(self.mapped, self.market_data)
        self.debt_analyzer = DebtAnalyzer(self.mapped, self.market_data, self.assumptions)
        # Initialize assumptions
        self.derive_assumptions_from_history()
        self.valuation_engine = ValuationEngine(self.mapped, self.market_data, self.assumptions)
        
        # Set financials in engines that need them
        self.debt_analyzer.set_financials(self.financials)
        self.valuation_engine.set_financials(self.financials)
        
        # Analyze earnings quality
        self.assumption_engine.analyze_earnings_quality()
        
    def derive_assumptions_from_history(self, ema_span=3, forecast_years=5):
        """Derive forecast assumptions from historical data."""
        self.assumptions = self.assumption_engine.derive_assumptions(ema_span, forecast_years)
        
        # Calculate cost of debt separately
        self.assumptions['cost_debt'] = self.debt_analyzer.calculate_cost_of_debt(method='ema')
        
    def calculate_wacc(self):
        """Calculate WACC."""
        return self.valuation_engine.calculate_wacc(self.assumptions['cost_debt'])
    
    def calculate_fcff(self, forecast_years=5):
        """Calculate FCFF forecast."""
        return self.valuation_engine.calculate_fcff(forecast_years)
    
    def terminal_value(self, last_fcff, wacc, method='perpetuity'):
        """Calculate terminal value."""
        return self.valuation_engine.terminal_value(last_fcff, wacc, method)
    
    def valuation(self, forecast_years=5, terminal_method='perpetuity'):
        """
        Perform full DCF valuation.
        """
        # Calculate price target
        result = self.valuation_engine.calculate_price_target(forecast_years, terminal_method)
        
        # Display results
        print(f"Implied Share Price: ${result['price_target']:.2f}")
        
        current_price = self.market_data.get('price')
        if current_price:
            diff = (result['price_target'] - current_price) / current_price
            print(f"Current Price: ${current_price:.2f}")
            print(f"Upside/Downside: {diff*100:+.1f}%")
        
        # Add assumptions to result
        result['assumptions'] = self.assumptions.copy()
        
        return result
    
    def sensitivity_analysis(self, wacc_range=None, growth_range=None, forecast_years=5):
        """
        Perform sensitivity analysis on WACC and terminal growth.
        """
        # Initialize sensitivity analyzer if not already done
        if self.sensitivity_analyzer is None:
            self.sensitivity_analyzer = SensitivityAnalyzer(self.valuation_engine)
        
        print(f"\n{'='*60}")
        print("SENSITIVITY ANALYSIS - Price Target by WACC and Terminal Growth")
        print(f"{'='*60}")
        
        return self.sensitivity_analyzer.analyze(wacc_range, growth_range, forecast_years)
    
    def generate_report(self):
        """
        Generate a comprehensive valuation report.
        """
        valuation_result = self.valuation()
        return self.report_generator.generate(valuation_result)
    
    def reset_assumptions(self):
        """Reset assumptions to historical defaults."""
        self.assumption_engine.earnings_quality = self.assumption_engine.analyze_earnings_quality()
        self.assumptions = self.assumption_engine.derive_assumptions()
        self.assumptions['cost_debt'] = self.debt_analyzer.calculate_cost_of_debt(method='ema')
    
    def set_assumptions(self, user_assumptions):
        """Set custom assumptions."""
        self.assumptions.update(user_assumptions)
        
        # Recalculate cost of debt if not provided
        if 'cost_debt' not in user_assumptions:
            self.assumptions['cost_debt'] = self.debt_analyzer.calculate_cost_of_debt(method='ema')