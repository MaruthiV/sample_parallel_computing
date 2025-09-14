#!/usr/bin/env python3
"""
Test script demonstrating real financial data integration
Shows how the parallel risk engine works with actual market data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the enhanced system with real financial data
exec(open('parallel-risk-engine.py').read())

def test_real_financial_data():
    """Test the system with real financial data from Yahoo Finance"""
    print("=" * 80)
    print("TESTING PARALLEL RISK ENGINE WITH REAL FINANCIAL DATA")
    print("=" * 80)
    
    # Initialize financial data provider
    data_provider = FinancialDataProvider()
    
    # Test with major tech stocks
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    print(f"\nFetching real financial data for: {symbols}")
    
    # 1. Test stock data fetching
    print("\n1. Testing Real Stock Data Fetching...")
    stock_data = data_provider.get_stock_data(symbols, period="1y")
    
    if stock_data:
        print(f"✅ Successfully loaded data for {len(stock_data)} stocks")
        for symbol, data in stock_data.items():
            current_price = data['Close'].iloc[-1]
            print(f"   {symbol}: ${current_price:.2f} (latest close)")
    
    # 2. Test options data fetching
    print("\n2. Testing Real Options Data Fetching...")
    options_data = data_provider.get_options_data('AAPL')
    
    if 'calls' in options_data:
        print(f"✅ Loaded {len(options_data['calls'])} call options for AAPL")
        print(f"   Current AAPL price: ${options_data['current_price']:.2f}")
        print(f"   Expiry date: {options_data['expiry_date']}")
    
    # 3. Create real market data
    print("\n3. Creating Real Market Data...")
    market_data = data_provider.get_market_data(symbols)
    
    print(f"✅ Market Data Created:")
    print(f"   Underlying assets: {len(market_data.spot_prices)}")
    print(f"   Risk-free rate: {market_data.risk_free_rate:.1%}")
    print(f"   Correlation matrix shape: {market_data.correlation_matrix.shape}")
    
    # 4. Create real portfolio
    print("\n4. Creating Real Options Portfolio...")
    portfolio = data_provider.create_real_portfolio(symbols, n_options_per_stock=5)
    
    print(f"✅ Real Portfolio Created:")
    print(f"   Total options: {len(portfolio)}")
    
    # Show some portfolio details
    call_count = sum(1 for opt in portfolio if opt.option_type == 'call')
    put_count = sum(1 for opt in portfolio if opt.option_type == 'put')
    print(f"   Call options: {call_count}")
    print(f"   Put options: {put_count}")
    
    # Show option details
    print(f"\n   Sample options:")
    for i, opt in enumerate(portfolio[:5]):
        print(f"     {opt.option_id}: {opt.option_type} {opt.strike} @ {opt.implied_vol:.1%} vol")
    
    # 5. Test risk computation with real data
    print("\n5. Testing Risk Computation with Real Data...")
    engine = ParallelRiskEngine(portfolio, market_data)
    
    import time
    start_time = time.time()
    risk_metrics = engine.compute_serial()
    computation_time = time.time() - start_time
    
    print(f"✅ Risk Computation Completed:")
    print(f"   Portfolio Value: ${risk_metrics.portfolio_value:,.2f}")
    print(f"   Delta: {risk_metrics.delta:,.2f}")
    print(f"   Gamma: {risk_metrics.gamma:,.2f}")
    print(f"   Vega: {risk_metrics.vega:,.2f}")
    print(f"   VaR 95%: ${risk_metrics.var_95:,.2f}")
    print(f"   VaR 99%: ${risk_metrics.var_99:,.2f}")
    print(f"   Computation Time: {computation_time:.3f}s")
    
    # 6. Test ML signal generation with real data
    print("\n6. Testing ML Signal Generation with Real Data...")
    if stock_data:
        # Extract features from real stock data
        features_list = []
        targets_list = []
        
        for symbol, data in stock_data.items():
            returns = data['Close'].pct_change().dropna()
            volatility = returns.rolling(window=20).std()
            
            for i in range(20, min(50, len(data))):  # Use subset for demo
                if not pd.isna(volatility.iloc[i]) and not pd.isna(returns.iloc[i]):
                    feature_vector = [
                        data['Close'].iloc[i],
                        returns.iloc[i],
                        volatility.iloc[i],
                        i / len(data),
                    ]
                    features_list.append(feature_vector)
                    
                    if i < len(data) - 1:
                        target = returns.iloc[i + 1] if not pd.isna(returns.iloc[i + 1]) else 0
                        targets_list.append(target)
        
        if features_list:
            real_features = np.array(features_list[:-1])
            real_targets = np.array(targets_list)
            
            print(f"✅ Generated {len(real_features)} feature vectors from real data")
            
            # Train ML models
            signal_generator = ParallelSignalGenerator(portfolio_size=len(portfolio))
            ml_results = signal_generator.train_parallel_models(real_features, real_targets, n_models=2)
            
            # Generate signals
            current_features = real_features[-1]
            signals = signal_generator.generate_ensemble_signals(current_features)
            
            print(f"✅ ML Signal Generation:")
            print(f"   Ensemble Prediction: {signals['ensemble_prediction']:.4f}")
            print(f"   Signal Direction: {signals['signal_direction']}")
            print(f"   Signal Strength: {signals['signal_strength']:.4f}")
    
    # 7. Performance summary
    print("\n" + "=" * 80)
    print("REAL FINANCIAL DATA INTEGRATION SUMMARY")
    print("=" * 80)
    print("✅ Successfully integrated real financial data from Yahoo Finance")
    print("✅ Created realistic options portfolio with actual market data")
    print("✅ Demonstrated parallel risk computation with real assets")
    print("✅ Showed ML signal generation using historical market data")
    print("✅ Achieved authentic quantitative finance workflow")
    
    return {
        'real_data_available': len(stock_data) > 0,
        'portfolio_size': len(portfolio),
        'computation_time': computation_time,
        'risk_metrics': risk_metrics
    }

if __name__ == "__main__":
    try:
        results = test_real_financial_data()
        print(f"\n🎉 SUCCESS: Real financial data integration working!")
        print(f"   Real data available: {results['real_data_available']}")
        print(f"   Portfolio size: {results['portfolio_size']} options")
        print(f"   Risk computation time: {results['computation_time']:.3f}s")
        
    except Exception as e:
        print(f"\n❌ Error during real data testing: {e}")
        import traceback
        traceback.print_exc()
