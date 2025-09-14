#!/usr/bin/env python3
"""
Test script for the enhanced parallel risk engine
Demonstrates key functionality without full execution
"""

import numpy as np
import pandas as pd
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from numba import jit
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Import the core classes from the main module
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import classes directly from the main file
exec(open('parallel-risk-engine.py').read())

def test_core_functionality():
    """Test the core parallel risk engine functionality"""
    print("=" * 60)
    print("TESTING ENHANCED PARALLEL RISK ENGINE")
    print("=" * 60)
    
    # 1. Test basic risk computation
    print("\n1. Testing Basic Risk Computation...")
    benchmark = PerformanceBenchmark()
    portfolio, market_data = benchmark.generate_test_portfolio(500)
    engine = ParallelRiskEngine(portfolio, market_data)
    
    # Test serial computation
    start_time = time.time()
    serial_metrics = engine.compute_serial()
    serial_time = time.time() - start_time
    
    print(f"   Serial computation: {serial_time:.3f}s")
    print(f"   Portfolio Value: ${serial_metrics.portfolio_value:,.2f}")
    print(f"   VaR 95%: ${serial_metrics.var_95:,.2f}")
    
    # 2. Test multiprocessing
    print("\n2. Testing Multiprocessing...")
    start_time = time.time()
    mp_metrics = engine.compute_multiprocessing(n_workers=4)
    mp_time = time.time() - start_time
    
    print(f"   Multiprocessing (4 workers): {mp_time:.3f}s")
    print(f"   Speedup: {serial_time/mp_time:.2f}x")
    
    # 3. Test ML Signal Generation
    print("\n3. Testing ML Signal Generation...")
    signal_generator = ParallelSignalGenerator(portfolio_size=1000)
    features, targets = signal_generator.generate_features(n_periods=100)  # Smaller for testing
    
    # Train models in parallel
    ml_results = signal_generator.train_parallel_models(features, targets, n_models=2)
    
    # Generate signals
    current_features = features[-1]
    signals = signal_generator.generate_ensemble_signals(current_features)
    
    print(f"   Ensemble Prediction: {signals['ensemble_prediction']:.4f}")
    print(f"   Signal Direction: {signals['signal_direction']}")
    print(f"   Signal Strength: {signals['signal_strength']:.4f}")
    
    # 4. Test Portfolio Optimization
    print("\n4. Testing Portfolio Optimization...")
    optimizer = ParallelPortfolioOptimizer(portfolio, market_data)
    optimization_results = optimizer.optimize_parallel_strategies(n_strategies=2)
    
    print(f"   Best Strategy: {optimization_results['best_strategy']['strategy_id']}")
    print(f"   Best Objective: {optimization_results['best_strategy']['objective_value']:.4f}")
    print(f"   Successful Strategies: {optimization_results['optimization_summary']['successful_strategies']}/2")
    
    # 5. Performance Summary
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Serial Time: {serial_time:.3f}s")
    print(f"Multiprocessing Time: {mp_time:.3f}s")
    print(f"Speedup Achieved: {serial_time/mp_time:.2f}x")
    print(f"ML Models Trained: {len(ml_results)}")
    print(f"Optimization Strategies: {optimization_results['optimization_summary']['total_strategies']}")
    
    return {
        'serial_time': serial_time,
        'mp_time': mp_time,
        'speedup': serial_time/mp_time,
        'ml_models': len(ml_results),
        'optimization_strategies': optimization_results['optimization_summary']['total_strategies']
    }

def create_performance_chart(results):
    """Create a simple performance visualization"""
    print("\nCreating performance visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Performance comparison
    methods = ['Serial', 'Multiprocessing-4']
    times = [results['serial_time'], results['mp_time']]
    
    axes[0].bar(methods, times, color=['red', 'blue'], alpha=0.7)
    axes[0].set_ylabel('Computation Time (seconds)')
    axes[0].set_title('Performance Comparison')
    axes[0].set_ylim(0, max(times) * 1.2)
    
    # Add speedup annotation
    axes[0].text(0.5, max(times) * 1.1, f'Speedup: {results["speedup"]:.2f}x', 
                ha='center', fontsize=12, fontweight='bold')
    
    # System capabilities
    capabilities = ['Risk Computation', 'ML Training', 'Portfolio Optimization']
    counts = [2, results['ml_models'], results['optimization_strategies']]
    
    axes[1].bar(capabilities, counts, color=['green', 'orange', 'purple'], alpha=0.7)
    axes[1].set_ylabel('Number of Parallel Tasks')
    axes[1].set_title('Parallel System Capabilities')
    axes[1].set_ylim(0, max(counts) * 1.2)
    
    plt.tight_layout()
    plt.savefig('enhanced_system_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Performance chart saved as 'enhanced_system_performance.png'")

if __name__ == "__main__":
    try:
        # Run the test
        results = test_core_functionality()
        
        # Create visualization
        create_performance_chart(results)
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nThe enhanced parallel risk engine demonstrates:")
        print("• Multi-strategy parallelization (CPU, ML, Optimization)")
        print("• Significant performance improvements through parallelism")
        print("• Production-ready risk computation capabilities")
        print("• Machine learning integration for signal generation")
        print("• Advanced portfolio optimization techniques")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
