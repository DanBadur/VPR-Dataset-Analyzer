#!/usr/bin/env python3
"""
Test script for the new debug modes in VPR Analyzer

This script tests that debug modes work correctly without CSV creation.
"""

import os
import tempfile
from vpr_analyzer import VPRAnalyzer


def test_debug_modes():
    """Test both debug modes work correctly."""
    print("Testing VPR Analyzer Debug Modes")
    print("=" * 40)
    
    # Create a temporary config file
    config_content = """
[PATHS]
reference_dataset_path = /tmp/test_reference
query_dataset_path = /tmp/test_query
output_csv_path = test_results.csv

[PARAMETERS]
top_k = 3
debug_mode = true
debug_visualize_n = 2
debug_visualize_k = 2

[UTM]
utm_pattern = @(\d+\.?\d*)@(\d+\.?\d*)@.*\.jpg
"""
    
    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write(config_content)
        config_path = f.name
    
    try:
        # Test 1: Debug mode (runs analysis then exits without CSV)
        print("\n1. Testing --debug mode...")
        analyzer = VPRAnalyzer(config_path)
        
        # Set debug mode to true
        analyzer.config['debug_mode'] = True
        
        # Run analysis in debug mode
        results = analyzer.run_analysis()
        
        if results is None:
            print("✅ Debug mode works correctly - no CSV created")
        else:
            print("❌ Debug mode failed - CSV was created")
        
        # Test 2: Debug-only mode
        print("\n2. Testing --debug-only mode...")
        results = analyzer.run_debug_only()
        
        if results is None:
            print("✅ Debug-only mode works correctly - no CSV created")
        else:
            print("❌ Debug-only mode failed - CSV was created")
        
        print("\n" + "=" * 40)
        print("Debug mode tests completed!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
    
    finally:
        # Clean up
        os.unlink(config_path)


if __name__ == "__main__":
    test_debug_modes()
