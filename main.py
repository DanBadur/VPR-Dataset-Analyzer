#!/usr/bin/env python3
"""
Main script for VPR Dataset Analysis

Usage:
    python main.py [--config CONFIG_FILE] [--reference REF_PATH] [--query QUERY_PATH] 
                   [--output OUTPUT_PATH] [--top-k K] [--debug] [--debug-only] [--debug-n N] [--debug-k K]
                   [--create-config]

Modes:
    --debug:        Run full analysis with debug visualization, then exit without CSV
    --debug-only:   Run only debug visualization without CSV creation (fastest)
    --create-config: Create a default configuration file and exit
    (no flags):     Run full analysis and save CSV results
"""

import argparse
import os
import sys
from vpr_analyzer import VPRAnalyzer


def create_default_config(config_path: str):
    """Create a default configuration file with proper structure."""
    import configparser
    
    config = configparser.ConfigParser()
    
    config['PATHS'] = {
        'reference_dataset_path': '',
        'query_dataset_path': '',
        'output_csv_path': 'results.csv'
    }
    config['PARAMETERS'] = {
        'top_k': '5',
        'debug_mode': 'false',
        'debug_visualize_n': '3',
        'debug_visualize_k': '3'
    }
    config['UTM'] = {
        'utm_pattern': '@(\\d+\\.?\\d*)@.*\\.jpg'
    }
    
    # Write default config
    with open(config_path, 'w') as configfile:
        config.write(configfile)
    
    print(f"Created default configuration file: {config_path}")
    print("Please edit this file to set your dataset paths before running the analysis.")


def update_config(config_path: str, **kwargs):
    """Update configuration file with new values."""
    import configparser
    
    config = configparser.ConfigParser()
    
    # Check if config file exists and has content
    if os.path.exists(config_path) and os.path.getsize(config_path) > 0:
        config.read(config_path)
    else:
        # Create default config structure if file doesn't exist or is empty
        config['PATHS'] = {
            'reference_dataset_path': '',
            'query_dataset_path': '',
            'output_csv_path': 'results.csv'
        }
        config['PARAMETERS'] = {
            'top_k': '5',
            'debug_mode': 'false',
            'debug_visualize_n': '3',
            'debug_visualize_k': '3'
        }
        config['UTM'] = {
            'utm_pattern': '@(\\d+\\.?\\d*)@(\\d+\\.?\\d*)@.*\\.jpg'
        }
    
    # Only update values that are explicitly provided (not None or empty)
    # This preserves existing config values when not overriding via command line
    
    # Update paths if explicitly provided
    if 'reference_path' in kwargs and kwargs['reference_path'] is not None:
        config.set('PATHS', 'reference_dataset_path', kwargs['reference_path'])
    
    if 'query_path' in kwargs and kwargs['query_path'] is not None:
        config.set('PATHS', 'query_dataset_path', kwargs['query_path'])
    
    if 'output_path' in kwargs and kwargs['output_path'] is not None:
        config.set('PATHS', 'output_csv_path', kwargs['output_path'])
    
    # Update parameters if explicitly provided
    if 'top_k' in kwargs and kwargs['top_k'] is not None:
        config.set('PARAMETERS', 'top_k', str(kwargs['top_k']))
    
    if 'debug_mode' in kwargs and kwargs['debug_mode'] is not None:
        config.set('PARAMETERS', 'debug_mode', str(kwargs['debug_mode']).lower())
    
    if 'debug_n' in kwargs and kwargs['debug_n'] is not None:
        config.set('PARAMETERS', 'debug_visualize_n', str(kwargs['debug_n']))
    
    if 'debug_k' in kwargs and kwargs['debug_k'] is not None:
        config.set('PARAMETERS', 'debug_visualize_k', str(kwargs['debug_k']))
    
    # Write updated config
    with open(config_path, 'w') as configfile:
        config.write(configfile)


def main():
    parser = argparse.ArgumentParser(description='VPR Dataset Analysis Tool')
    parser.add_argument('--config', default='config.ini', help='Configuration file path')
    parser.add_argument('--reference', help='Reference dataset path')
    parser.add_argument('--query', help='Query dataset path')
    parser.add_argument('--output', help='Output CSV path')
    parser.add_argument('--top-k', type=int, help='Number of top-k nearest neighbors')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--debug-only', action='store_true', help='Run only debug visualization without CSV creation')
    parser.add_argument('--debug-n', type=int, help='Number of queries to visualize in debug mode')
    parser.add_argument('--debug-k', type=int, help='Number of nearest neighbors to show in debug mode')
    parser.add_argument('--create-config', action='store_true', help='Create a default configuration file and exit')
    
    args = parser.parse_args()
    
    # Handle create-config option
    if args.create_config:
        create_default_config(args.config)
        return
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Configuration file {args.config} not found. Creating default config...")
        create_default_config(args.config)
    else:
        print(f"Using existing configuration file: {args.config}")
    
    # Only update config if we have actual values to update
    # This prevents overwriting existing config with None/empty values
    update_kwargs = {}
    
    if args.reference is not None:
        update_kwargs['reference_path'] = args.reference
    if args.query is not None:
        update_kwargs['query_path'] = args.query
    if args.output is not None:
        update_kwargs['output_path'] = args.output
    if args.top_k is not None:
        update_kwargs['top_k'] = args.top_k
    if args.debug is not None:
        update_kwargs['debug_mode'] = args.debug
    if args.debug_n is not None:
        update_kwargs['debug_n'] = args.debug_n
    if args.debug_k is not None:
        update_kwargs['debug_k'] = args.debug_k
    
    # Only call update_config if we have values to update
    if update_kwargs:
        update_config(args.config, **update_kwargs)
    
    try:
        # Initialize analyzer
        analyzer = VPRAnalyzer(args.config)
        
        # Debug: Show what was loaded from config
        print(f"Debug: Config loaded - Reference path: '{analyzer.config['reference_path']}'")
        print(f"Debug: Config loaded - Query path: '{analyzer.config['query_path']}'")
        
        # Check if required paths are set
        if not analyzer.config['reference_path']:
            print("Error: Reference dataset path not set. Please set it in config.ini or use --reference")
            sys.exit(1)
        
        if not analyzer.config['query_path']:
            print("Error: Query dataset path not set. Please set it in config.ini or use --query")
            sys.exit(1)
        
        # Run analysis based on mode
        if args.debug_only:
            # Debug-only mode
            results = analyzer.run_debug_only()
            print(f"\nDebug-only mode completed successfully!")
            print("No CSV file was created.")
        else:
            # Normal analysis mode
            results = analyzer.run_analysis()
            
            # Check if debug mode was used (results will be None)
            if results is None:
                print(f"\nDebug mode completed successfully!")
                print("No CSV file was created.")
            else:
                print(f"\nAnalysis completed successfully!")
                print(f"Results saved to: {analyzer.config['output_path']}")
                print(f"Total queries processed: {len(results)}")
                print(f"Top-k nearest neighbors: {analyzer.config['top_k']}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

