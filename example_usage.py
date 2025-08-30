#!/usr/bin/env python3
"""
Example usage of the VPR Dataset Analysis Infrastructure

This script demonstrates how to use the VPR analyzer programmatically
and shows various configuration options.
"""

import os
import pandas as pd
from vpr_analyzer import VPRAnalyzer
from visualization import VPRVisualizer


def example_basic_usage():
    """Example of basic usage with configuration file."""
    print("=== Basic Usage Example ===")
    
    # Initialize analyzer with config file
    analyzer = VPRAnalyzer("config.ini")
    
    # Check configuration
    print(f"Reference dataset path: {analyzer.config['reference_path']}")
    print(f"Query dataset path: {analyzer.config['query_path']}")
    print(f"Top-k: {analyzer.config['top_k']}")
    print(f"Debug mode: {analyzer.config['debug_mode']}")
    
    # Note: You need to set the paths in config.ini before running
    if not analyzer.config['reference_path'] or not analyzer.config['query_path']:
        print("Please set the dataset paths in config.ini first!")
        return
    
    # Run analysis
    try:
        results = analyzer.run_analysis()
        print(f"Analysis completed! Processed {len(results)} queries.")
    except Exception as e:
        print(f"Analysis failed: {e}")


def example_programmatic_config():
    """Example of programmatic configuration."""
    print("\n=== Programmatic Configuration Example ===")
    
    # Create a temporary config file
    config_content = """
[PATHS]
reference_dataset_path = /path/to/reference
query_dataset_path = /path/to/query
output_csv_path = example_results.csv

[PARAMETERS]
top_k = 3
debug_mode = true
debug_visualize_n = 2
debug_visualize_k = 2

[UTM]
utm_pattern = @(\d+\.?\d*)@(\d+\.?\d*)@.*\.jpg
"""
    
    with open("temp_config.ini", "w") as f:
        f.write(config_content)
    
    # Use the temporary config
    analyzer = VPRAnalyzer("temp_config.ini")
    print(f"Created analyzer with top-k={analyzer.config['top_k']}")
    print(f"Debug mode: {analyzer.config['debug_mode']}")
    
    # Clean up
    os.remove("temp_config.ini")


def example_visualization():
    """Example of visualization usage."""
    print("\n=== Visualization Example ===")
    
    # Create sample results for demonstration
    sample_data = {
        'query_image_path': ['/path/to/query1.jpg', '/path/to/query2.jpg'],
        'utm_east': [123456.789, 123457.123],
        'utm_north': [987654.321, 987655.456],
        'reference_1_path': ['/path/to/ref1.jpg', '/path/to/ref2.jpg'],
        'reference_1_distance': [15.2, 23.7],
        'reference_1_utm_east': [123456.800, 123457.200],
        'reference_1_utm_north': [987654.300, 987655.500],
        'reference_2_path': ['/path/to/ref3.jpg', '/path/to/ref4.jpg'],
        'reference_2_distance': [25.8, 34.1],
        'reference_2_utm_east': [123456.900, 123457.300],
        'reference_2_utm_north': [987654.400, 987655.600]
    }
    
    results_df = pd.DataFrame(sample_data)
    
    # Create visualizer
    visualizer = VPRVisualizer(results_df)
    
    # Generate plots
    print("Generating sample visualizations...")
    
    # UTM coordinates plot
    fig, ax = visualizer.plot_utm_coordinates()
    ax.set_title("Sample UTM Coordinates Plot")
    plt.show()
    
    # Distance distribution
    fig, axes = visualizer.plot_distance_distribution()
    plt.show()
    
    # Performance metrics
    fig, axes = visualizer.plot_performance_metrics()
    plt.show()
    
    print("Visualization examples completed!")


def example_custom_analysis():
    """Example of custom analysis workflow."""
    print("\n=== Custom Analysis Example ===")
    
    # This would be your actual dataset paths
    reference_path = "/path/to/your/reference/dataset"
    query_path = "/path/to/your/query/dataset"
    
    if not os.path.exists(reference_path) or not os.path.exists(query_path):
        print("Please update the paths in this example to point to your actual datasets!")
        return
    
    # Create analyzer
    analyzer = VPRAnalyzer()
    
    # Load datasets manually
    print("Loading reference dataset...")
    ref_images, ref_coords = analyzer.load_dataset(reference_path)
    print(f"Loaded {len(ref_images)} reference images")
    
    print("Loading query dataset...")
    query_images, query_coords = analyzer.load_dataset(query_path)
    print(f"Loaded {len(query_images)} query images")
    
    # Calculate distances
    print("Calculating distances...")
    distances = analyzer.calculate_distances(query_coords, ref_coords)
    
    # Find top-3 nearest neighbors
    print("Finding top-3 nearest neighbors...")
    top_k_indices, top_k_distances = analyzer.find_top_k_nearest(distances, 3)
    
    # Generate results
    results_df = analyzer.generate_results_csv(
        query_images, query_coords,
        ref_images, top_k_indices, top_k_distances
    )
    
    # Save results
    output_path = "custom_analysis_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    
    # Show sample results
    print("\nSample results:")
    print(results_df.head())


if __name__ == "__main__":
    print("VPR Dataset Analysis - Example Usage")
    print("=" * 50)
    
    # Run examples
    example_basic_usage()
    example_programmatic_config()
    example_visualization()
    example_custom_analysis()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nTo use with your own data:")
    print("1. Update config.ini with your dataset paths")
    print("2. Run: python main.py")
    print("3. Or use the classes programmatically as shown in the examples")

