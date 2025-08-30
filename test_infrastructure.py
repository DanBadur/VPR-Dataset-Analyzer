#!/usr/bin/env python3
"""
Test script for VPR Dataset Analysis Infrastructure

This script tests the core functionality without requiring actual image datasets.
"""

import os
import tempfile
import unittest
import pandas as pd
import numpy as np
from vpr_analyzer import VPRAnalyzer
from visualization import VPRVisualizer


class TestVPRAnalyzer(unittest.TestCase):
    """Test cases for VPRAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary config file
        self.config_content = """
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
        self.config_fd, self.config_path = tempfile.mkstemp(suffix='.ini')
        with os.fdopen(self.config_fd, 'w') as f:
            f.write(self.config_content)
        
        # Create temporary directories
        self.reference_dir = tempfile.mkdtemp(prefix='test_ref_')
        self.query_dir = tempfile.mkdtemp(prefix='test_query_')
        
        # Create test image files with UTM coordinates
        self._create_test_images()
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary files and directories
        os.close(self.config_fd)
        os.unlink(self.config_path)
        
        # Remove test directories and files
        for root, dirs, files in os.walk(self.reference_dir, topdown=False):
            for file in files:
                os.unlink(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        os.rmdir(self.reference_dir)
        
        for root, dirs, files in os.walk(self.query_dir, topdown=False):
            for file in files:
                os.unlink(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        os.rmdir(self.query_dir)
    
    def _create_test_images(self):
        """Create test image files with UTM coordinates in filenames."""
        # Create reference images
        ref_coords = [
            (100.0, 200.0),
            (150.0, 250.0),
            (200.0, 300.0),
            (250.0, 350.0),
            (300.0, 400.0)
        ]
        
        for i, (east, north) in enumerate(ref_coords):
            filename = f"@{east}@{north}@ref_image_{i}@.jpg"
            filepath = os.path.join(self.reference_dir, filename)
            with open(filepath, 'w') as f:
                f.write("fake image content")
        
        # Create query images
        query_coords = [
            (120.0, 220.0),
            (180.0, 280.0)
        ]
        
        for i, (east, north) in enumerate(query_coords):
            filename = f"@{east}@{north}@query_image_{i}@.jpg"
            filepath = os.path.join(self.query_dir, filename)
            with open(filepath, 'w') as f:
                f.write("fake image content")
    
    def test_config_loading(self):
        """Test configuration loading."""
        analyzer = VPRAnalyzer(self.config_path)
        
        self.assertEqual(analyzer.config['reference_path'], '/tmp/test_reference')
        self.assertEqual(analyzer.config['query_path'], '/tmp/test_query')
        self.assertEqual(analyzer.config['top_k'], 3)
        self.assertTrue(analyzer.config['debug_mode'])
    
    def test_utm_parsing(self):
        """Test UTM coordinate parsing."""
        analyzer = VPRAnalyzer(self.config_path)
        
        # Test valid filename
        filename = "@123.456@789.012@test_image@.jpg"
        east, north = analyzer.parse_utm_coordinates(filename)
        self.assertEqual(east, 123.456)
        self.assertEqual(north, 789.012)
        
        # Test invalid filename
        with self.assertRaises(ValueError):
            analyzer.parse_utm_coordinates("invalid_filename.jpg")
    
    def test_dataset_loading(self):
        """Test dataset loading."""
        analyzer = VPRAnalyzer(self.config_path)
        
        # Load reference dataset
        ref_images, ref_coords = analyzer.load_dataset(self.reference_dir)
        self.assertEqual(len(ref_images), 5)
        self.assertEqual(len(ref_coords), 5)
        
        # Load query dataset
        query_images, query_coords = analyzer.load_dataset(self.query_dir)
        self.assertEqual(len(query_images), 2)
        self.assertEqual(len(query_coords), 2)
    
    def test_distance_calculation(self):
        """Test distance calculation."""
        analyzer = VPRAnalyzer(self.config_path)
        
        # Load datasets
        ref_images, ref_coords = analyzer.load_dataset(self.reference_dir)
        query_images, query_coords = analyzer.load_dataset(self.query_dir)
        
        # Calculate distances
        distances = analyzer.calculate_distances(query_coords, ref_coords)
        
        # Check distance matrix shape
        self.assertEqual(distances.shape, (2, 5))  # 2 queries, 5 references
        
        # Check that distances are non-negative
        self.assertTrue(np.all(distances >= 0))
    
    def test_top_k_nearest(self):
        """Test top-k nearest neighbor finding."""
        analyzer = VPRAnalyzer(self.config_path)
        
        # Load datasets
        ref_images, ref_coords = analyzer.load_dataset(self.reference_dir)
        query_images, query_coords = analyzer.load_dataset(self.query_dir)
        
        # Calculate distances
        distances = analyzer.calculate_distances(query_coords, ref_coords)
        
        # Find top-3 nearest neighbors
        indices, k_distances = analyzer.find_top_k_nearest(distances, 3)
        
        # Check shapes
        self.assertEqual(indices.shape, (2, 3))  # 2 queries, top-3
        self.assertEqual(k_distances.shape, (2, 3))
        
        # Check that indices are within valid range
        self.assertTrue(np.all(indices >= 0))
        self.assertTrue(np.all(indices < 5))  # 5 reference images
    
    def test_results_generation(self):
        """Test CSV results generation."""
        analyzer = VPRAnalyzer(self.config_path)
        
        # Load datasets
        ref_images, ref_coords = analyzer.load_dataset(self.reference_dir)
        query_images, query_coords = analyzer.load_dataset(self.query_dir)
        
        # Calculate distances and find nearest neighbors
        distances = analyzer.calculate_distances(query_coords, ref_coords)
        indices, k_distances = analyzer.find_top_k_nearest(distances, 3)
        
        # Generate results
        results_df = analyzer.generate_results_csv(
            query_images, query_coords,
            ref_images, indices, k_distances
        )
        
        # Check DataFrame structure
        self.assertEqual(len(results_df), 2)  # 2 queries
        self.assertIn('query_image_path', results_df.columns)
        self.assertIn('utm_east', results_df.columns)
        self.assertIn('utm_north', results_df.columns)
        self.assertIn('reference_1_path', results_df.columns)
        self.assertIn('reference_1_distance', results_df.columns)


class TestVPRVisualizer(unittest.TestCase):
    """Test cases for VPRVisualizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample results DataFrame
        self.sample_data = {
            'query_image_path': ['/path/to/query1.jpg', '/path/to/query2.jpg'],
            'utm_east': [100.0, 150.0],
            'utm_north': [200.0, 250.0],
            'reference_1_path': ['/path/to/ref1.jpg', '/path/to/ref2.jpg'],
            'reference_1_distance': [15.2, 23.7],
            'reference_1_utm_east': [100.1, 150.1],
            'reference_1_utm_north': [200.1, 250.1],
            'reference_2_path': ['/path/to/ref3.jpg', '/path/to/ref4.jpg'],
            'reference_2_distance': [25.8, 34.1],
            'reference_2_utm_east': [100.2, 150.2],
            'reference_2_utm_north': [200.2, 250.2]
        }
        
        self.results_df = pd.DataFrame(self.sample_data)
        self.visualizer = VPRVisualizer(self.results_df)
    
    def test_coordinate_extraction(self):
        """Test coordinate extraction from results."""
        self.assertEqual(len(self.visualizer.query_coords), 2)
        self.assertEqual(len(self.visualizer.reference_coords), 2)  # top-2
        
        # Check first query coordinates
        self.assertEqual(self.visualizer.query_coords[0], (100.0, 200.0))
        self.assertEqual(self.visualizer.query_coords[1], (150.0, 250.0))
    
    def test_plot_creation(self):
        """Test that plots can be created without errors."""
        # Test UTM coordinates plot
        fig, ax = self.visualizer.plot_utm_coordinates()
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        plt.close(fig)
        
        # Test distance distribution plot
        fig, axes = self.visualizer.plot_distance_distribution()
        self.assertIsNotNone(fig)
        self.assertIsNotNone(axes)
        plt.close(fig)
        
        # Test performance metrics plot
        fig, axes = self.visualizer.plot_performance_metrics()
        self.assertIsNotNone(fig)
        self.assertIsNotNone(axes)
        plt.close(fig)


def run_tests():
    """Run all tests."""
    print("Running VPR Infrastructure Tests...")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestVPRAnalyzer))
    suite.addTests(loader.loadTestsFromTestCase(TestVPRVisualizer))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Import matplotlib for plotting tests
    import matplotlib.pyplot as plt
    
    # Run tests
    success = run_tests()
    
    # Exit with appropriate code
    exit(0 if success else 1)

