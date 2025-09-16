#!/usr/bin/env python3
"""
Interactive CSV Results Browser for VPR Dataset Analysis

This script loads a CSV file with VPR analysis results and creates an interactive
window where you can browse through the results, seeing each query image and its
top-k nearest neighbor matches from the CSV data.

Usage:
    python csv_browser.py <csv_file> [options]

Examples:
    python csv_browser.py results.csv
    python csv_browser.py results.csv --k 5
    python csv_browser.py results.csv --start-query 10
"""

import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from PIL import Image
import numpy as np


class CSVResultsBrowser:
    """Interactive browser for VPR CSV results."""
    
    def __init__(self, csv_file, k=3, debug_paths=False):
        """
        Initialize the browser with CSV data.
        
        Args:
            csv_file: Path to CSV file with VPR results
            k: Number of nearest neighbors to display
            debug_paths: If True, print debug information about path resolution
        """
        self.csv_file = csv_file
        self.k = k
        self.debug_paths = debug_paths
        self.df = None
        self.current_query = 0
        self.fig = None
        self.axes = None
        self.status_text = None
        
        # Load and validate CSV
        self._load_csv()
        self._validate_csv()
    
    def _load_csv(self):
        """Load CSV file."""
        try:
            print(f"Loading CSV file: {self.csv_file}")
            self.df = pd.read_csv(self.csv_file)
            print(f"Loaded {len(self.df)} queries from CSV file")
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            sys.exit(1)
    
    def _validate_csv(self):
        """Validate CSV structure."""
        required_columns = ['query_image_path', 'utm_east', 'utm_north']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            sys.exit(1)
        
        # Check available reference columns
        self.available_k = 0
        for i in range(1, 10):  # Check up to 9 references
            if f'reference_{i}_path' in self.df.columns:
                self.available_k = i
            else:
                break
        
        if self.available_k == 0:
            print("Error: No reference columns found in CSV")
            sys.exit(1)
        
        # Adjust k if needed
        if self.k > self.available_k:
            print(f"Warning: Requested k={self.k} but only {self.available_k} references available. Using k={self.available_k}")
            self.k = self.available_k
    
    def _get_compact_filename(self, path):
        """Get compact filename for display."""
        basename = os.path.basename(path)
        name_without_ext = os.path.splitext(basename)[0]
        
        if len(name_without_ext) > 25:
            return name_without_ext[:22] + "..."
        return name_without_ext
    
    def _load_query(self, query_idx):
        """Load and display a specific query with its references."""
        if query_idx < 0 or query_idx >= len(self.df):
            return
        
        self.current_query = query_idx
        query_row = self.df.iloc[query_idx]
        
        # Clear all axes
        for ax in self.axes.flatten():
            ax.clear()
        
        # Helper function to convert WSL path to Windows path
        def convert_wsl_to_windows(wsl_path):
            """Convert WSL path to Windows path."""
            if wsl_path.startswith('/mnt/'):
                # Convert /mnt/c/... to C:\...
                windows_path = wsl_path.replace('/mnt/c/', 'C:\\').replace('/', '\\')
                return windows_path
            return wsl_path
        
        # Helper function to display image with error handling
        def display_image(ax, image_path, title, subtitle=""):
            try:
                # Clean the path - preserve WSL paths and handle spaces properly
                original_path = image_path.strip()
                
                # Convert WSL path to Windows path for file operations
                windows_path = convert_wsl_to_windows(original_path)
                
                if self.debug_paths:
                    print(f"Debug: Original path: {original_path}")
                    print(f"Debug: Windows path: {windows_path}")
                    print(f"Debug: Path exists check: {os.path.exists(windows_path)}")
                
                if os.path.exists(windows_path):
                    img = Image.open(windows_path)
                    ax.imshow(img)
                    ax.set_title(f'{title}\n{subtitle}', fontsize=10, pad=5)
                    if self.debug_paths:
                        print(f"Debug: Successfully loaded: {windows_path}")
                else:
                    # Try alternative path formats
                    alt_paths = [
                        windows_path,
                        original_path,
                        # Try with different separators
                        original_path.replace('\\', '/'),
                        original_path.replace('/', '\\'),
                        # Try absolute path
                        os.path.abspath(windows_path),
                        # Try expanding user
                        os.path.expanduser(windows_path),
                        # Try with spaces handled differently
                        windows_path.replace(' ', '%20'),
                        # Try with quotes
                        f'"{windows_path}"'
                    ]
                    
                    if self.debug_paths:
                        print(f"Debug: Original path not found, trying alternatives...")
                        for i, alt_path in enumerate(alt_paths):
                            # Clean quotes for existence check
                            check_path = alt_path.replace('"', '')
                            exists = os.path.exists(check_path)
                            print(f"  {i+1}. {check_path} -> {'EXISTS' if exists else 'NOT FOUND'}")
                    
                    found = False
                    for alt_path in alt_paths:
                        # Clean quotes for existence check
                        check_path = alt_path.replace('"', '')
                        if os.path.exists(check_path):
                            img = Image.open(check_path)
                            ax.imshow(img)
                            ax.set_title(f'{title}\n{subtitle}', fontsize=10, pad=5)
                            found = True
                            if self.debug_paths:
                                print(f"Debug: Successfully loaded alternative: {check_path}")
                            break
                    
                    if not found:
                        error_text = f'Image not found:\n{os.path.basename(windows_path)}\n\nPath: {windows_path[:80]}...'
                        if self.debug_paths:
                            error_text += f'\n\nDebug info:\nOriginal: {original_path}\nWindows: {windows_path}\n\nNote: Check if the directory exists'
                        ax.text(0.5, 0.5, error_text, 
                               ha='center', va='center', transform=ax.transAxes, fontsize=8)
                        ax.set_title(f'{title}\n{subtitle}', fontsize=10, pad=5)
            except Exception as e:
                error_msg = str(e)[:100] + "..." if len(str(e)) > 100 else str(e)
                if self.debug_paths:
                    print(f"Debug: Exception loading image: {e}")
                    print(f"Debug: Exception type: {type(e).__name__}")
                ax.text(0.5, 0.5, f'Error loading image:\n{error_msg}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=8)
                ax.set_title(f'{title}\n{subtitle}', fontsize=10, pad=5)
            ax.axis('off')
        
        # Display query image
        query_path = query_row['query_image_path']
        query_utm = f"UTM: ({query_row['utm_east']:.2f}, {query_row['utm_north']:.2f})"
        query_title = f"Query {query_idx + 1}/{len(self.df)}"
        query_subtitle = f"{self._get_compact_filename(query_path)}\n{query_utm}"
        
        display_image(self.axes[0, 0], query_path, query_title, query_subtitle)
        
        # Display reference images
        for i in range(self.k):
            ref_path_col = f'reference_{i+1}_path'
            ref_distance_col = f'reference_{i+1}_distance'
            ref_utm_east_col = f'reference_{i+1}_utm_east'
            ref_utm_north_col = f'reference_{i+1}_utm_north'
            
            if ref_path_col in query_row:
                ref_path = query_row[ref_path_col]
                ref_distance = query_row[ref_distance_col] if ref_distance_col in query_row else 0
                ref_utm_east = query_row[ref_utm_east_col] if ref_utm_east_col in query_row else 0
                ref_utm_north = query_row[ref_utm_north_col] if ref_utm_north_col in query_row else 0
                
                ref_title = f"Ref {i+1}"
                ref_subtitle = f"{self._get_compact_filename(ref_path)}\nDist: {ref_distance:.2f}\nUTM: ({ref_utm_east:.2f}, {ref_utm_north:.2f})"
                
                display_image(self.axes[0, i+1], ref_path, ref_title, ref_subtitle)
            else:
                self.axes[0, i+1].text(0.5, 0.5, 'No reference', 
                                     ha='center', va='center', transform=self.axes[0, i+1].transAxes)
                self.axes[0, i+1].set_title(f'Ref {i+1}\nNo data', fontsize=10, pad=5)
                self.axes[0, i+1].axis('off')
        
        # Add info panels below
        # Query info
        query_info = f"""Query {query_idx + 1}/{len(self.df)}
UTM: ({query_row['utm_east']:.2f}, {query_row['utm_north']:.2f})
File: {os.path.basename(query_path)}"""
        
        self.axes[1, 0].text(0.1, 0.5, query_info, transform=self.axes[1, 0].transAxes, 
                           fontsize=10, verticalalignment='center')
        self.axes[1, 0].set_title('Query Info', fontsize=10, pad=5)
        self.axes[1, 0].axis('off')
        
        # Reference info
        for i in range(self.k):
            ref_path_col = f'reference_{i+1}_path'
            ref_distance_col = f'reference_{i+1}_distance'
            ref_utm_east_col = f'reference_{i+1}_utm_east'
            ref_utm_north_col = f'reference_{i+1}_utm_north'
            
            if ref_path_col in query_row:
                ref_distance = query_row[ref_distance_col] if ref_distance_col in query_row else 0
                ref_utm_east = query_row[ref_utm_east_col] if ref_utm_east_col in query_row else 0
                ref_utm_north = query_row[ref_utm_north_col] if ref_utm_north_col in query_row else 0
                
                ref_info = f"""Reference {i+1}
Distance: {ref_distance:.2f}
UTM: ({ref_utm_east:.2f}, {ref_utm_north:.2f})
File: {os.path.basename(query_row[ref_path_col])}"""
                
                self.axes[1, i+1].text(0.1, 0.5, ref_info, transform=self.axes[1, i+1].transAxes, 
                                     fontsize=9, verticalalignment='center')
                self.axes[1, i+1].set_title(f'Ref {i+1} Info', fontsize=10, pad=5)
                self.axes[1, i+1].axis('off')
            else:
                self.axes[1, i+1].text(0.5, 0.5, 'No data', 
                                     ha='center', va='center', transform=self.axes[1, i+1].transAxes)
                self.axes[1, i+1].set_title(f'Ref {i+1} Info', fontsize=10, pad=5)
                self.axes[1, i+1].axis('off')
        
        # Update status text
        self.status_text.set_text(f'Query {query_idx + 1}/{len(self.df)} - Use arrow keys or buttons to navigate')
        
        # Update the figure
        self.fig.canvas.draw_idle()
    
    def _next_query(self, event):
        """Go to next query."""
        self._load_query(self.current_query + 1)
    
    def _prev_query(self, event):
        """Go to previous query."""
        self._load_query(self.current_query - 1)
    
    def _on_key(self, event):
        """Handle keyboard events."""
        if event.key == 'right':
            self._next_query(None)
        elif event.key == 'left':
            self._prev_query(None)
        elif event.key == 'q':
            plt.close(self.fig)
        elif event.key.isdigit():
            # Jump to specific query number
            try:
                query_num = int(event.key)
                if query_num > 0:
                    self._load_query(query_num - 1)
            except:
                pass
    
    def browse(self, start_query=0):
        """
        Start the interactive browser.
        
        Args:
            start_query: Index of query to start with (0-based)
        """
        print(f"\n=== INTERACTIVE CSV RESULTS BROWSER ===")
        print(f"CSV file: {self.csv_file}")
        print(f"Total queries: {len(self.df)}")
        print(f"Showing top-{self.k} nearest neighbors")
        print("Navigation:")
        print("  - Left/Right arrow keys or buttons to navigate")
        print("  - Number keys (1-9) to jump to specific query")
        print("  - 'q' to quit")
        
        # Create the figure and subplots
        self.fig, self.axes = plt.subplots(2, self.k + 1, figsize=(4 * (self.k + 1), 8))
        self.fig.suptitle(f'VPR Results Browser - {os.path.basename(self.csv_file)}', fontsize=14)
        
        # Flatten axes for easier indexing
        if self.k == 1:
            self.axes = self.axes.reshape(2, 2)
        
        # Add status text
        self.status_text = self.fig.text(0.5, 0.02, '', ha='center', va='bottom', fontsize=11)
        
        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        
        # Create navigation buttons
        ax_prev = plt.axes([0.1, 0.02, 0.1, 0.04])
        ax_next = plt.axes([0.8, 0.02, 0.1, 0.04])
        
        btn_prev = Button(ax_prev, '← Previous')
        btn_next = Button(ax_next, 'Next →')
        
        btn_prev.on_clicked(self._prev_query)
        btn_next.on_clicked(self._next_query)
        
        # Load the starting query
        self._load_query(start_query)
        
        # Show the plot
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12, top=0.88, hspace=0.4, wspace=0.2)
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Interactive browser for VPR CSV results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s results.csv                    # Browse with default k=3
  %(prog)s results.csv --k 5              # Show top-5 nearest neighbors
  %(prog)s results.csv --start-query 10   # Start at query 11 (0-based)
  %(prog)s results.csv --debug-paths      # Enable debug output for path issues
        """
    )
    
    parser.add_argument('csv_file', help='Path to CSV file with VPR results')
    parser.add_argument('--k', type=int, default=3, 
                       help='Number of nearest neighbors to display (default: 3)')
    parser.add_argument('--start-query', type=int, default=0,
                       help='Index of query to start with (0-based, default: 0)')
    parser.add_argument('--debug-paths', action='store_true',
                       help='Enable debug output for path resolution')
    
    args = parser.parse_args()
    
    # Check if CSV file exists
    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file '{args.csv_file}' not found.")
        sys.exit(1)
    
    try:
        # Create and start browser
        browser = CSVResultsBrowser(args.csv_file, k=args.k, debug_paths=args.debug_paths)
        browser.browse(start_query=args.start_query)
        
    except Exception as e:
        print(f"Error during browsing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
