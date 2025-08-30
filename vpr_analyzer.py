import os
import re
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
from tqdm import tqdm


class VPRAnalyzer:
    """
    Visual Place Recognition Dataset Analyzer
    
    Analyzes VPR datasets by finding the top-k closest reference images
    to each query image based on UTM coordinates.
    """
    
    def __init__(self, config_path: str = "config.ini"):
        """Initialize the VPR analyzer with configuration."""
        self.config = self._load_config(config_path)
        self.reference_images = []
        self.query_images = []
        self.reference_coords = []
        self.query_coords = []
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from file."""
        import configparser
        
        config = configparser.ConfigParser()
        config.read(config_path)
        
        # Get paths and strip quotes if present
        reference_path = config.get('PATHS', 'reference_dataset_path').strip('"\'')
        query_path = config.get('PATHS', 'query_dataset_path').strip('"\'')
        output_path = config.get('PATHS', 'output_csv_path').strip('"\'')
        
        return {
            'reference_path': reference_path,
            'query_path': query_path,
            'output_path': output_path,
            'top_k': config.getint('PARAMETERS', 'top_k'),
            'debug_mode': config.getboolean('PARAMETERS', 'debug_mode'),
            'debug_n': config.getint('PARAMETERS', 'debug_visualize_n'),
            'debug_k': config.getint('PARAMETERS', 'debug_visualize_k'),
            'utm_pattern': config.get('UTM', 'utm_pattern')
        }
    
    def parse_utm_coordinates(self, filename: str) -> Tuple[float, float]:
        """
        Parse UTM coordinates from filename.
        
        Args:
            filename: Filename in format @UTM_east@UTM_north@whatever@.jpg
            
        Returns:
            Tuple of (utm_east, utm_north) coordinates
        """
        pattern = self.config['utm_pattern']
        match = re.search(pattern, filename)
        
        if match:
            utm_east = float(match.group(1))
            utm_north = float(match.group(2))
            return utm_east, utm_north
        else:
            raise ValueError(f"Could not parse UTM coordinates from filename: {filename}")
    
    def load_dataset(self, dataset_path: str) -> Tuple[List[str], List[Tuple[float, float]]]:
        """
        Load dataset and extract UTM coordinates.
        
        Args:
            dataset_path: Path to dataset directory
            
        Returns:
            Tuple of (image_paths, coordinates)
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
        
        image_paths = []
        coordinates = []
        
        # Get all jpg files
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith('.jpg'):
                    full_path = os.path.join(root, file)
                    try:
                        utm_east, utm_north = self.parse_utm_coordinates(file)
                        image_paths.append(full_path)
                        coordinates.append((utm_east, utm_north))
                    except ValueError as e:
                        print(f"Warning: Skipping file {file} - {e}")
        
        return image_paths, coordinates
    
    def calculate_distances(self, query_coords: List[Tuple[float, float]], 
                          reference_coords: List[Tuple[float, float]]) -> np.ndarray:
        """
        Calculate Euclidean distances between query and reference coordinates.
        
        Args:
            query_coords: List of query coordinates
            reference_coords: List of reference coordinates
            
        Returns:
            Distance matrix where [i, j] is distance from query i to reference j
        """
        query_array = np.array(query_coords)
        reference_array = np.array(reference_coords)
        
        # Calculate Euclidean distances
        distances = cdist(query_array, reference_array, metric='euclidean')
        return distances
    
    def find_top_k_nearest(self, distances: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find top-k nearest neighbors for each query.
        
        Args:
            distances: Distance matrix
            k: Number of nearest neighbors to find
            
        Returns:
            Tuple of (indices, distances) for top-k nearest neighbors
        """
        # Get indices of k smallest distances for each query
        indices = np.argsort(distances, axis=1)[:, :k]
        
        # Get the corresponding distances
        k_distances = np.take_along_axis(distances, indices, axis=1)
        
        return indices, k_distances
    
    def generate_results_csv(self, query_paths: List[str], query_coords: List[Tuple[float, float]],
                           reference_paths: List[str], top_k_indices: np.ndarray,
                           top_k_distances: np.ndarray) -> pd.DataFrame:
        """
        Generate results DataFrame for CSV export.
        
        Args:
            query_paths: List of query image paths
            query_coords: List of query coordinates
            reference_paths: List of reference image paths
            top_k_indices: Top-k nearest neighbor indices
            top_k_distances: Top-k nearest neighbor distances
            
        Returns:
            DataFrame with results
        """
        results = []
        
        for i, (query_path, (utm_east, utm_north)) in enumerate(zip(query_paths, query_coords)):
            row = {
                'query_image_path': query_path,
                'utm_east': utm_east,
                'utm_north': utm_north
            }
            
            # Add top-k reference images
            for j in range(top_k_indices.shape[1]):
                ref_idx = top_k_indices[i, j]
                ref_path = reference_paths[ref_idx]
                ref_distance = top_k_distances[i, j]
                
                row[f'reference_{j+1}_path'] = ref_path
                row[f'reference_{j+1}_distance'] = ref_distance
                
                # Add UTM coordinates of reference
                ref_utm_east, ref_utm_north = self.reference_coords[ref_idx] if ref_idx < len(self.reference_coords) else (0, 0)
                row[f'reference_{j+1}_utm_east'] = ref_utm_east
                row[f'reference_{j+1}_utm_north'] = ref_utm_north
            
            results.append(row)
        
        return pd.DataFrame(results)
    
    def visualize_debug(self, query_paths: List[str], query_coords: List[Tuple[float, float]],
                       reference_paths: List[str], top_k_indices: np.ndarray, 
                       top_k_distances: np.ndarray):
        """
        Interactive visualization of debug information showing each query and its top-k nearest neighbors.
        Use left/right arrow keys or click buttons to navigate between queries.
        
        Args:
            query_paths: List of query image paths
            reference_paths: List of reference image paths
            top_k_indices: Top-k nearest neighbor indices
            top_k_distances: Top-k nearest neighbor distances
        """
        n = min(self.config['debug_n'], len(query_paths))
        k = min(self.config['debug_k'], top_k_indices.shape[1])
        
        if n == 0:
            print("No queries to visualize!")
            return
        
        print(f"\n=== INTERACTIVE DEBUG VISUALIZATION ===")
        print(f"Showing {n} queries with their top {k} nearest neighbors")
        print("Use left/right arrow keys or click navigation buttons to browse")
        print("Press 'q' to close the visualization")
        
        # Create the interactive visualization
        self._create_interactive_debug_viewer(
            query_paths, query_coords, reference_paths, 
            top_k_indices, top_k_distances, n, k
        )
    
    def _create_interactive_debug_viewer(self, query_paths, query_coords, reference_paths, 
                                       top_k_indices, top_k_distances, n, k):
        """Create an interactive matplotlib window for debugging visualization."""
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button
        
        # Set up the figure and subplots
        fig, axes = plt.subplots(2, k+1, figsize=(3.5*(k+1), 7))
        fig.suptitle('VPR Debug Visualization - Query vs Top-K References', fontsize=14)
        
        # Flatten axes for easier indexing
        if k == 1:
            axes = axes.reshape(2, 2)
        axes_flat = axes.flatten()
        
        # Current query index
        current_query = [0]
        
        # Variables to store full filenames for tooltips
        full_query_name = [""]
        full_ref_names = [[]]
        
        # Add status text first (before load_query function)
        status_text = fig.text(0.5, 0.02, f'Query 1/{n} - Use arrow keys or buttons to navigate', 
                              ha='center', va='bottom', fontsize=11)
        
        def load_query(query_idx):
            """Load and display a specific query with its references."""
            if query_idx < 0 or query_idx >= n:
                return
            
            current_query[0] = query_idx
            
            # Clear all axes
            for ax in axes_flat:
                ax.clear()
            
            # Helper function to get compact filename
            def get_compact_filename(path):
                """Extract a compact filename for display."""
                basename = os.path.basename(path)
                name_without_ext = os.path.splitext(basename)[0]
                
                # If filename is too long, truncate it
                if len(name_without_ext) > 20:
                    return name_without_ext[:17] + "..."
                return name_without_ext
            
            # Store full filenames for tooltips
            full_query_name[0] = os.path.basename(query_paths[query_idx])
            full_ref_names[0] = [os.path.basename(reference_paths[top_k_indices[query_idx, j]]) for j in range(k)]
            
            # Load and display query image
            try:
                query_img = Image.open(query_paths[query_idx])
                axes[0, 0].imshow(query_img)
                compact_query_name = get_compact_filename(query_paths[query_idx])
                axes[0, 0].set_title(f'Query {query_idx+1}/{n}\n{compact_query_name}\nUTM: ({query_coords[query_idx][0]:.2f}, {query_coords[query_idx][1]:.2f})', fontsize=9, pad=5)
                axes[0, 0].axis('off')
            except Exception as e:
                axes[0, 0].text(0.5, 0.5, f'Error loading image:\n{str(e)}', 
                               ha='center', va='center', transform=axes[0, 0].transAxes)
                axes[0, 0].set_title(f'Query {query_idx+1}/{n} - Error', fontsize=9, pad=5)
                axes[0, 0].axis('off')
            
            # Load and display reference images
            for j in range(k):
                try:
                    ref_idx = top_k_indices[query_idx, j]
                    ref_path = reference_paths[ref_idx]
                    ref_distance = top_k_distances[query_idx, j]
                    
                    # Get reference coordinates
                    ref_coords = self.reference_coords[ref_idx]
                    
                    ref_img = Image.open(ref_path)
                    compact_ref_name = get_compact_filename(ref_path)
                    axes[0, j+1].imshow(ref_img)
                    axes[0, j+1].set_title(f'Ref {j+1}\n{compact_ref_name}\nDist: {ref_distance:.2f}\nUTM: ({ref_coords[0]:.2f}, {ref_coords[1]:.2f})', fontsize=9, pad=5)
                    axes[0, j+1].axis('off')
                    
                    # Add distance info below
                    axes[1, j+1].text(0.5, 0.5, f'Distance: {ref_distance:.2f}\nUTM: ({ref_coords[0]:.2f}, {ref_coords[1]:.2f})', 
                                     ha='center', va='center', transform=axes[1, j+1].transAxes, fontsize=11)
                    axes[1, j+1].set_title(f'Reference {j+1} Info', fontsize=9, pad=5)
                    axes[1, j+1].axis('off')
                    
                except Exception as e:
                    axes[0, j+1].text(0.5, 0.5, f'Error loading image:\n{str(e)}', 
                                     ha='center', va='center', transform=axes[0, j+1].transAxes)
                    axes[0, j+1].set_title(f'Ref {j+1} - Error', fontsize=9, pad=5)
                    axes[0, j+1].axis('off')
                    
                    axes[1, j+1].text(0.5, 0.5, f'Error loading info', 
                                     ha='center', va='center', transform=axes[1, j+1].transAxes, fontsize=11)
                    axes[1, j+1].set_title(f'Reference {j+1} Info', fontsize=9, pad=5)
                    axes[1, j+1].axis('off')
            
            # Add query info below
            axes[1, 0].text(0.5, 0.5, f'Query {query_idx+1}/{n}\nUTM: ({query_coords[query_idx][0]:.2f}, {query_coords[query_idx][1]:.2f})', 
                           ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=11)
            axes[1, 0].set_title('Query Info', fontsize=9, pad=5)
            axes[1, 0].axis('off')
            
            # Update status text
            status_text.set_text(f'Query {query_idx+1}/{n} - Use arrow keys or buttons to navigate')
            
            # Update the figure
            fig.canvas.draw_idle()
        
        def next_query(event):
            """Go to next query."""
            load_query(current_query[0] + 1)
        
        def prev_query(event):
            """Go to previous query."""
            load_query(current_query[0] - 1)
        
        def on_key(event):
            """Handle keyboard events."""
            if event.key == 'right':
                next_query(None)
            elif event.key == 'left':
                prev_query(None)
            elif event.key == 'q':
                plt.close(fig)
        
        # Connect keyboard events
        fig.canvas.mpl_connect('key_press_event', on_key)
        
        # Create navigation buttons
        ax_prev = plt.axes([0.1, 0.02, 0.1, 0.04])
        ax_next = plt.axes([0.8, 0.02, 0.1, 0.04])
        
        btn_prev = Button(ax_prev, '← Previous')
        btn_next = Button(ax_next, 'Next →')
        
        btn_prev.on_clicked(prev_query)
        btn_next.on_clicked(next_query)
        
        # Load the first query
        load_query(0)
        
        # Add tooltip functionality
        def on_hover(event):
            """Show full filename on hover."""
            if event.inaxes:
                # Find which subplot we're hovering over
                for i, ax in enumerate(axes_flat):
                    if ax == event.inaxes:
                        if i == 0:  # Query image
                            ax.set_title(f'Query {current_query[0]+1}/{n}\n{full_query_name[0]}\nUTM: ({query_coords[current_query[0]][0]:.2f}, {query_coords[current_query[0]][1]:.2f})', fontsize=9, pad=5)
                        elif i <= k:  # Reference images
                            ref_idx = i - 1
                            if ref_idx < len(full_ref_names[0]):
                                ref_coords = self.reference_coords[top_k_indices[current_query[0], ref_idx]]
                                ref_distance = top_k_distances[current_query[0], ref_idx]
                                ax.set_title(f'Ref {ref_idx+1}\n{full_ref_names[0][ref_idx]}\nDist: {ref_distance:.2f}\nUTM: ({ref_coords[0]:.2f}, {ref_coords[1]:.2f})', fontsize=9, pad=5)
                        fig.canvas.draw_idle()
                        break
        
        def on_leave(event):
            """Restore compact filenames when leaving."""
            if event.inaxes:
                load_query(current_query[0])
        
        # Connect hover events
        fig.canvas.mpl_connect('motion_notify_event', on_hover)
        fig.canvas.mpl_connect('axes_leave_event', on_leave)
        
        # Show the plot
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12, top=0.88, hspace=0.4, wspace=0.2)
        plt.show()
    
    def run_analysis(self):
        """Run the complete VPR analysis."""
        print("Loading reference dataset...")
        self.reference_images, self.reference_coords = self.load_dataset(self.config['reference_path'])
        print(f"Loaded {len(self.reference_images)} reference images")
        
        print("Loading query dataset...")
        self.query_images, self.query_coords = self.load_dataset(self.config['query_path'])
        print(f"Loaded {len(self.query_images)} query images")
        
        # If debug mode is enabled, only show debug info and exit
        if self.config['debug_mode']:
            print("\n=== DEBUG MODE ENABLED ===")
            print("Showing debug visualization and exiting...")
            
            # Calculate distances for debug visualization
            print("Calculating distances for debug visualization...")
            distances = self.calculate_distances(self.query_coords, self.reference_coords)
            
            # Find top-k nearest neighbors for debug
            top_k_indices, top_k_distances = self.find_top_k_nearest(distances, self.config['top_k'])
            
            # Show debug visualization
            self.visualize_debug(
                self.query_images, self.query_coords,
                self.reference_images, top_k_indices, top_k_distances
            )
            
            print("\nDebug mode complete - exiting without CSV creation.")
            return None
        
        # Normal analysis mode (non-debug)
        print("Calculating distances...")
        distances = self.calculate_distances(self.query_coords, self.reference_coords)
        
        print("Finding top-k nearest neighbors...")
        top_k_indices, top_k_distances = self.find_top_k_nearest(distances, self.config['top_k'])
        
        print("Generating results...")
        results_df = self.generate_results_csv(
            self.query_images, self.query_coords,
            self.reference_images, top_k_indices, top_k_distances
        )
        
        # Save results
        print(f"Saving results to {self.config['output_path']}")
        results_df.to_csv(self.config['output_path'], index=False)
        
        print("Analysis complete!")
        return results_df
    
    def run_debug_only(self):
        """
        Run only the debug visualization without CSV creation.
        This is useful for quickly checking datasets and UTM parsing.
        """
        print("=== DEBUG MODE ONLY ===")
        print("Loading reference dataset...")
        self.reference_images, self.reference_coords = self.load_dataset(self.config['reference_path'])
        print(f"Loaded {len(self.reference_images)} reference images")
        
        print("Loading query dataset...")
        self.query_images, self.query_coords = self.load_dataset(self.config['query_path'])
        print(f"Loaded {len(self.query_images)} query images")
        
        # Calculate distances for debug visualization
        print("Calculating distances for debug visualization...")
        distances = self.calculate_distances(self.query_coords, self.reference_coords)
        
        # Find top-k nearest neighbors for debug
        top_k_indices, top_k_distances = self.find_top_k_nearest(distances, self.config['top_k'])
        
        # Show debug visualization
        self.visualize_debug(
            self.query_images, self.query_coords,
            self.reference_images, top_k_indices, top_k_distances
        )
        
        print("\nDebug visualization complete!")
        return None

