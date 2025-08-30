"""
Visualization utilities for VPR Dataset Analysis

Provides advanced plotting capabilities including:
- UTM coordinate scatter plots
- Distance distribution histograms
- Top-k nearest neighbor visualization
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import os
from typing import List, Tuple


class VPRVisualizer:
    """Visualization utilities for VPR analysis results."""
    
    def __init__(self, results_df: pd.DataFrame):
        """
        Initialize visualizer with results DataFrame.
        
        Args:
            results_df: DataFrame with VPR analysis results
        """
        self.results = results_df
        self._extract_coordinates()
    
    def _extract_coordinates(self):
        """Extract UTM coordinates from results DataFrame."""
        self.query_coords = list(zip(self.results['utm_east'], self.results['utm_north']))
        
        # Extract reference coordinates for each top-k
        self.reference_coords = []
        k = 1
        while f'reference_{k}_utm_east' in self.results.columns:
            ref_coords = list(zip(
                self.results[f'reference_{k}_utm_east'],
                self.results[f'reference_{k}_utm_north']
            ))
            self.reference_coords.append(ref_coords)
            k += 1
    
    def plot_utm_coordinates(self, figsize: Tuple[int, int] = (12, 8), 
                           show_connections: bool = True, max_queries: int = 100):
        """
        Plot UTM coordinates showing query and reference image locations.
        
        Args:
            figsize: Figure size (width, height)
            show_connections: Whether to show lines connecting queries to their nearest neighbors
            max_queries: Maximum number of queries to plot (for performance)
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Limit queries for performance
        n_queries = min(max_queries, len(self.query_coords))
        
        # Plot reference images (all)
        all_ref_coords = []
        for ref_list in self.reference_coords:
            all_ref_coords.extend(ref_list)
        
        if all_ref_coords:
            ref_east, ref_north = zip(*all_ref_coords)
            ax.scatter(ref_east, ref_north, c='lightblue', s=20, alpha=0.6, 
                      label='Reference Images', zorder=1)
        
        # Plot query images
        query_east, query_north = zip(*self.query_coords[:n_queries])
        ax.scatter(query_east, query_north, c='red', s=50, marker='*', 
                  label='Query Images', zorder=3)
        
        # Show connections if requested
        if show_connections and n_queries <= 50:  # Limit for clarity
            for i in range(n_queries):
                query_coord = self.query_coords[i]
                
                # Connect to top-1 nearest neighbor
                if self.reference_coords and i < len(self.reference_coords[0]):
                    ref_coord = self.reference_coords[0][i]
                    ax.plot([query_coord[0], ref_coord[0]], 
                           [query_coord[1], ref_coord[1]], 
                           'g-', alpha=0.3, linewidth=0.5)
        
        ax.set_xlabel('UTM East')
        ax.set_ylabel('UTM North')
        ax.set_title('VPR Dataset UTM Coordinates')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_distance_distribution(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Plot distribution of distances to nearest neighbors.
        
        Args:
            figsize: Figure size (width, height)
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        # Get distance columns
        distance_cols = [col for col in self.results.columns if 'distance' in col]
        
        for i, col in enumerate(distance_cols[:4]):  # Plot first 4 distance columns
            if i >= len(axes):
                break
                
            distances = self.results[col].dropna()
            if len(distances) > 0:
                axes[i].hist(distances, bins=30, alpha=0.7, edgecolor='black')
                axes[i].set_xlabel('Distance')
                axes[i].set_ylabel('Frequency')
                axes[i].set_title(f'{col.replace("_", " ").title()}')
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(distance_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig, axes
    
    def visualize_query_matches(self, query_idx: int, k: int = 3, 
                              figsize: Tuple[int, int] = (15, 5)):
        """
        Visualize a specific query and its top-k nearest neighbors.
        
        Args:
            query_idx: Index of query to visualize
            k: Number of nearest neighbors to show
            figsize: Figure size (width, height)
        """
        if query_idx >= len(self.results):
            raise ValueError(f"Query index {query_idx} out of range")
        
        query_row = self.results.iloc[query_idx]
        query_path = query_row['query_image_path']
        
        fig, axes = plt.subplots(1, k + 1, figsize=figsize)
        if k == 1:
            axes = [axes]
        
        # Show query image
        try:
            query_img = Image.open(query_path)
            axes[0].imshow(query_img)
            axes[0].set_title(f'Query: {os.path.basename(query_path)}')
            axes[0].axis('off')
        except Exception as e:
            axes[0].text(0.5, 0.5, f'Error loading image:\n{str(e)}', 
                        ha='center', va='center', transform=axes[0].transAxes)
            axes[0].set_title('Query Image (Error)')
            axes[0].axis('off')
        
        # Show top-k reference images
        for i in range(k):
            ref_path_col = f'reference_{i+1}_path'
            ref_distance_col = f'reference_{i+1}_distance'
            
            if ref_path_col in query_row and ref_distance_col in query_row:
                ref_path = query_row[ref_path_col]
                ref_distance = query_row[ref_distance_col]
                
                try:
                    ref_img = Image.open(ref_path)
                    axes[i + 1].imshow(ref_img)
                    axes[i + 1].set_title(f'Ref {i+1}: {os.path.basename(ref_path)}\nDist: {ref_distance:.2f}')
                    axes[i + 1].axis('off')
                except Exception as e:
                    axes[i + 1].text(0.5, 0.5, f'Error loading image:\n{str(e)}', 
                                    ha='center', va='center', transform=axes[i + 1].transAxes)
                    axes[i + 1].set_title(f'Ref {i+1} (Error)')
                    axes[i + 1].axis('off')
            else:
                axes[i + 1].text(0.5, 0.5, 'No reference', 
                                ha='center', va='center', transform=axes[i + 1].transAxes)
                axes[i + 1].set_title(f'Ref {i+1}')
                axes[i + 1].axis('off')
        
        plt.tight_layout()
        return fig, axes
    
    def plot_performance_metrics(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Plot performance metrics and statistics.
        
        Args:
            figsize: Figure size (width, height)
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Distance statistics
        distance_cols = [col for col in self.results.columns if 'distance' in col]
        if distance_cols:
            distances_df = self.results[distance_cols]
            
            # Box plot of distances
            axes[0].boxplot([distances_df[col].dropna() for col in distance_cols])
            axes[0].set_xticklabels([col.replace('_', ' ').title() for col in distance_cols], rotation=45)
            axes[0].set_ylabel('Distance')
            axes[0].set_title('Distance Distribution by Rank')
            axes[0].grid(True, alpha=0.3)
            
            # Mean distance by rank
            mean_distances = [distances_df[col].mean() for col in distance_cols]
            ranks = list(range(1, len(distance_cols) + 1))
            axes[1].plot(ranks, mean_distances, 'bo-')
            axes[1].set_xlabel('Rank')
            axes[1].set_ylabel('Mean Distance')
            axes[1].set_title('Mean Distance vs Rank')
            axes[1].grid(True, alpha=0.3)
        
        # UTM coordinate ranges
        query_east, query_north = zip(*self.query_coords)
        axes[2].scatter(query_east, query_north, c='red', s=20, alpha=0.7)
        axes[2].set_xlabel('UTM East')
        axes[2].set_ylabel('UTM North')
        axes[2].set_title('Query Image Distribution')
        axes[2].grid(True, alpha=0.3)
        
        # Dataset statistics
        stats_text = f"""
        Total Queries: {len(self.results)}
        UTM East Range: {min(query_east):.2f} - {max(query_east):.2f}
        UTM North Range: {min(query_north):.2f} - {max(query_north):.2f}
        """
        axes[3].text(0.1, 0.5, stats_text, transform=axes[3].transAxes, 
                     fontsize=12, verticalalignment='center')
        axes[3].set_title('Dataset Statistics')
        axes[3].axis('off')
        
        plt.tight_layout()
        return fig, axes
    
    def save_all_plots(self, output_dir: str = "vpr_plots"):
        """
        Save all visualization plots to a directory.
        
        Args:
            output_dir: Directory to save plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # UTM coordinates plot
        fig, _ = self.plot_utm_coordinates()
        fig.savefig(os.path.join(output_dir, 'utm_coordinates.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Distance distribution
        fig, _ = self.plot_distance_distribution()
        fig.savefig(os.path.join(output_dir, 'distance_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Performance metrics
        fig, _ = self.plot_performance_metrics()
        fig.savefig(os.path.join(output_dir, 'performance_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Sample query visualizations (first 3 queries)
        for i in range(min(3, len(self.results))):
            try:
                fig, _ = self.visualize_query_matches(i, k=3)
                fig.savefig(os.path.join(output_dir, f'query_{i}_matches.png'), dpi=300, bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                print(f"Warning: Could not create visualization for query {i}: {e}")
        
        print(f"All plots saved to {output_dir}/")

