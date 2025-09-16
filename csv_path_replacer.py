#!/usr/bin/env python3
"""
CSV Path Replacer for VPR Dataset Analysis

This script finds and replaces path patterns in CSV files, which is useful when
moving datasets or changing directory structures.

Usage:
    python csv_path_replacer.py <csv_file> <old_pattern> <new_pattern> [options]

Examples:
    python csv_path_replacer.py results.csv "/raid/ygalron/" "/mnt/c/Users/danba/"
    python csv_path_replacer.py results.csv "/old/path/" "/new/path/" --backup
    python csv_path_replacer.py results.csv "C:\\old\\" "D:\\new\\" --dry-run
"""

import argparse
import os
import sys
import pandas as pd
import shutil
from datetime import datetime


class CSVPathReplacer:
    """Replace path patterns in CSV files."""
    
    def __init__(self, csv_file, old_pattern, new_pattern, dry_run=False, backup=True, output_file=None):
        """
        Initialize the path replacer.
        
        Args:
            csv_file: Path to CSV file to process
            old_pattern: Pattern to find and replace
            new_pattern: Replacement pattern
            dry_run: If True, only show what would be changed without making changes
            backup: If True, create a backup of the original file
            output_file: Path for output file (if None, will create auto-generated name)
        """
        self.csv_file = csv_file
        self.old_pattern = old_pattern
        self.new_pattern = new_pattern
        self.dry_run = dry_run
        self.backup = backup
        self.output_file = output_file
        self.df = None
        self.changes_made = 0
        self.columns_affected = []
    
    def _load_csv(self):
        """Load CSV file."""
        try:
            print(f"Loading CSV file: {self.csv_file}")
            self.df = pd.read_csv(self.csv_file)
            print(f"Loaded {len(self.df)} rows and {len(self.df.columns)} columns")
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            sys.exit(1)
    
    def _create_backup(self):
        """Create backup of original file."""
        if self.backup and not self.dry_run:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = f"{self.csv_file}.backup_{timestamp}"
            try:
                shutil.copy2(self.csv_file, backup_file)
                print(f"Backup created: {backup_file}")
            except Exception as e:
                print(f"Warning: Could not create backup: {e}")
    
    def _find_path_columns(self):
        """Find columns that likely contain file paths."""
        path_columns = []
        for col in self.df.columns:
            if 'path' in col.lower() or 'file' in col.lower():
                path_columns.append(col)
        return path_columns
    
    def _replace_paths_in_column(self, column_name):
        """Replace paths in a specific column."""
        if column_name not in self.df.columns:
            return 0
        
        column_changes = 0
        original_values = self.df[column_name].copy()
        
        # Replace the pattern in the column
        self.df[column_name] = self.df[column_name].astype(str).str.replace(
            self.old_pattern, self.new_pattern, regex=False
        )
        
        # Count changes
        column_changes = (original_values != self.df[column_name]).sum()
        
        if column_changes > 0:
            self.columns_affected.append(column_name)
            print(f"  Column '{column_name}': {column_changes} paths updated")
        
        return column_changes
    
    def _show_sample_changes(self, column_name, max_samples=5):
        """Show sample changes for a column."""
        if column_name not in self.df.columns:
            return
        
        # Find rows where changes were made
        original_df = pd.read_csv(self.csv_file)
        changed_mask = original_df[column_name].astype(str) != self.df[column_name].astype(str)
        changed_rows = self.df[changed_mask]
        
        if len(changed_rows) == 0:
            return
        
        print(f"\n  Sample changes in '{column_name}':")
        for i, (idx, row) in enumerate(changed_rows.head(max_samples).iterrows()):
            old_path = original_df.loc[idx, column_name]
            new_path = row[column_name]
            print(f"    Row {idx}:")
            print(f"      Old: {old_path}")
            print(f"      New: {new_path}")
            print()
    
    def replace_paths(self):
        """Replace path patterns in the CSV file."""
        print(f"\n=== CSV PATH REPLACER ===")
        print(f"File: {self.csv_file}")
        print(f"Old pattern: '{self.old_pattern}'")
        print(f"New pattern: '{self.new_pattern}'")
        print(f"Dry run: {self.dry_run}")
        print(f"Backup: {self.backup}")
        
        # Load CSV
        self._load_csv()
        
        # Find path columns
        path_columns = self._find_path_columns()
        print(f"\nFound {len(path_columns)} potential path columns: {path_columns}")
        
        if not path_columns:
            print("No path columns found. Searching all columns...")
            path_columns = list(self.df.columns)
        
        # Create backup if needed
        if not self.dry_run:
            self._create_backup()
        
        # Replace paths in each column
        print(f"\nReplacing paths...")
        for col in path_columns:
            changes = self._replace_paths_in_column(col)
            self.changes_made += changes
        
        # Show summary
        print(f"\n=== SUMMARY ===")
        print(f"Total changes made: {self.changes_made}")
        print(f"Columns affected: {len(self.columns_affected)}")
        
        if self.columns_affected:
            print(f"Affected columns: {self.columns_affected}")
        
        # Show sample changes
        if self.changes_made > 0 and not self.dry_run:
            print(f"\nSample changes:")
            for col in self.columns_affected[:3]:  # Show samples from first 3 columns
                self._show_sample_changes(col)
        
        # Save changes if not dry run
        if not self.dry_run and self.changes_made > 0:
            try:
                # Determine output file name
                if self.output_file is None:
                    # Auto-generate output filename
                    base_name = os.path.splitext(self.csv_file)[0]
                    extension = os.path.splitext(self.csv_file)[1]
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    self.output_file = f"{base_name}_updated_{timestamp}{extension}"
                
                self.df.to_csv(self.output_file, index=False)
                print(f"\nChanges saved to: {self.output_file}")
                print(f"Original file preserved: {self.csv_file}")
            except Exception as e:
                print(f"Error saving changes: {e}")
                sys.exit(1)
        elif self.dry_run:
            print(f"\nDry run complete - no changes were made")
            if self.changes_made > 0:
                print("Run without --dry-run to apply changes")
        else:
            print(f"\nNo changes needed - no paths matched the pattern")


def main():
    parser = argparse.ArgumentParser(
        description='Replace path patterns in CSV files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s results.csv "/raid/ygalron/" "/mnt/c/Users/danba/"
  %(prog)s results.csv "/old/path/" "/new/path/" --output updated_results.csv
  %(prog)s results.csv "C:\\old\\" "D:\\new\\" --dry-run
  %(prog)s results.csv "/raid/" "/home/user/" --no-backup
        """
    )
    
    parser.add_argument('csv_file', help='Path to CSV file to process')
    parser.add_argument('old_pattern', help='Pattern to find and replace')
    parser.add_argument('new_pattern', help='Replacement pattern')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be changed without making changes')
    parser.add_argument('--no-backup', action='store_true',
                       help='Do not create a backup of the original file')
    parser.add_argument('--columns', nargs='+',
                       help='Specific columns to process (default: auto-detect path columns)')
    parser.add_argument('--output', '-o',
                       help='Output file path (default: auto-generate with timestamp)')
    
    args = parser.parse_args()
    
    # Check if CSV file exists
    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file '{args.csv_file}' not found.")
        sys.exit(1)
    
    # Validate patterns
    if not args.old_pattern:
        print("Error: Old pattern cannot be empty")
        sys.exit(1)
    
    if not args.new_pattern:
        print("Error: New pattern cannot be empty")
        sys.exit(1)
    
    try:
        # Create replacer and run
        replacer = CSVPathReplacer(
            csv_file=args.csv_file,
            old_pattern=args.old_pattern,
            new_pattern=args.new_pattern,
            dry_run=args.dry_run,
            backup=not args.no_backup,
            output_file=args.output
        )
        
        replacer.replace_paths()
        
    except Exception as e:
        print(f"Error during path replacement: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
