# VPR Dataset Analysis Infrastructure

A comprehensive infrastructure for analyzing Visual Place Recognition (VPR) datasets based on UTM coordinates. This tool finds the top-k closest reference images to each query image using geographic proximity.

## Features

- **UTM Coordinate Parsing**: Automatically extracts UTM coordinates from filenames in format `@UTM_east@UTM_north@whatever@.jpg`
- **Top-K Nearest Neighbor Search**: Configurable number of closest reference images per query
- **CSV Output**: Structured results with query images, coordinates, and reference matches
- **Debug Visualization**: Visualize query images and their nearest neighbors
- **Advanced Plotting**: Generate comprehensive visualizations of results
- **Configurable**: Easy configuration through INI files and command line arguments

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd VPR_dataset_analysis
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.ini` to set your dataset paths and parameters:

```ini
[PATHS]
reference_dataset_path = /path/to/reference/dataset
query_dataset_path = /path/to/query/dataset
output_csv_path = results.csv

[PARAMETERS]
top_k = 5
debug_mode = false
debug_visualize_n = 3
debug_visualize_k = 3

[UTM]
utm_pattern = @(\d+\.?\d*)@(\d+\.?\d*)@.*\.jpg
```

## Usage

### Basic Usage

1. **Set paths in config.ini** or use command line arguments
2. **Run the analysis**:
```bash
python main.py
```

### Command Line Options

```bash
python main.py [OPTIONS]

Options:
  --config CONFIG_FILE    Configuration file path (default: config.ini)
  --reference REF_PATH    Reference dataset path
  --query QUERY_PATH      Query dataset path
  --output OUTPUT_PATH    Output CSV path
  --top-k K              Number of top-k nearest neighbors
  --debug                Enable debug mode (runs analysis then exits without CSV)
  --debug-only           Run only debug visualization without CSV creation
  --debug-n N            Number of queries to visualize in debug mode
  --debug-k K            Number of nearest neighbors to show in debug mode
  --create-config        Create a default configuration file and exit
```

### Examples

**Basic analysis with config file:**
```bash
python main.py
```

**Override paths and enable debug:**
```bash
python main.py --reference /path/to/ref --query /path/to/query --debug --top-k 10
```

**Debug-only mode (fastest, no CSV):**
```bash
python main.py --debug-only --debug-n 5 --debug-k 3
```

**Custom output and parameters:**
```bash
python main.py --output my_results.csv --top-k 3 --debug-n 5 --debug-k 3
```

**Create default configuration file:**
```bash
python main.py --create-config
```

## Output Format

The CSV output contains the following columns:

- `query_image_path`: Full path to the query image
- `utm_east`: UTM East coordinate of the query
- `utm_north`: UTM North coordinate of the query
- `reference_1_path`: Path to the closest reference image
- `reference_1_distance`: Distance to the closest reference image
- `reference_1_utm_east`: UTM East coordinate of the closest reference
- `reference_1_utm_north`: UTM North coordinate of the closest reference
- `reference_2_path`: Path to the second closest reference image
- `reference_2_distance`: Distance to the second closest reference image
- ... (continues for top-k references)

## Advanced Visualization

After running the analysis, you can create comprehensive visualizations:

```python
from visualization import VPRVisualizer
import pandas as pd

# Load results
results_df = pd.read_csv('results.csv')

# Create visualizer
visualizer = VPRVisualizer(results_df)

# Generate plots
visualizer.plot_utm_coordinates()
visualizer.plot_distance_distribution()
visualizer.plot_performance_metrics()

# Visualize specific queries
visualizer.visualize_query_matches(query_idx=0, k=3)

# Save all plots
visualizer.save_all_plots("vpr_plots/")
```

## File Naming Convention

Your images must follow this naming pattern for UTM coordinate extraction:

```
@UTM_east@UTM_north@whatever@.jpg
```

Examples:
- `@123456.789@987654.321@image001@.jpg`
- `@500000@600000@camera1@.jpg`
- `@123.45@456.78@sensor_data@.jpg`

## Directory Structure

```
VPR_dataset_analysis/
├── vpr_analyzer.py      # Core analysis class
├── main.py              # Main execution script
├── visualization.py      # Visualization utilities
├── config.ini           # Configuration file
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── results.csv          # Output file (generated)
```

## Performance Considerations

- **Large Datasets**: For datasets with >10,000 images, consider processing in batches
- **Memory Usage**: Distance calculations can be memory-intensive for very large datasets
- **File I/O**: Ensure your storage system can handle the file access patterns

## Troubleshooting

### Common Issues

1. **"Could not parse UTM coordinates"**: Check your filename format matches the pattern
2. **"Dataset path does not exist"**: Verify the paths in config.ini are correct
3. **Memory errors**: Reduce the number of images or process in smaller batches

### Debug Modes

There are two debug modes available:

**1. Debug Mode (--debug):**
```bash
python main.py --debug
```
This runs the full analysis with debug visualization, then exits without creating a CSV file.

**2. Debug-Only Mode (--debug-only):**
```bash
python main.py --debug-only
```
This runs only the debug visualization without CSV creation - fastest way to check your datasets.

Both debug modes will show:
- Number of images loaded from each dataset
- Sample query images and their nearest neighbors
- Processing progress and timing information

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{vpr_dataset_analysis,
  title={VPR Dataset Analysis Infrastructure},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/VPR_dataset_analysis}
}
```

