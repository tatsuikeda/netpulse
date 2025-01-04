# NetPulse

A comprehensive network diagnostics and speed testing tool that helps you analyze and troubleshoot network performance issues. NetPulse combines multiple diagnostic tools into a single, easy-to-use interface, providing detailed insights into your network's health and performance.

## Features

- **Speed Testing**: Measure load times across multiple websites concurrently
- **Network Diagnostics**:
  - DNS Resolution Time Analysis
  - Ping Tests with detailed statistics
  - MTU Size Testing
  - Optional Traceroute Analysis
- **Comprehensive Reporting**:
  - Detailed performance metrics
  - Visual analysis with charts and graphs
  - Markdown reports with full analysis
  - JSON and CSV exports
- **Real-time Monitoring**:
  - Success rate tracking
  - Data transfer statistics
  - Response size analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/tatsuikeda/netpulse.git
cd netpulse

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root (or copy the example):

```ini
# List of websites to test (comma-separated)
WEBSITES=https://www.google.com,https://www.amazon.com
ITERATIONS=5
TIMEOUT_SECONDS=30
CONCURRENT_REQUESTS=3
SAVE_RESULTS=true
RESULTS_DIR=results
RUN_TRACEROUTE=false
```

## Usage

```bash
python netpulse.py
```

The tool will:
1. Run network diagnostics (DNS, ping, MTU tests)
2. Perform speed tests on specified websites
3. Generate detailed reports and visualizations
4. Save results to the specified directory

## Output Files

NetPulse generates comprehensive reports and data files in the `results` directory:

- **Markdown Report** (`report_<timestamp>.md`):
  - Complete test results in readable format
  - Network diagnostics summary
  - Speed test analysis
  - Links to visualizations and raw data
- **Visualization** (`network_analysis_<timestamp>.png`):
  - Load time distribution plots
  - Time series analysis
  - Size vs. load time correlation
  - Success rate visualization
- **Raw Data**:
  - CSV file (`results_<timestamp>.csv`): Detailed test results
  - JSON file (`results_<timestamp>.json`): Complete data including diagnostics

## License

MIT License - see [LICENSE](LICENSE) for details

## Author

Tatsu Ikeda

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
