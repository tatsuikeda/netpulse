#!/usr/bin/env python3
"""
Network Speed Test - Compare website load times across different networks
"""

import asyncio
import aiohttp
import time
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import socket
import subprocess
import dns.resolver
from urllib.parse import urlparse
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.progress import track
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout
from dotenv import load_dotenv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np

console = Console()

class NetworkDiagnostics:
    def __init__(self):
        self.dns_times = {}
        self.ping_results = {}
        self.mtu_results = {}
        self.traceroute_results = {}
        
    def measure_dns_resolution(self, url):
        """Measure DNS resolution time for a domain."""
        domain = urlparse(url).netloc
        try:
            start_time = time.time()
            dns.resolver.resolve(domain, 'A')
            end_time = time.time()
            return end_time - start_time
        except Exception as e:
            return None

    def ping_test(self, url, count=5):
        """Run ping test to measure latency."""
        domain = urlparse(url).netloc
        try:
            cmd = ['ping', '-c', str(count), domain]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Parse ping statistics
            if result.returncode == 0:
                stats = {}
                for line in result.stdout.split('\n'):
                    if 'min/avg/max' in line:
                        # Extract min/avg/max/stddev values
                        values = line.split('=')[1].strip().split('/')
                        stats = {
                            'min': float(values[0]),
                            'avg': float(values[1]),
                            'max': float(values[2]),
                            'stddev': float(values[3].split()[0])
                        }
                return stats
        except Exception as e:
            pass
        return None

    def test_mtu_sizes(self, url, sizes=[1500, 1400, 1200]):
        """Test different packet sizes to identify MTU issues."""
        domain = urlparse(url).netloc
        results = {}
        for size in sizes:
            try:
                cmd = ['ping', '-D', '-s', str(size), '-c', '3', domain]
                result = subprocess.run(cmd, capture_output=True, text=True)
                results[size] = result.returncode == 0
            except:
                results[size] = False
        return results

    def run_traceroute(self, url):
        """Run traceroute to identify network path."""
        domain = urlparse(url).netloc
        try:
            cmd = ['traceroute', '-w', '1', domain]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return result.stdout
        except:
            return None

    def run_diagnostics(self, urls):
        """Run all network diagnostics."""
        console.print("[bold blue]Running network diagnostics...[/bold blue]")
        
        with ThreadPoolExecutor() as executor:
            # DNS resolution times
            dns_futures = {url: executor.submit(self.measure_dns_resolution, url) for url in urls}
            self.dns_times = {url: future.result() for url, future in dns_futures.items()}

            # Ping tests
            ping_futures = {url: executor.submit(self.ping_test, url) for url in urls}
            self.ping_results = {url: future.result() for url, future in ping_futures.items()}

            # MTU tests
            mtu_futures = {url: executor.submit(self.test_mtu_sizes, url) for url in urls}
            self.mtu_results = {url: future.result() for url, future in mtu_futures.items()}

            # Traceroute (optional, can be slow)
            if os.getenv('RUN_TRACEROUTE', 'false').lower() == 'true':
                trace_futures = {url: executor.submit(self.run_traceroute, url) for url in urls}
                self.traceroute_results = {url: future.result() for url, future in trace_futures.items()}

    def generate_report(self):
        """Generate network diagnostics report."""
        # DNS Resolution Table
        dns_table = Table(title="DNS Resolution Times")
        dns_table.add_column("Domain")
        dns_table.add_column("Resolution Time (s)")
        
        for url, time in self.dns_times.items():
            domain = urlparse(url).netloc
            dns_table.add_row(domain, f"{time:.3f}" if time else "Failed")

        # Ping Statistics Table
        ping_table = Table(title="Ping Statistics")
        ping_table.add_column("Domain")
        ping_table.add_column("Min (ms)")
        ping_table.add_column("Avg (ms)")
        ping_table.add_column("Max (ms)")
        ping_table.add_column("Std Dev")

        for url, stats in self.ping_results.items():
            domain = urlparse(url).netloc
            if stats:
                ping_table.add_row(
                    domain,
                    f"{stats['min']:.1f}",
                    f"{stats['avg']:.1f}",
                    f"{stats['max']:.1f}",
                    f"{stats['stddev']:.1f}"
                )
            else:
                ping_table.add_row(domain, "Failed", "Failed", "Failed", "Failed")

        # MTU Test Table
        mtu_table = Table(title="MTU Test Results")
        mtu_table.add_column("Domain")
        mtu_table.add_column("1500 bytes")
        mtu_table.add_column("1400 bytes")
        mtu_table.add_column("1200 bytes")

        for url, results in self.mtu_results.items():
            domain = urlparse(url).netloc
            mtu_table.add_row(
                domain,
                "✓" if results.get(1500) else "✗",
                "✓" if results.get(1400) else "✗",
                "✓" if results.get(1200) else "✗"
            )

        # Print all tables
        console.print(dns_table)
        console.print(ping_table)
        console.print(mtu_table)

        # Print traceroute results if available
        if self.traceroute_results:
            console.print("\n[bold]Traceroute Results:[/bold]")
            for url, trace in self.traceroute_results.items():
                domain = urlparse(url).netloc
                console.print(f"\n[bold]{domain}:[/bold]")
                if trace:
                    console.print(trace)
                else:
                    console.print("Traceroute failed")

class NetworkSpeedTest:
    def __init__(self):
        load_dotenv()
        self.websites = os.getenv('WEBSITES').split(',')
        self.iterations = int(os.getenv('ITERATIONS', 5))
        self.timeout = int(os.getenv('TIMEOUT_SECONDS', 30))
        self.concurrent_requests = int(os.getenv('CONCURRENT_REQUESTS', 3))
        self.results_dir = Path(os.getenv('RESULTS_DIR', 'results'))
        self.results_dir.mkdir(exist_ok=True)
        self.session = None
        self.results = []
        self.start_time = None
        self.end_time = None
        self.diagnostics = NetworkDiagnostics()

    async def measure_load_time(self, url):
        """Measure the load time for a single website."""
        try:
            start_time = time.time()
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=self.timeout)) as response:
                await response.read()
                end_time = time.time()
                
                return {
                    'url': url,
                    'status': response.status,
                    'load_time': end_time - start_time,
                    'timestamp': datetime.now().isoformat(),
                    'success': True,
                    'size': len(await response.read()),
                    'headers': dict(response.headers)
                }
        except Exception as e:
            return {
                'url': url,
                'status': -1,
                'load_time': self.timeout,
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': str(e),
                'size': 0,
                'headers': {}
            }

    async def run_batch(self, urls):
        """Run a batch of URL tests concurrently."""
        tasks = [self.measure_load_time(url) for url in urls]
        return await asyncio.gather(*tasks)

    async def run_tests(self):
        """Run all tests across all websites."""
        # First run network diagnostics
        self.diagnostics.run_diagnostics(self.websites)
        
        # Then run speed tests
        self.start_time = time.time()
        async with aiohttp.ClientSession() as session:
            self.session = session
            
            for iteration in track(range(self.iterations), description="Running speed tests..."):
                for i in range(0, len(self.websites), self.concurrent_requests):
                    batch = self.websites[i:i + self.concurrent_requests]
                    results = await self.run_batch(batch)
                    self.results.extend(results)
        self.end_time = time.time()

    def generate_report(self):
        """Generate a comprehensive report of the test results."""
        # First show network diagnostics
        console.print("\n[bold cyan]Network Diagnostics Report[/bold cyan]")
        self.diagnostics.generate_report()
        
        # Then show speed test results
        console.print("\n[bold cyan]Speed Test Results[/bold cyan]")
        
        df = pd.DataFrame(self.results)
        
        # Calculate statistics
        stats = df.groupby('url').agg({
            'load_time': ['mean', 'min', 'max', 'std'],
            'success': 'mean',
            'size': 'mean'
        }).round(3)
        
        # Create rich table
        table = Table(title="Website Load Time Results")
        table.add_column("Website")
        table.add_column("Avg Load Time (s)")
        table.add_column("Min (s)")
        table.add_column("Max (s)")
        table.add_column("Std Dev")
        table.add_column("Success Rate")
        table.add_column("Avg Size (KB)")

        for url in stats.index:
            row = stats.loc[url]
            table.add_row(
                url,
                f"{row['load_time']['mean']:.3f}",
                f"{row['load_time']['min']:.3f}",
                f"{row['load_time']['max']:.3f}",
                f"{row['load_time']['std']:.3f}",
                f"{row['success']['mean']*100:.1f}%",
                f"{row['size']['mean']/1024:.1f}"
            )

        # Display total time information
        total_time = self.end_time - self.start_time
        total_requests = len(self.results)
        successful_requests = df['success'].sum()
        avg_concurrent_time = total_time / (total_requests / self.concurrent_requests)
        total_data = df['size'].sum() / (1024 * 1024)  # Convert to MB

        summary = Text()
        summary.append(f"\nTotal Test Duration: {total_time:.2f} seconds\n")
        summary.append(f"Total Requests: {total_requests}\n")
        summary.append(f"Successful Requests: {successful_requests} ({(successful_requests/total_requests)*100:.1f}%)\n")
        summary.append(f"Average Time per Concurrent Batch: {avg_concurrent_time:.2f} seconds\n")
        summary.append(f"Requests per Second: {total_requests/total_time:.2f}\n")
        summary.append(f"Total Data Transferred: {total_data:.2f} MB\n")
        summary.append(f"Average Transfer Speed: {(total_data/total_time):.2f} MB/s\n")

        console.print(Panel(summary, title="Test Summary"))
        console.print(table)
        
        # Generate plots
        self.generate_plots(df)
        
        # Save results
        if os.getenv('SAVE_RESULTS', 'true').lower() == 'true':
            self.save_results(df)

    def save_results(self, df):
        """Save results to JSON and CSV files."""
        timestamp = int(time.time())
        df.to_csv(self.results_dir / f'results_{timestamp}.csv', index=False)
        
        # Convert numpy values to native Python types for JSON serialization
        def convert_to_native(obj):
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, 
                              np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.bool_)):
                return bool(obj)
            elif isinstance(obj, (np.void)):
                return None
            return obj

        # Add summary statistics to JSON output
        results_with_summary = {
            'results': self.results,
            'summary': {
                'total_time': float(self.end_time - self.start_time),
                'total_requests': int(len(self.results)),
                'successful_requests': int(df['success'].sum()),
                'requests_per_second': float(len(self.results)/(self.end_time - self.start_time)),
                'total_data_mb': float(df['size'].sum() / (1024 * 1024)),
                'timestamp': datetime.now().isoformat(),
                'dns_times': {k: convert_to_native(v) for k, v in self.diagnostics.dns_times.items()},
                'ping_results': {k: {k2: convert_to_native(v2) for k2, v2 in v.items()} if v else None 
                                for k, v in self.diagnostics.ping_results.items()},
                'mtu_results': {k: {k2: convert_to_native(v2) for k2, v2 in v.items()} 
                               for k, v in self.diagnostics.mtu_results.items()}
            }
        }
        
        with open(self.results_dir / f'results_{timestamp}.json', 'w') as f:
            json.dump(results_with_summary, f, indent=2)

    def generate_plots(self, df):
        """Generate visualization plots."""
        plt.figure(figsize=(15, 10))
        
        # Box plot of load times
        ax1 = plt.subplot(2, 2, 1)
        df.boxplot(column='load_time', by='url', rot=45)
        plt.title('Load Time Distribution by Website')
        plt.ylabel('Load Time (seconds)')
        plt.xticks(fontsize=8)
        
        # Time series plot
        ax2 = plt.subplot(2, 2, 2)
        for url in df['url'].unique():
            site_data = df[df['url'] == url]
            plt.plot(range(len(site_data)), site_data['load_time'], 
                    label=url.replace('https://', ''))
        plt.title('Load Times Over Time')
        plt.xlabel('Test Number')
        plt.ylabel('Load Time (seconds)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # Size vs Load Time scatter plot
        ax3 = plt.subplot(2, 2, 3)
        plt.scatter(df['size']/1024, df['load_time'])
        plt.xlabel('Response Size (KB)')
        plt.ylabel('Load Time (seconds)')
        plt.title('Response Size vs Load Time')
        
        # Success rate pie chart
        ax4 = plt.subplot(2, 2, 4)
        success_counts = df['success'].value_counts()
        plt.pie(success_counts, 
                labels=['Success', 'Failure'] if len(success_counts) > 1 else ['Success'],
                autopct='%1.1f%%', 
                colors=['green', 'red'] if len(success_counts) > 1 else ['green'])
        plt.title('Request Success Rate')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / f'network_analysis_{int(time.time())}.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()  # Close the figure to free memory

async def main():
    console.print("[bold green]Starting Network Speed Test[/bold green]")
    test = NetworkSpeedTest()
    await test.run_tests()
    test.generate_report()
    console.print("\n[bold green]Test completed! Results saved in the 'results' directory.[/bold green]")

if __name__ == "__main__":
    asyncio.run(main())
