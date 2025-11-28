"""
Launch MLflow UI to view experiment results.

This script starts the MLflow UI server on the default port (5000).
Navigate to http://127.0.0.1:5000 in your browser to view experiments.

Usage:
    python scripts/launch_mlflow_ui.py
    python scripts/launch_mlflow_ui.py --port 5001
    python scripts/launch_mlflow_ui.py --host 0.0.0.0 --port 5000
"""

import argparse
import subprocess
import sys
import webbrowser
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Launch MLflow UI for viewing experiment results"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address for MLflow UI (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port for MLflow UI (default: 5000)",
    )
    parser.add_argument(
        "--backend-store-uri",
        type=str,
        default=None,
        help="Backend store URI (default: ./mlruns)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically",
    )
    
    args = parser.parse_args()
    
    # Build command
    cmd = ["mlflow", "ui", "--host", args.host, "--port", str(args.port)]
    
    if args.backend_store_uri:
        cmd.extend(["--backend-store-uri", args.backend_store_uri])
    
    # Check if mlruns directory exists
    mlruns_dir = Path("mlruns")
    if not mlruns_dir.exists():
        print("=" * 80)
        print("Warning: 'mlruns' directory not found.")
        print("No experiments have been run yet, or tracking URI is set to a remote server.")
        print("=" * 80)
        print()
    
    # Print info
    url = f"http://{args.host}:{args.port}"
    print("=" * 80)
    print("Starting MLflow UI...")
    print("=" * 80)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"URL:  {url}")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 80)
    print()
    
    # Open browser after a short delay
    if not args.no_browser:
        def open_browser():
            time.sleep(2)  # Wait for server to start
            try:
                webbrowser.open(url)
                print(f"Opened browser to {url}")
            except Exception as e:
                print(f"Could not open browser automatically: {e}")
                print(f"Please navigate to {url} manually")
        
        import threading
        browser_thread = threading.Thread(target=open_browser, daemon=True)
        browser_thread.start()
    
    # Start MLflow UI
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n\nStopping MLflow UI...")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        print(f"\nError starting MLflow UI: {e}")
        print("\nMake sure MLflow is installed:")
        print("  pip install mlflow")
        sys.exit(1)
    except FileNotFoundError:
        print("\nError: 'mlflow' command not found.")
        print("\nMake sure MLflow is installed:")
        print("  pip install mlflow")
        sys.exit(1)


if __name__ == "__main__":
    main()
