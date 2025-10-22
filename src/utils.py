"""
Utility functions for the thyroid CT project.
"""

import os
import socket
import getpass
import re
from pathlib import Path
from typing import Optional
import yaml


def get_data_path(config_path: Optional[str] = None, set_env: bool = True) -> Path:
    """
    Automatically determine and optionally set the DATA_PATH environment variable
    based on the current machine hostname and username.
    
    The function reads a YAML configuration file that maps hostname/username
    combinations to data paths. It supports regex patterns for flexible matching.
    
    Args:
        config_path: Path to the YAML configuration file. If None, uses the default
                    config/paths.yaml in the project root.
        set_env: If True, sets the DATA_PATH environment variable.
    
    Returns:
        Path: The data path as a Path object.
    
    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
        ValueError: If no matching configuration is found and no default is set.
        KeyError: If the configuration file is malformed.
    
    Example:
        >>> data_path = get_data_path()
        >>> print(f"Data path: {data_path}")
        Data path: K:\\499-ProjectData\\...
    """
    # Get current machine info
    hostname = socket.gethostname().lower()
    username = getpass.getuser().lower()
    
    # Determine config file path
    if config_path is None:
        # Assume this file is in src/ and config is in project root
        project_root = Path(__file__).parent.parent
        config_path = project_root / "config" / "paths.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Please create a paths.yaml file with hostname/username/data_path mappings."
        )
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if not config or 'paths' not in config:
        raise KeyError(
            f"Invalid configuration file: {config_path}\n"
            f"Expected a 'paths' key with a list of path configurations."
        )
    
    # Search for matching configuration
    matched_path = None
    
    for path_config in config['paths']:
        hostname_pattern = path_config.get('hostname', '').lower()
        username_pattern = path_config.get('username', '').lower()
        data_path = path_config.get('data_path')
        
        if not data_path:
            continue
        
        # Check if patterns match (supports regex)
        hostname_match = re.fullmatch(hostname_pattern, hostname)
        username_match = re.fullmatch(username_pattern, username)
        
        if hostname_match and username_match:
            matched_path = data_path
            description = path_config.get('description', 'No description')
            print(f"✓ Matched configuration: {description}")
            print(f"  Hostname: {hostname} (pattern: {hostname_pattern})")
            print(f"  Username: {username} (pattern: {username_pattern})")
            print(f"  Data path: {data_path}")
            break
    
    # Use default if no match found
    if matched_path is None:
        default_path = config.get('default_path')
        if default_path:
            print(f"⚠ No exact match found. Using default path.")
            matched_path = default_path
        else:
            raise ValueError(
                f"No matching configuration found for:\n"
                f"  Hostname: {hostname}\n"
                f"  Username: {username}\n"
                f"Please add a configuration entry in {config_path}"
            )
    
    # Convert to Path object
    data_path_obj = Path(matched_path)
    
    # Optionally set environment variable
    if set_env:
        os.environ['DATA_PATH'] = str(data_path_obj)
        print(f"✓ Set DATA_PATH environment variable")
    
    # Warn if path doesn't exist
    if not data_path_obj.exists():
        print(f"⚠ Warning: Data path does not exist: {data_path_obj}")
    
    return data_path_obj


def get_data_path_from_env() -> Optional[Path]:
    """
    Get the DATA_PATH from environment variable if it exists.
    
    Returns:
        Path object if DATA_PATH is set, None otherwise.
    
    Example:
        >>> path = get_data_path_from_env()
        >>> if path is None:
        >>>     path = get_data_path()
    """
    data_path = os.environ.get('DATA_PATH')
    if data_path:
        return Path(data_path)
    return None


def ensure_data_path(config_path: Optional[str] = None) -> Path:
    """
    Ensure DATA_PATH is set, either from environment or by auto-detection.
    
    This is a convenience function that first checks if DATA_PATH is already
    set in the environment. If not, it calls get_data_path() to determine it.
    
    Args:
        config_path: Optional path to the configuration file.
    
    Returns:
        Path: The data path as a Path object.
    
    Example:
        >>> # Use this at the start of your scripts
        >>> data_path = ensure_data_path()
        >>> volume_files = list(data_path.glob("*.tif"))
    """
    existing_path = get_data_path_from_env()
    if existing_path:
        print(f"✓ Using existing DATA_PATH: {existing_path}")
        return existing_path
    else:
        print("DATA_PATH not set. Auto-detecting...")
        return get_data_path(config_path=config_path, set_env=True)


if __name__ == "__main__":
    # Test the function
    print("=" * 80)
    print("Testing DATA_PATH configuration")
    print("=" * 80)
    
    try:
        data_path = get_data_path()
        print(f"\n✓ Success! Data path: {data_path}")
        print(f"✓ Environment variable DATA_PATH: {os.environ.get('DATA_PATH')}")
        
        if data_path.exists():
            print(f"✓ Path exists and is accessible")
            # List some files as a test
            files = list(data_path.glob("*.tif"))[:5]
            if files:
                print(f"✓ Found {len(files)} .tif files (showing first 5):")
                for f in files:
                    print(f"  - {f.name}")
        else:
            print(f"✗ Path does not exist: {data_path}")
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease check your config/paths.yaml file.")
