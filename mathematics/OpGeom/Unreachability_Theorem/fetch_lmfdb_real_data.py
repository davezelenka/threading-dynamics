#!/usr/bin/env python3
"""
Fetch Real Elliptic Curve Data from LMFDB
Corrected API queries with proper error handling and rate limiting
"""

import requests
import json
import time
import pandas as pd
from urllib.parse import quote
import sys

# ============================================================================
# LMFDB API FETCHER (CORRECTED)
# ============================================================================

class LMFDBFetcher:
    """
    Fetch elliptic curves from LMFDB with proper error handling.
    """
    
    def __init__(self, base_url="https://www.lmfdb.org/api/ec_curves/", 
                 timeout=30, max_retries=3, delay=1.0):
        """
        Initialize fetcher.
        
        Args:
            base_url: LMFDB API base URL
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            delay: Delay between requests in seconds
        """
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Educational Research)'
        })
    
    def fetch_by_conductor_range(self, min_conductor=1, max_conductor=1000, 
                                 limit=100):
        """
        Fetch curves by conductor range.
        
        Args:
            min_conductor: Minimum conductor
            max_conductor: Maximum conductor
            limit: Results per request
        
        Returns:
            List of curve dictionaries
        """
        print(f"Fetching curves with conductor in [{min_conductor}, {max_conductor}]...")
        
        all_curves = []
        offset = 0
        
        while True:
            # Build query
            query = {
                'conductor': {'$gte': min_conductor, '$lte': max_conductor},
                '_format': 'json',
                '_limit': limit,
                '_offset': offset,
            }
            
            # Convert to URL-encoded query string
            query_str = self._build_query_string(query)
            url = self.base_url + "?" + query_str
            
            print(f"  Fetching offset {offset}...", end=' ', flush=True)
            
            # Fetch with retries
            data = self._fetch_with_retry(url)
            
            if data is None:
                print("FAILED")
                break
            
            if 'data' not in data or len(data['data']) == 0:
                print(f"Done ({len(all_curves)} total)")
                break
            
            curves = data['data']
            all_curves.extend(curves)
            print(f"Got {len(curves)} curves")
            
            # Check if we got fewer results than requested (end of data)
            if len(curves) < limit:
                print(f"Reached end of data ({len(all_curves)} total)")
                break
            
            offset += limit
            time.sleep(self.delay)  # Rate limiting
        
        return all_curves
    
    def fetch_by_rank(self, rank, conductor_max=1000, limit=100):
        """
        Fetch curves by rank.
        
        Args:
            rank: Rank to fetch
            conductor_max: Maximum conductor
            limit: Results per request
        
        Returns:
            List of curve dictionaries
        """
        print(f"Fetching rank {rank} curves with conductor <= {conductor_max}...")
        
        all_curves = []
        offset = 0
        
        while True:
            query = {
                'rank': rank,
                'conductor': {'$lte': conductor_max},
                '_format': 'json',
                '_limit': limit,
                '_offset': offset,
            }
            
            query_str = self._build_query_string(query)
            url = self.base_url + "?" + query_str
            
            print(f"  Fetching offset {offset}...", end=' ', flush=True)
            
            data = self._fetch_with_retry(url)
            
            if data is None:
                print("FAILED")
                break
            
            if 'data' not in data or len(data['data']) == 0:
                print(f"Done ({len(all_curves)} total)")
                break
            
            curves = data['data']
            all_curves.extend(curves)
            print(f"Got {len(curves)} curves")
            
            if len(curves) < limit:
                print(f"Reached end of data ({len(all_curves)} total)")
                break
            
            offset += limit
            time.sleep(self.delay)
        
        return all_curves
    
    def _build_query_string(self, query_dict):
        """
        Build URL query string from dictionary.
        
        Args:
            query_dict: Dictionary with query parameters
        
        Returns:
            URL-encoded query string
        """
        parts = []
        
        for key, value in query_dict.items():
            if isinstance(value, dict):
                # Handle MongoDB-style operators
                json_str = json.dumps(value)
                parts.append(f"{key}={quote(json_str)}")
            else:
                parts.append(f"{key}={quote(str(value))}")
        
        return "&".join(parts)
    
    def _fetch_with_retry(self, url):
        """
        Fetch URL with retry logic.
        
        Args:
            url: URL to fetch
        
        Returns:
            JSON data or None if failed
        """
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, timeout=self.timeout)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Rate limited
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Rate limited, waiting {wait_time}s...", end=' ', flush=True)
                    time.sleep(wait_time)
                    continue
                elif response.status_code == 404:
                    print(f"Not found")
                    return None
                else:
                    print(f"Error {response.status_code}")
                    return None
            
            except requests.exceptions.Timeout:
                print(f"Timeout (attempt {attempt+1}/{self.max_retries})", end=' ', flush=True)
                time.sleep(2 ** attempt)
                continue
            
            except requests.exceptions.ConnectionError:
                print(f"Connection error (attempt {attempt+1}/{self.max_retries})", end=' ', flush=True)
                time.sleep(2 ** attempt)
                continue
            
            except Exception as e:
                print(f"Error: {e}")
                return None
        
        return None


# ============================================================================
# DATA PARSING
# ============================================================================

def parse_curve(curve_data):
    """
    Parse LMFDB curve data.
    
    Args:
        curve_data: Dictionary from LMFDB API
    
    Returns:
        Parsed dictionary or None
    """
    try:
        # Extract fields
        label = curve_data.get('label')
        conductor = curve_data.get('conductor')
        discriminant = curve_data.get('discriminant')
        rank = curve_data.get('rank')
        
        # Heights (canonical heights of generators)
        heights = curve_data.get('heights', [])
        
        if not heights or len(heights) == 0:
            return None
        
        h_min = min(heights)
        
        if h_min is None or conductor is None or discriminant is None:
            return None
        
        return {
            'label': label,
            'conductor': conductor,
            'discriminant': discriminant,
            'rank': rank,
            'h_min': h_min,
            'num_generators': len(heights),
            'heights': heights,
        }
    
    except Exception as e:
        return None


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Fetch real elliptic curve data from LMFDB.
    """
    print("="*70)
    print("Fetching Real Elliptic Curve Data from LMFDB")
    print("="*70)
    
    # Initialize fetcher
    fetcher = LMFDBFetcher(delay=0.5)  # 0.5s delay between requests
    
    # Fetch curves by conductor range
    print("\nStrategy: Fetch by conductor range (more reliable than rank)")
    print()
    
    all_curves = []
    
    # Fetch in chunks to avoid overwhelming the API
    conductor_ranges = [
        (1, 100),
        (101, 500),
        (501, 1000),
        (1001, 5000),
        (5001, 10000),
    ]
    
    for min_c, max_c in conductor_ranges:
        curves = fetcher.fetch_by_conductor_range(min_c, max_c, limit=50)
        all_curves.extend(curves)
        print()
    
    print(f"\nTotal curves fetched: {len(all_curves)}")
    
    # Parse curves
    print(f"\nParsing curves...")
    parsed_curves = []
    
    for i, curve in enumerate(all_curves):
        if (i + 1) % 100 == 0:
            print(f"  Parsed {i+1}/{len(all_curves)}...", flush=True)
        
        parsed = parse_curve(curve)
        if parsed:
            parsed_curves.append(parsed)
    
    print(f"Successfully parsed: {len(parsed_curves)} curves")
    
    # Create DataFrame
    df = pd.DataFrame(parsed_curves)
    
    # Summary statistics
    print(f"\n{'='*70}")
    print("DATA SUMMARY")
    print(f"{'='*70}")
    print(f"\nTotal curves: {len(df)}")
    print(f"\nRank distribution:")
    print(df['rank'].value_counts().sort_index())
    print(f"\nConductor range: [{df['conductor'].min()}, {df['conductor'].max()}]")
    print(f"Discriminant range: [{df['discriminant'].min()}, {df['discriminant'].max()}]")
    print(f"Height range: [{df['h_min'].min():.6f}, {df['h_min'].max():.6f}]")
    
    # Save to CSV
    output_file = 'lmfdb_real_curves.csv'
    df.to_csv(output_file, index=False)
    print(f"\nData saved to: {output_file}")
    
    print(f"\n{'='*70}")
    print("Fetch complete!")
    print(f"{'='*70}")
    
    return df


if __name__ == '__main__':
    df = main()