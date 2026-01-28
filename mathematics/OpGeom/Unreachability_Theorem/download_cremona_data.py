#!/usr/bin/env python3
"""
Download Elliptic Curve Data from Cremona's Database
More reliable than LMFDB API - direct file download
"""

import requests
import gzip
import io
import pandas as pd
import numpy as np
from pathlib import Path
import time

# ============================================================================
# CREMONA DATABASE DOWNLOADER
# ============================================================================

class CremonaDownloader:
    """
    Download elliptic curves from Cremona's database.
    
    Source: http://www.warwick.ac.uk/~masgaj/ftp/data/
    Contains all curves with conductor <= 500,000
    """
    
    def __init__(self, base_url="http://www.warwick.ac.uk/~masgaj/ftp/data/"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Educational Research)'
        })
    
    def download_allcurves(self, conductor_max=10000):
        """
        Download allcurves file for given conductor range.
        
        Files available:
        - allcurves.00000-09999.gz (conductor 0-9999)
        - allcurves.10000-99999.gz (conductor 10000-99999)
        - allcurves.100000-499999.gz (conductor 100000-499999)
        
        Args:
            conductor_max: Maximum conductor to download
        
        Returns:
            List of curve data dictionaries
        """
        print("="*70)
        print("Downloading Elliptic Curves from Cremona's Database")
        print("="*70)
        
        all_curves = []
        
        # Determine which files to download
        files_to_download = []
        
        if conductor_max >= 10000:
            files_to_download.append("allcurves.00000-09999.gz")
        else:
            files_to_download.append("allcurves.00000-09999.gz")
        
        if conductor_max >= 100000:
            files_to_download.append("allcurves.10000-99999.gz")
        
        if conductor_max >= 500000:
            files_to_download.append("allcurves.100000-499999.gz")
        
        # Download and parse each file
        for filename in files_to_download:
            print(f"\nDownloading {filename}...")
            url = self.base_url + filename
            
            try:
                response = self.session.get(url, timeout=60, stream=True)
                response.raise_for_status()
                
                # Decompress gzip
                print(f"Decompressing...")
                with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as gz:
                    content = gz.read().decode('utf-8')
                
                # Parse lines
                print(f"Parsing curves...")
                curves = self._parse_allcurves(content, conductor_max)
                all_curves.extend(curves)
                print(f"  Found {len(curves)} curves")
            
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        return all_curves
    
    def _parse_allcurves(self, content, conductor_max):
        """
        Parse allcurves file format.
        
        Format (space-separated):
        conductor isogeny_class curve_number a1 a2 a3 a4 a6 rank torsion_order
        
        Example:
        11 a 1 0 -1 1 -10 -20 0 5
        
        Args:
            content: File content as string
            conductor_max: Maximum conductor to include
        
        Returns:
            List of curve dictionaries
        """
        curves = []
        
        for line_num, line in enumerate(content.strip().split('\n')):
            if not line.strip():
                continue
            
            try:
                parts = line.split()
                
                if len(parts) < 10:
                    continue
                
                conductor = int(parts[0])
                
                # Skip if conductor exceeds max
                if conductor > conductor_max:
                    continue
                
                isogeny_class = parts[1]
                curve_number = int(parts[2])
                a1 = int(parts[3])
                a2 = int(parts[4])
                a3 = int(parts[5])
                a4 = int(parts[6])
                a6 = int(parts[7])
                rank = int(parts[8])
                torsion = int(parts[9])
                
                # Compute discriminant from Weierstrass coefficients
                # For y^2 + a1*x*y + a3*y = x^3 + a2*x^2 + a4*x + a6
                # Δ = -4*a2^3*a6 + a2^2*a4^2 + 18*a1*a2*a4*a6 - 4*a4^3 - 27*a6^2 + a1^2*a3^2 - a1^4*a6 + a3^2*a4 - a1^2*a2*a4 + a1^3*a3*a6
                # Simplified for a1=a3=0: Δ = -4*a2^3*a6 + a2^2*a4^2 - 4*a4^3 - 27*a6^2
                
                b2 = a1**2 + 4*a2
                b4 = a1*a3 + 2*a4
                b6 = a3**2 + 4*a6
                b8 = a1**2*a6 - a1*a3*a4 + a2*a3**2 - a4**2 + a2*a4*a1 - a2**2*a3
                
                discriminant = -b2**2*b8 - 8*b4**3 + b2*b4*b6*9 - b6**2*27
                
                label = f"{conductor}{isogeny_class}{curve_number}"
                
                curves.append({
                    'label': label,
                    'conductor': conductor,
                    'discriminant': discriminant,
                    'rank': rank,
                    'torsion': torsion,
                    'a1': a1,
                    'a2': a2,
                    'a3': a3,
                    'a4': a4,
                    'a6': a6,
                })
            
            except Exception as e:
                if line_num < 10:  # Only print first few errors
                    print(f"  Warning: Could not parse line {line_num}: {e}")
                continue
        
        return curves


# ============================================================================
# ALTERNATIVE: LMFDB CSV DOWNLOAD
# ============================================================================

def download_lmfdb_csv():
    """
    Download LMFDB elliptic curves as CSV.
    
    LMFDB provides downloadable data at:
    https://www.lmfdb.org/download/
    
    This function provides instructions for manual download.
    """
    print("="*70)
    print("LMFDB CSV Download Instructions")
    print("="*70)
    print()
    print("LMFDB provides downloadable CSV files:")
    print()
    print("1. Visit: https://www.lmfdb.org/download/")
    print("2. Look for 'Elliptic curves' section")
    print("3. Download: 'Elliptic curves (conductor <= 10000)'")
    print("4. Extract the CSV file")
    print("5. Place in current directory as 'lmfdb_curves.csv'")
    print()
    print("Then run: python analyze_real_data.py")
    print()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Download real elliptic curve data.
    """
    print("\nAttempting to download from Cremona's database...\n")
    
    downloader = CremonaDownloader()
    
    try:
        # Download curves with conductor <= 10000
        curves = downloader.download_allcurves(conductor_max=10000)
        
        if len(curves) == 0:
            print("\nNo curves downloaded. Trying alternative approach...")
            download_lmfdb_csv()
            return None
        
        # Create DataFrame
        df = pd.DataFrame(curves)
        
        # Summary statistics
        print(f"\n{'='*70}")
        print("DATA SUMMARY")
        print(f"{'='*70}")
        print(f"\nTotal curves: {len(df)}")
        print(f"\nRank distribution:")
        print(df['rank'].value_counts().sort_index())
        print(f"\nConductor range: [{df['conductor'].min()}, {df['conductor'].max()}]")
        print(f"Discriminant range: [{df['discriminant'].min()}, {df['discriminant'].max()}]")
        
        # Save to CSV
        output_file = 'cremona_real_curves.csv'
        df.to_csv(output_file, index=False)
        print(f"\nData saved to: {output_file}")
        
        print(f"\n{'='*70}")
        print("Download complete!")
        print(f"{'='*70}")
        
        return df
    
    except Exception as e:
        print(f"\nError: {e}")
        print("\nFalling back to manual download instructions...\n")
        download_lmfdb_csv()
        return None


if __name__ == '__main__':
    df = main()