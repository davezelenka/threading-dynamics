#!/usr/bin/env python3
"""
Query LMFDB PostgreSQL Database Directly
More reliable than API, no file size limits

Connection details:
- Host: lmfdb.warwick.ac.uk
- Database: lmfdb
- User: lmfdb (read-only)
- Password: (none required for read-only access)
"""

import psycopg2
import pandas as pd
import numpy as np
from psycopg2.extras import RealDictCursor
import sys

# ============================================================================
# LMFDB PostgreSQL CONNECTOR
# ============================================================================

class LMFDBPostgres:
    """
    Connect to LMFDB PostgreSQL database and fetch elliptic curves.
    """
    
    def __init__(self, host="lmfdb.warwick.ac.uk", 
                 database="lmfdb",
                 user="lmfdb",
                 password=""):
        """
        Initialize connection.
        
        Args:
            host: PostgreSQL host
            database: Database name
            user: Username (read-only)
            password: Password (empty for read-only)
        """
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.conn = None
    
    def connect(self):
        """
        Establish connection to LMFDB database.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"Connecting to {self.host}:{self.database}...")
            
            self.conn = psycopg2.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password,
                connect_timeout=10
            )
            
            print("✓ Connected successfully!")
            return True
        
        except psycopg2.OperationalError as e:
            print(f"✗ Connection failed: {e}")
            return False
        
        except Exception as e:
            print(f"✗ Error: {e}")
            return False
    
    def fetch_curves(self, conductor_max=10000, limit=None):
        """
        Fetch elliptic curves from database.
        
        Args:
            conductor_max: Maximum conductor to fetch
            limit: Maximum number of curves (None = no limit)
        
        Returns:
            List of curve dictionaries
        """
        if not self.conn:
            print("Not connected to database")
            return []
        
        try:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            
            # Query elliptic curves table
            # Table: ec_curves
            # Columns: label, conductor, discriminant, rank, torsion, a1, a2, a3, a4, a6
            
            query = """
            SELECT 
                label,
                conductor,
                discriminant,
                rank,
                torsion,
                a1, a2, a3, a4, a6
            FROM ec_curves
            WHERE conductor <= %s
            ORDER BY conductor, label
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            print(f"\nQuerying curves with conductor <= {conductor_max}...")
            cursor.execute(query, (conductor_max,))
            
            # Fetch in batches
            curves = []
            batch_size = 1000
            
            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break
                
                curves.extend([dict(row) for row in rows])
                print(f"  Fetched {len(curves)} curves...", flush=True)
            
            cursor.close()
            
            print(f"\n✓ Successfully fetched {len(curves)} curves")
            return curves
        
        except Exception as e:
            print(f"✗ Query failed: {e}")
            return []
    
    def fetch_curves_with_heights(self, conductor_max=10000, rank_min=0, rank_max=4):
        """
        Fetch curves with height information.
        
        Note: Heights may be in a separate table (ec_nfcurves or similar)
        This query attempts to get basic curve data.
        
        Args:
            conductor_max: Maximum conductor
            rank_min: Minimum rank
            rank_max: Maximum rank
        
        Returns:
            List of curve dictionaries
        """
        if not self.conn:
            print("Not connected to database")
            return []
        
        try:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            
            query = """
            SELECT 
                label,
                conductor,
                discriminant,
                rank,
                torsion,
                a1, a2, a3, a4, a6
            FROM ec_curves
            WHERE conductor <= %s
              AND rank >= %s
              AND rank <= %s
            ORDER BY conductor, label
            """
            
            print(f"\nQuerying curves with:")
            print(f"  Conductor <= {conductor_max}")
            print(f"  Rank in [{rank_min}, {rank_max}]")
            
            cursor.execute(query, (conductor_max, rank_min, rank_max))
            
            curves = []
            batch_size = 1000
            
            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break
                
                curves.extend([dict(row) for row in rows])
                print(f"  Fetched {len(curves)} curves...", flush=True)
            
            cursor.close()
            
            print(f"\n✓ Successfully fetched {len(curves)} curves")
            return curves
        
        except Exception as e:
            print(f"✗ Query failed: {e}")
            return []
    
    def close(self):
        """
        Close database connection.
        """
        if self.conn:
            self.conn.close()
            print("Connection closed")


# ============================================================================
# COMPUTE HEIGHTS FROM WEIERSTRASS COEFFICIENTS
# ============================================================================

def compute_discriminant_from_coefficients(a1, a2, a3, a4, a6):
    """
    Compute discriminant from Weierstrass coefficients.
    
    For curve: y^2 + a1*x*y + a3*y = x^3 + a2*x^2 + a4*x + a6
    
    Args:
        a1, a2, a3, a4, a6: Weierstrass coefficients
    
    Returns:
        Discriminant value
    """
    b2 = a1**2 + 4*a2
    b4 = a1*a3 + 2*a4
    b6 = a3**2 + 4*a6
    b8 = a1**2*a6 - a1*a3*a4 + a2*a3**2 - a4**2 + a2*a4*a1 - a2**2*a3
    
    discriminant = -b2**2*b8 - 8*b4**3 + 9*b2*b4*b6 - 27*b6**2
    
    return discriminant


def estimate_height_from_coefficients(a1, a2, a3, a4, a6):
    """
    Estimate canonical height from Weierstrass coefficients.
    
    Rough approximation: h ~ log(max(|a_i|))
    
    Args:
        a1, a2, a3, a4, a6: Weierstrass coefficients
    
    Returns:
        Estimated height
    """
    max_coeff = max(abs(a1), abs(a2), abs(a3), abs(a4), abs(a6), 1)
    h_est = np.log(max_coeff) / 2  # Rough estimate
    return max(h_est, 0.01)  # Minimum height


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Main execution: connect to LMFDB, fetch curves, analyze.
    """
    print("="*70)
    print("Fetching Real Elliptic Curves from LMFDB PostgreSQL")
    print("="*70)
    
    # Connect to database
    db = LMFDBPostgres()
    
    if not db.connect():
        print("\n✗ Could not connect to LMFDB database")
        print("\nAlternative: Manual download from LMFDB")
        print("1. Visit: https://www.lmfdb.org/download/")
        print("2. Download elliptic curves CSV (smaller conductor ranges)")
        print("3. Save as: lmfdb_curves.csv")
        return None
    
    # Fetch curves
    curves = db.fetch_curves_with_heights(
        conductor_max=10000,
        rank_min=0,
        rank_max=4
    )
    
    db.close()
    
    if len(curves) == 0:
        print("\n✗ No curves fetched")
        return None
    
    # Process curves
    print(f"\nProcessing {len(curves)} curves...")
    
    processed = []
    for i, curve in enumerate(curves):
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{len(curves)}...", flush=True)
        
        try:
            # Verify discriminant
            disc_computed = compute_discriminant_from_coefficients(
                curve['a1'], curve['a2'], curve['a3'], curve['a4'], curve['a6']
            )
            
            # Estimate height
            h_est = estimate_height_from_coefficients(
                curve['a1'], curve['a2'], curve['a3'], curve['a4'], curve['a6']
            )
            
            processed.append({
                'label': curve['label'],
                'conductor': curve['conductor'],
                'discriminant': curve['discriminant'],
                'discriminant_computed': disc_computed,
                'rank': curve['rank'],
                'torsion': curve['torsion'],
                'h_min_est': h_est,
            })
        
        except Exception as e:
            continue
    
    # Create DataFrame
    df = pd.DataFrame(processed)
    
    # Summary statistics
    print(f"\n{'='*70}")
    print("DATA SUMMARY")
    print(f"{'='*70}")
    print(f"\nTotal curves: {len(df)}")
    print(f"\nRank distribution:")
    print(df['rank'].value_counts().sort_index())
    print(f"\nConductor range: [{df['conductor'].min()}, {df['conductor'].max()}]")
    print(f"Discriminant range: [{df['discriminant'].min()}, {df['discriminant'].max()}]")
    print(f"Estimated height range: [{df['h_min_est'].min():.6f}, {df['h_min_est'].max():.6f}]")
    
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