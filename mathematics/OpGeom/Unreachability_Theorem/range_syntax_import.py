import requests
import pandas as pd
import time

def fetch_informed_data_v2(conductor_range="1-1000", rank_limit=2):
    """
    Revised method using correct API type prefixes (ls:, i:) and 
    standardized field names to retrieve heights from ec_mwbsd.
    """
    base_url = "https://www.lmfdb.org/api/ec_curves/"
    mw_url = "https://www.lmfdb.org/api/ec_mwbsd/"
    
    all_data = []
    
    for rank in range(1, rank_limit + 1):
        # API requires 'i' prefix for integers
        params = {
            "conductor": conductor_range,
            "rank": f"i{rank}",
            "_format": "json",
            "_limit": 100
        }
        
        print(f"Fetching Rank {rank} curves in range {conductor_range}...")
        try:
            response = requests.get(base_url, params=params)
            curves = response.json().get('data', [])
            if not curves:
                continue

            # Extract labels and format as a comma-separated list with 'ls:' prefix
            labels = [c['lmfdb_label'] for c in curves]
            label_query = "ls:" + ",".join(labels)
            
            # Query ec_mwbsd using the 'label' field
            mw_params = {
                "label": label_query,
                "_format": "json"
            }
            
            mw_response = requests.get(mw_url, params=mw_params)
            mw_results = mw_response.json().get('data', [])
            
            # Map heights back to curve data
            mw_lookup = {row['label']: row.get('heights', []) for row in mw_results}
            
            for c in curves:
                label = c['lmfdb_label']
                heights = mw_lookup.get(label, [])
                if heights:
                    all_data.append({
                        'label': label,
                        'absD': float(c.get('absD', 0)),
                        'rank': rank,
                        'min_height': min(heights),
                        'conductor': int(c.get('conductor', 0))
                    })
            
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error at Rank {rank}: {e}")
            
    return pd.DataFrame(all_data)

# Execution
df = fetch_informed_data_v2(conductor_range="1-1000")
if not df.empty:
    print(f"Successfully fetched {len(df)} curves with heights.")
    print(df.head())
else:
    print("No data found. Check if the conductor range is valid for the API.")