import xml.etree.ElementTree as ET
import re

def extract_path_data(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    # Find the path with many points - likely the data line
    paths = root.findall('.//{http://www.w3.org/2000/svg}path')
    data_paths = []
    for p in paths:
        d = p.get('d', '')
        # Count L (line to) commands
        if d.count('L') > 50:
            data_paths.append(d)
    
    return data_paths

path_data = extract_path_data("panel_g_intercept_trajectory_transformation_rate[@name='exhausted_T_cell']_0_16.svg")
if path_data:
    # Parse one of them
    d = path_data[0]
    points = []
    for m in re.finditer(r'([ML])\s+([\d.]+)\s+([\d.]+)', d):
        points.append((float(m.group(2)), float(m.group(3))))
    
    print(f"Extracted {len(points)} points")
    # Save to a temporary pickle for generate_figure_6.py to use
    import pickle
    import numpy as np
    
    # We need to normalize these points back to data space if possible, 
    # but for assembly we can just plot them as is if we have the axes limits.
    # Actually, better to just load them and plot them.
    with open('data/fig_6g_recovered.pkl', 'wb') as f:
        pickle.dump({'points': points}, f)
else:
    print("No data path found")
