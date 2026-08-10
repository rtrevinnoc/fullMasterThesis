import pandas as pd
import numpy as np
import sys

# Monkeypatch for compatibility with older pickle files
import pandas.core.indexes
sys.modules['pandas.indexes'] = pandas.core.indexes

file_path = 'research/nn/cnn/LSWMD.pkl'

import pickle

print(f"Loading {file_path}...")
with open(file_path, 'rb') as f:
    data = pickle.load(f, encoding='latin1')

if isinstance(data, pd.DataFrame):
    df = data
else:
    df = pd.DataFrame(data)

# 2. Identify the columns
print("\n1. Columns in the dataset:")
print(df.columns.tolist())

# 3. Check the unique values of failureType
print("\n2. Unique values in 'failureType':")
# Often in LSWMD, failureType is a list of lists, e.g., [['none']]
def extract_failure_type(x):
    if isinstance(x, (list, np.ndarray)):
        if len(x) > 0 and isinstance(x[0], (list, np.ndarray)):
            if len(x[0]) > 0:
                return str(x[0][0])
            else:
                return 'empty_inner'
        elif len(x) > 0:
            return str(x[0])
        else:
            return 'empty'
    return str(x)

df['failureType_clean'] = df['failureType'].apply(extract_failure_type)
print(df['failureType_clean'].unique())

# 4. Check the dimensions of the waferMap arrays
print("\n3. Dimensions of the 'waferMap' arrays (first 5 samples):")
for i in range(min(5, len(df))):
    wm = df['waferMap'].iloc[i]
    if hasattr(wm, 'shape'):
        print(f"Sample {i} shape: {wm.shape}")
    else:
        print(f"Sample {i} is not a numpy array, type: {type(wm)}")

# 5. Summary of labeled vs "none"
print("\n4. Summary of failure types:")
summary = df['failureType_clean'].value_counts()
print(summary)

total = len(df)
none_count = summary.get('none', 0)
empty_count = summary.get('empty', 0)
actual_failures = total - none_count - empty_count

print(f"\nTotal samples: {total}")
print(f"Samples with actual failure labels (Edge-Ring, Loc, etc.): {actual_failures}")
print(f"Samples with 'none' label (normal wafers): {none_count}")
print(f"Samples that are 'empty' (unlabeled): {empty_count}")
print(f"Total labeled samples (Failures + None): {actual_failures + none_count}")
