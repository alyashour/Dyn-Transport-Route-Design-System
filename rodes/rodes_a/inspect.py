"""
I use this to look into npy files.
"""

#!/usr/bin/env python3
import sys
import numpy as np

def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_npy.py <file.npy>")
        return

    path = sys.argv[1]

    print(f"Loading: {path}")
    try:
        arr = np.load(path, allow_pickle=True)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print("\n=== Basic Info ===")
    print("Type:", type(arr))

    if isinstance(arr, np.ndarray):
        print("Shape:", arr.shape)
        print("Dtype:", arr.dtype)

        # If it’s numeric, print stats
        if np.issubdtype(arr.dtype, np.number):
            print("\n=== Stats ===")
            try:
                print("Min:", arr.min())
                print("Max:", arr.max())
                print("Mean:", arr.mean())
            except Exception:
                pass

        print("\n=== Sample Values ===")
        # Print up to a small slice safely
        try:
            sample = arr
            for _ in range(max(0, arr.ndim - 1)):
                sample = sample[0]
            print(sample[:10] if hasattr(sample, "__getitem__") else sample)
        except Exception:
            print("(Could not extract sample values)")

    else:
        print("\n(Not a NumPy array — maybe a Python object?)")
        print("Value:", arr)

if __name__ == "__main__":
    main()
