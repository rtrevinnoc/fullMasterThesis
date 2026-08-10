import os
import subprocess
import sys

def download_data():
    dataset = "qingyi/wm811k-wafer-map"
    target_file = "LSWMD.pkl"
    
    if os.path.exists(target_file):
        print(f"{target_file} already exists. Skipping download.")
        return

    # Kaggle CLI will automatically look for ~/.kaggle/kaggle.json or KAGGLE_* env vars
    print(f"Downloading dataset {dataset}...")
    try:
        # Using absolute path for kaggle CLI in this environment
        kaggle_path = "/Users/rtrevinnoc/maestria/venv/bin/kaggle"
        subprocess.run([kaggle_path, "datasets", "download", "-d", dataset, "--unzip"], check=True)
        print("Download complete.")
    except FileNotFoundError:
        print("Error: 'kaggle' CLI not found. Please install it using 'pip install kaggle'.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")

if __name__ == "__main__":
    download_data()
