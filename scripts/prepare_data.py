import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import argparse
from datasets.download_utils import ensure_dataset_structure

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='./data')
    args = ap.parse_args()
    print(ensure_dataset_structure(args.root))
