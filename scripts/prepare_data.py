import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import argparse
import json
from datasets.download_utils import prepare_dataset

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='./data')
    ap.add_argument('--no-download', action='store_true', help='Only create folder structure; do not try public downloads.')
    ap.add_argument('--strict-download', action='store_true', help='Exit non-zero only if benchmark auto-download warning occurs.')
    args = ap.parse_args()
    result = prepare_dataset(args.root, auto_download=not args.no_download)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.strict_download and result.get('benchmark_report', {}).get('status') == 'warning_unavailable_network':
        raise SystemExit(2)
