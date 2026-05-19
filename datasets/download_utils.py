import glob
import os
import zipfile
from urllib.request import urlretrieve


DEFAULT_SR_SOURCES = {
    # Public mirrors may occasionally be unavailable; failures are handled gracefully.
    'div2k_train_hr': {
        'url': 'https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip',
        'target_split': 'train_hr',
    },
    'div2k_valid_hr': {
        'url': 'https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip',
        'target_split': 'val_hr',
    },
}


def _count_images(folder):
    exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.webp')
    n = 0
    for ext in exts:
        n += len(glob.glob(os.path.join(folder, ext)))
    return n


def ensure_dataset_structure(root):
    paths = {}
    for split in ['train_hr', 'val_hr', 'test_hr']:
        p = os.path.join(root, split)
        os.makedirs(p, exist_ok=True)
        paths[split] = p
    return paths


def try_download_public_sr_data(root, timeout_note=True):
    """Try to download public SR data; return per-source status and never raise."""
    cache_dir = os.path.join(root, '_downloads')
    os.makedirs(cache_dir, exist_ok=True)

    results = {}
    for name, meta in DEFAULT_SR_SOURCES.items():
        split_dir = os.path.join(root, meta['target_split'])
        before = _count_images(split_dir)
        if before > 0:
            results[name] = {'status': 'skipped', 'reason': f'{meta["target_split"]} already has {before} images'}
            continue

        zip_path = os.path.join(cache_dir, f'{name}.zip')
        try:
            urlretrieve(meta['url'], zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(cache_dir)
            extracted_images = glob.glob(os.path.join(cache_dir, '**', '*.png'), recursive=True)
            moved = 0
            for src in extracted_images:
                dst = os.path.join(split_dir, os.path.basename(src))
                if not os.path.exists(dst):
                    os.replace(src, dst)
                    moved += 1
            after = _count_images(split_dir)
            results[name] = {'status': 'ok', 'moved': moved, 'final_images': after}
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if timeout_note and len(msg) > 180:
                msg = msg[:180] + '...'
            results[name] = {'status': 'failed', 'error': msg}

    return results


def prepare_dataset(root, auto_download=True):
    ensure_dataset_structure(root)
    download_report = try_download_public_sr_data(root) if auto_download else {}
    summary = {
        split: _count_images(os.path.join(root, split))
        for split in ['train_hr', 'val_hr', 'test_hr']
    }
    return {
        'status': 'ready',
        'root': root,
        'images': summary,
        'download_report': download_report,
        'message': (
            'Dataset prepared. If some splits are empty, place PNG/JPG files in '
            'train_hr/val_hr/test_hr manually; training/testing will still run with graceful fallbacks.'
        ),
    }
