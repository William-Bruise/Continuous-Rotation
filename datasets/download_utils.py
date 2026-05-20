import glob
import os
import tarfile
import zipfile
from urllib.request import urlretrieve

DEFAULT_SR_SOURCES = {
    'div2k_train_hr': {
        'url': 'https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip',
        'target_split': 'train_hr',
    },
    'div2k_valid_hr': {
        'url': 'https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip',
        'target_split': 'val_hr',
    },
}

BENCHMARK_DATASETS = ['Set5', 'Set14', 'BSD100', 'Urban100']
BENCHMARK_URLS = [
    'https://cv.snu.ac.kr/research/EDSR/benchmark.tar',
    'https://www.dropbox.com/s/zg8wz5n0kud3u8v/benchmark.tar?dl=1',
    'https://huggingface.co/datasets/eugenesiow/Set5/resolve/main/Set5.zip',
]


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


def ensure_benchmark_structure(root):
    base = os.path.join(root, 'benchmarks')
    paths = {}
    for name in BENCHMARK_DATASETS:
        p = os.path.join(base, name, 'HR')
        os.makedirs(p, exist_ok=True)
        paths[name] = p
    return paths


def try_download_public_sr_data(root, timeout_note=True):
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
            results[name] = {'status': 'ok', 'moved': moved, 'final_images': _count_images(split_dir)}
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if timeout_note and len(msg) > 180:
                msg = msg[:180] + '...'
            results[name] = {'status': 'failed', 'error': msg}
    return results


def try_download_benchmarks(root):
    base = os.path.join(root, 'benchmarks')
    cache_dir = os.path.join(root, '_downloads')
    os.makedirs(cache_dir, exist_ok=True)
    if sum(_count_images(os.path.join(base, d, 'HR')) for d in BENCHMARK_DATASETS) > 0:
        return {'status': 'skipped', 'reason': 'benchmark images already present'}

    # Allow local offline package import first (for restricted servers)
    local_pack = os.path.join(cache_dir, 'benchmark.tar')
    if os.path.exists(local_pack):
        try:
            with tarfile.open(local_pack, 'r') as tf:
                tf.extractall(cache_dir)
        except Exception:
            pass

    attempts = []
    downloaded = False
    archive_path = None
    for url in BENCHMARK_URLS:
        try:
            ext = '.zip' if url.lower().endswith('.zip') else '.tar'
            archive_path = os.path.join(cache_dir, f'benchmark_auto{ext}')
            urlretrieve(url, archive_path)
            downloaded = True
            break
        except Exception as e:  # noqa: BLE001
            attempts.append(f'{url} -> {e}')

    extracted_any = False
    if downloaded and archive_path:
        try:
            if archive_path.endswith('.zip'):
                with zipfile.ZipFile(archive_path, 'r') as zf:
                    zf.extractall(cache_dir)
            else:
                with tarfile.open(archive_path, 'r') as tf:
                    tf.extractall(cache_dir)
        except Exception as e:  # noqa: BLE001
            attempts.append(f'extract {archive_path} -> {e}')

    report = {}
    for d in BENCHMARK_DATASETS:
        src_candidates = glob.glob(os.path.join(cache_dir, '**', d, '**', '*'), recursive=True)
        dst_dir = os.path.join(base, d, 'HR')
        os.makedirs(dst_dir, exist_ok=True)
        moved = 0
        for src in src_candidates:
            if not os.path.isfile(src):
                continue
            low = src.lower()
            if not low.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                continue
            dst = os.path.join(dst_dir, os.path.basename(src))
            if not os.path.exists(dst):
                os.replace(src, dst)
                moved += 1
        final_images = _count_images(dst_dir)
        extracted_any = extracted_any or final_images > 0
        report[d] = {'moved': moved, 'final_images': final_images}

    if extracted_any:
        return {'status': 'ok', 'datasets': report, 'attempts': attempts}
    return {
        'status': 'warning_unavailable_network',
        'warning': 'benchmark mirrors unreachable or blocked in current network',
        'attempts': attempts,
        'manual_hint': 'If your server is offline/blocked, manually copy images to data/benchmarks/{Set5,Set14,BSD100,Urban100}/HR',
    }


def prepare_dataset(root, auto_download=True):
    ensure_dataset_structure(root)
    ensure_benchmark_structure(root)
    download_report = try_download_public_sr_data(root) if auto_download else {}
    benchmark_report = try_download_benchmarks(root) if auto_download else {}
    summary = {split: _count_images(os.path.join(root, split)) for split in ['train_hr', 'val_hr', 'test_hr']}
    benchmark_summary = {d: _count_images(os.path.join(root, 'benchmarks', d, 'HR')) for d in BENCHMARK_DATASETS}
    return {
        'status': 'ready',
        'root': root,
        'images': summary,
        'benchmark_images': benchmark_summary,
        'download_report': download_report,
        'benchmark_report': benchmark_report,
        'message': 'Dataset prepared. Benchmark auto-download is best-effort; network-restricted environments may require manual benchmark file copy to benchmarks/<dataset>/HR.',
    }
