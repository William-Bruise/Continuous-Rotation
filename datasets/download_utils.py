import os

def ensure_dataset_structure(root):
    for split in ['train_hr', 'val_hr', 'test_hr']:
        os.makedirs(os.path.join(root, split), exist_ok=True)
    return {
        'status': 'ready',
        'message': 'Dataset folders ensured. Please place PNG/JPG files in train_hr/val_hr/test_hr if auto-download is unavailable.'
    }
