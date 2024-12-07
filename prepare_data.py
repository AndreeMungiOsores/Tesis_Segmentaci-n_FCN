import numpy as np, os, random
from pathlib import Path
import pandas as pd
from imageio import imsave
import pickle
import zipfile
import nibabel as nib

def adjust_ct(ct_scan, level=40, width=120):
    min_val = level - width / 2
    max_val = level + width / 2
    for i in range(ct_scan.shape[2]):
        slice_img = ct_scan[:, :, i]
        slice_img = (slice_img - min_val) * (255 / (max_val - min_val))
        slice_img[slice_img < 0] = 0
        slice_img[slice_img > 255] = 255
        ct_scan[:, :, i] = slice_img
    return ct_scan

def load_data(dataset_path, subject_id, window_params):
    ct_path = Path(dataset_path, 'ct_scans', f"{subject_id:03d}.nii")
    mask_path = Path(dataset_path, 'masks', f"{subject_id:03d}.nii")
    ct_scan = nib.load(str(ct_path)).get_fdata()
    masks = nib.load(str(mask_path)).get_fdata()
    ct_scan = adjust_ct(ct_scan, window_params[0], window_params[1])
    return ct_scan, masks

def split_slices(ct_scan, img_dim, crop_dim, steps):
    crops = np.zeros([crop_dim, crop_dim, steps * steps], dtype=np.uint8)
    idx = 0
    for x in range(steps):
        for y in range(steps):
            crops[:, :, idx] = ct_scan[
                int(x * img_dim / (steps + 1)):int(x * img_dim / (steps + 1) + crop_dim),
                int(y * img_dim / (steps + 1)):int(y * img_dim / (steps + 1) + crop_dim)
            ]
            idx += 1
    return crops

def process_data(zip_path, out_dir, num_subjects, img_dim, crop_dim, stride, steps, window_params):
    current_path = Path(os.getcwd())
    extract_dir = Path(current_path, 'extracted_data')
    if not os.path.isfile(Path(out_dir, 'Processed.pkl')):
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        
        if os.path.exists(zip_path):
            if not os.path.exists(str(extract_dir)):
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall('extracted_data')
            
            label_data = pd.read_csv(Path(extract_dir, 'labels.csv')).values
            label_data[:, 0] -= 49

            shuffled_subjects = np.random.permutation(np.unique(label_data[:, 0]))

            for idx, subject in enumerate(shuffled_subjects[:num_subjects]):
                print(f"Processing subject {subject}")
                ct_scan, masks = load_data(extract_dir, subject, window_params)
                for slice_idx in range(ct_scan.shape[2]):
                    crops_ct = split_slices(ct_scan[:, :, slice_idx], img_dim, crop_dim, steps)
                    crops_masks = split_slices(masks[:, :, slice_idx], img_dim, crop_dim, steps)
                    for crop_idx in range(steps * steps):
                        if np.sum(crops_masks[:, :, crop_idx]) > 0:
                            imsave(Path(out_dir, f"crop_{idx}_{slice_idx}_{crop_idx}.png"), crops_ct[:, :, crop_idx])
