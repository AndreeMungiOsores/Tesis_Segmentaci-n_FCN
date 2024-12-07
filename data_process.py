from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np, os, glob
import skimage.transform as trans
from imageio import imsave, imread
from pathlib import Path

def normalize_data(input_img, input_mask, multi_class, num_classes):
    if multi_class:
        input_img = input_img / 255
        input_mask = input_mask[:, :, :, 0] if len(input_mask.shape) == 4 else input_mask[:, :, 0]
        transformed_mask = np.zeros(input_mask.shape + (num_classes,))
        for class_idx in range(num_classes):
            transformed_mask[input_mask == class_idx, class_idx] = 1
        transformed_mask = np.reshape(transformed_mask, (
            transformed_mask.shape[0], transformed_mask.shape[1] * transformed_mask.shape[2], transformed_mask.shape[3]
        )) if multi_class else np.reshape(transformed_mask, (
            transformed_mask.shape[0] * transformed_mask.shape[1], transformed_mask.shape[2]))
        input_mask = transformed_mask
    elif np.max(input_img) > 1:
        input_img = input_img / 255
        input_mask = input_mask / 255
        input_mask[input_mask > 0.5] = 1
        input_mask[input_mask <= 0.5] = 0
    return input_img, input_mask

def generator_train(batch_size, train_path, img_dir, mask_dir, augment_args,
                    img_mode="grayscale", mask_mode="grayscale", img_prefix="img", mask_prefix="mask",
                    multi_class=False, num_classes=2, save_dir=None, target_shape=(128, 128), seed=1):
    img_gen = ImageDataGenerator(**augment_args)
    mask_gen = ImageDataGenerator(**augment_args)
    img_stream = img_gen.flow_from_directory(
        train_path, classes=[img_dir], class_mode=None, color_mode=img_mode,
        target_size=target_shape, batch_size=batch_size, save_to_dir=save_dir,
        save_prefix=img_prefix, seed=seed)
    mask_stream = mask_gen.flow_from_directory(
        train_path, classes=[mask_dir], class_mode=None, color_mode=mask_mode,
        target_size=target_shape, batch_size=batch_size, save_to_dir=save_dir,
        save_prefix=mask_prefix, seed=seed)
    stream = zip(img_stream, mask_stream)
    for img, mask in stream:
        img, mask = normalize_data(img, mask, multi_class, num_classes)
        yield img, mask

def generator_validate(batch_size, val_path, img_dir, mask_dir, img_mode="grayscale",
                       mask_mode="grayscale", img_prefix="img", mask_prefix="mask",
                       multi_class=False, num_classes=2, save_dir=None, target_shape=(128, 128), seed=1):
    img_gen = ImageDataGenerator()
    mask_gen = ImageDataGenerator()
    img_stream = img_gen.flow_from_directory(
        val_path, classes=[img_dir], class_mode=None, color_mode=img_mode,
        target_size=target_shape, batch_size=batch_size, save_to_dir=save_dir,
        save_prefix=img_prefix, seed=seed)
    mask_stream = mask_gen.flow_from_directory(
        val_path, classes=[mask_dir], class_mode=None, color_mode=mask_mode,
        target_size=target_shape, batch_size=batch_size, save_to_dir=save_dir,
        save_prefix=mask_prefix, seed=seed)
    stream = zip(img_stream, mask_stream)
    for img, mask in stream:
        img, mask = normalize_data(img, mask, multi_class, num_classes)
        yield img, mask

def generator_test(test_dir, target_shape=(128, 128), multi_class=False, grayscale=True):
    img_files = glob.glob(os.path.join(test_dir, "*.png"))
    for item in img_files:
        img = imread(item)
        img = img / 255
        if img.shape[0] != target_shape[0] or img.shape[1] != target_shape[1]:
            img = trans.resize(img, target_shape)
        img = np.reshape(img, img.shape + (1,)) if not multi_class else img
        img = np.reshape(img, (1,) + img.shape)
        yield img

def save_results(test_dir, result_dir, predictions, multi_class=False, num_classes=2):
    img_files = glob.glob(os.path.join(test_dir, "*.png"))
    for idx, img_path in enumerate(img_files):
        pred_img = predictions[idx, :, :, 0]
        imsave(Path(result_dir, os.path.basename(img_path)), np.uint8(pred_img * 255))

def prepare_test_mask(test_dir, total_imgs, target_shape=(128, 128), multi_class=False, grayscale=True):
    masks = np.zeros([total_imgs, target_shape[0], target_shape[1], 1], dtype=np.float32)
    img_files = glob.glob(os.path.join(test_dir, "*.png"))
    for idx, item in enumerate(img_files):
        mask = imread(item)
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        mask = trans.resize(mask, target_shape)
        mask = np.expand_dims(mask, axis=-1)
        mask = np.expand_dims(mask, axis=0)
        masks[idx] = mask
    return masks
