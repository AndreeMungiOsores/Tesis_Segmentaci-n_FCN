import numpy as np, os, pickle, cv2, glob
from imageio import imread, imsave
from keras.callbacks import ModelCheckpoint
from sklearn import metrics
from pathlib import Path

from prepare_data import *
from data_process import *
from model import *

def calc_sensitivity(gt, pred):
    confusion = metrics.confusion_matrix(gt, pred, labels=[1, 0])
    return confusion[0, 0] / (confusion[0, 0] + confusion[0, 1])

def calc_specificity(gt, pred):
    confusion = metrics.confusion_matrix(gt, pred, labels=[1, 0])
    return confusion[1, 1] / (confusion[1, 0] + confusion[1, 1])

def calc_iou(gt, pred):
    iou = 0
    count = 0
    for i in range(gt.shape[0]):
        if np.sum(gt[i]) > 0:
            img1 = np.asarray(gt[i]).astype(bool)
            img2 = np.asarray(pred[i]).astype(bool)
            intersection = np.logical_and(img1, img2)
            union = np.logical_or(img1, img2)
            iou += np.sum(intersection) / np.sum(union)
            count += 1
    return iou / count if count > 0 else np.nan

def calc_dice(gt, pred):
    dice_score = 0
    count = 0
    for i in range(gt.shape[0]):
        if np.sum(gt[i]) > 0:
            dice_score += dice_coeff(gt[i], pred[i])
            count += 1
    return dice_score / count if count > 0 else np.nan

def dice_coeff(mask1, mask2):
    mask1 = np.asarray(mask1).astype(bool)
    mask2 = np.asarray(mask2).astype(bool)
    intersection = np.logical_and(mask1, mask2)
    return 2. * intersection.sum() / (mask1.sum() + mask2.sum())

def evaluate_model(model_path, test_dir, save_dir):
    unet_model = unet(pretrained_weights=model_path, input_size=(input_dim, input_dim, 1))
    test_gen = generator_test(test_dir, target_shape=(input_dim, input_dim, 1))
    predictions = unet_model.predict_generator(test_gen, steps=num_test_images, verbose=1)
    save_results(test_dir, save_dir, predictions)

if __name__ == '__main__':
    num_splits = 5
    epochs = 100
    eval_epochs = 1
    batch_sz = 32
    learning_rate = 1e-5
    decay_rate = learning_rate / epochs
    detection_area = 400
    prob_threshold = 0.5
    detection_thresh = prob_threshold * 256
    total_subjects = 75
    slices_ct = 49
    img_dim = 512
    crop_dim = 128
    stride = 64
    num_moves = int(img_dim / stride) - 1
    window_settings = [40, 120]
    close_kernel = np.ones((10, 10), np.uint8)
    open_kernel = np.ones((5, 5), np.uint8)

    counter = 1
    result_dir = Path(f'results_{counter}')
    while os.path.isdir(str(result_dir)):
        counter += 1
        result_dir = Path(f'results_{counter}')
    os.mkdir(str(result_dir))

    os.mkdir(str(Path(result_dir, 'crops')))
    os.mkdir(str(Path(result_dir, 'full_ct_raw')))
    os.mkdir(str(Path(result_dir, f'full_ct_processed_{prob_threshold}')))

    dataset_zip = 'ct-images.zip'
    cross_dir = 'ProcessedData'
    prepare_data(dataset_zip, cross_dir, total_subjects, img_dim, crop_dim, stride, num_moves, window_settings)

    with open(str(Path(cross_dir, 'Data.pkl')), 'rb') as data_file:
        [labels, ct_data, test_masks, shuffled_subjects] = pickle.load(data_file)
    del ct_data
    test_masks = np.uint8(test_masks)
    test_avg = np.where(np.sum(np.sum(test_masks, axis=1), axis=1) > detection_area, 1, 0)
    predictions = np.zeros((test_masks.shape[0], img_dim, img_dim), dtype=np.uint8)

    for split in range(num_splits):
        print(f"Split {split}: Training")
        split_dir = Path(result_dir, 'crops', f'Split{split}')
        os.makedirs(str(split_dir), exist_ok=True)

        train_dir = Path(cross_dir, f'Split{split}', 'train')
        val_dir = Path(cross_dir, f'Split{split}', 'validate')
        test_dir = Path(cross_dir, f'Split{split}', 'test', 'crops')

        train_gen = generator_train(batch_sz, train_dir, 'image', 'label', {}, target_shape=(crop_dim, crop_dim))
        val_gen = generator_validate(batch_sz, val_dir, 'image', 'label', target_shape=(crop_dim, crop_dim))

        model = unet(learningRate=learning_rate, decayRate=decay_rate, input_size=(crop_dim, crop_dim, 1))
        checkpoint = ModelCheckpoint(str(Path(result_dir, f'model_split{split}.hdf5')), save_best_only=True)
        model.fit_generator(train_gen, epochs=epochs, validation_data=val_gen, validation_steps=5, callbacks=[checkpoint])

        print(f"Split {split}: Testing")
        evaluate_model(str(Path(result_dir, f'model_split{split}.hdf5')), test_dir, split_dir)

        for i in range(predictions.shape[0]):
            predictions[i] = calc_dice(test_masks[i], predictions[i])

    print(f"Average dice score: {np.mean(predictions):.3f}")
