# Kidney Segmentation

2D kidney segmentation on abdominal CT from the KiTS19 dataset. I take the NIfTI volumes, cut them into 128x128 PNG image/mask pairs, and train a small encoder-decoder convnet in TensorFlow/Keras to predict a kidney mask per slice. There is a third script that loads a saved model and reports mean Dice, accuracy, precision, recall and Jaccard over the test split, plus a side-by-side ground truth vs prediction plot.

## What is in the repo

```
kidney_03_loadData_and_preprocessing.py   # step 1: NIfTI volumes -> 128x128 PNG slices
kidney_02_trainingModel_segmentation.py   # step 2: load the PNGs, train, save the .h5
visualize.py                              # step 3: load a saved .h5, score it, plot one example
.gitignore                                # excludes the .h5 checkpoints, the NIfTI volumes and the generated 2d_data folder
```

That is the whole repo, plus this README. The dataset and the trained weights are not in it.

## Run order (the file numbering is misleading)

Ignore the numbers in the filenames. The file named `_03` is the preprocessing step and has to run **before** the file named `_02`, because `_02` reads the PNGs that `_03` writes. I never got around to renaming them. The order that actually works:

```bash
python kidney_03_loadData_and_preprocessing.py
python kidney_02_trainingModel_segmentation.py
python visualize.py
```

## Data

KiTS19 is not included here. Clone https://github.com/neheller/kits19 and then run their downloader (`python -m starter_code.get_imaging`) from inside that clone. This matters: the git clone ships only `segmentation.nii.gz`, and `imaging.nii.gz` is fetched separately because it is too large for git. If you skip the downloader, every case prints `Skipping case ...: Required files not found.` and you get no PNGs. Point `data_path` at the clone's `data/` folder, which holds one folder per case containing both files. Budget on the order of 30 GB for the imaging volumes.

Preprocessing iterates over axis 2 of every volume and drops any slice whose mask is entirely zero, so only slices with some annotated kidney end up in the dataset. Read the note on the slicing axis below before you trust that. Images are resized to 128x128 with bilinear interpolation, masks with nearest neighbor, and both are written out with matplotlib's `imsave` using a gray colormap. The script prints a line for every slice it writes and every slice it skips, so you can tell it is making progress rather than hanging.

## The model, and what it actually is

The function is called `create_deeplab_model` and the checkpoint is called `deeplab_kidney_model_2d.h5`, but the architecture it builds is not DeepLabV3+. Here is everything it contains:

- Conv2D 32, 3x3, ReLU, same padding -> MaxPooling 2x2
- Conv2D 64, 3x3, ReLU, same padding -> MaxPooling 2x2
- Conv2DTranspose 64, 3x3, stride 2
- Conv2DTranspose 32, 3x3, stride 2
- Conv2D 1, 1x1, sigmoid

That is a small plain encoder-decoder. Real DeepLabV3+ has atrous spatial pyramid pooling on top of a dilated backbone, and none of that is in this code. There are no skip connections between the encoder and decoder either, and no batch normalization. The name is left over from an earlier plan I did not follow through on. I am pointing this out because someone reading the filenames could reasonably assume otherwise, and any comparison against published DeepLab numbers would be meaningless.

Training setup as written: Adam at lr 0.001, binary crossentropy loss, 20 epochs, batch size 32, 80/20 train/test split with `random_state=42`. Metrics are Keras accuracy plus a custom `DiceCoefficient` metric class that rounds both mask and prediction, then averages the per-update Dice score. Note that the same 20 percent is passed as `validation_data` during `fit`, evaluated at the end of training, and scored again by `visualize.py`. There is no separate held-out set, so treat every reported number as optimistic.

`visualize.py` recomputes the same load and the same split (same seed), then loops over the test set one image at a time, thresholds predictions at 0.5, and averages accuracy, precision (with `zero_division=1`), recall and Jaccard from scikit-learn, plus Dice from a local `dice_coef` function that uses `smooth=1`. That local Dice is not the same formula as the `DiceCoefficient` metric used in training, so do not compare the two numbers. Note also that `dice_coef` is passed the raw float ground truth while the scikit-learn metrics get a boolean cast of it. The per-sample loop is slow on a full test set. The plotting block at the bottom uses `sample_index = 4`, change that to look at a different slice.

## Requirements

Python 3 with TensorFlow 2.x on the Keras 2 side. The scripts use `tensorflow.keras` import paths and `.h5` checkpoints, so TensorFlow 2.16 and later, which ships Keras 3 by default, will not run this as written.

```bash
pip install "tensorflow<2.16" opencv-python nibabel numpy matplotlib scikit-learn
```

## Known rough edges

**Hardcoded absolute paths.** All three scripts have Windows paths baked in that point at my old machine, for example `C:\Users\DELL XPS 8910\PycharmProjects\pythonProject\kits19\...`. Every one of these has to be changed before anything runs. In `_03` it is `data_path`, `image_output_dir` and `mask_output_dir`. In `_02` it is `image_path` and `mask_path`. In `visualize.py` it is `image_path`, `mask_path`, and a third path hardcoded inside the `load_model` call on line 69, which sits in a different parent directory from the other two, so a find and replace on the data root will not catch it. There is no config file and no argparse.

**The slicing axis is probably wrong.** KiTS19 volumes load as (slices, height, width), so axis 0 is the axial axis. The preprocessing loop iterates axis 2, which cuts sagittal planes of shape (slices, 512) and then squeezes that rectangle into a 128x128 square. If you want axial slices, change the loop to `range(image.shape[0])` and index `image[i, :, :]`. Verify this against your own download before training.

**Mask intensities go through a colormap.** Masks are saved with `imsave` and a gray colormap, so values are rescaled per slice before being written to PNG, then divided by 255 on the way back in. Normalization is per slice, so the kidney label lands at 255 in a slice that has no tumor but at 128 in a slice that has one. The same anatomy therefore reaches the network as a target of 1.0 in some samples and 0.502 in others, and binary crossentropy trains against that 0.502 directly because the loss never thresholds. Write the mask PNGs directly instead of going through matplotlib.

**Images and masks are paired by position, not by name.** `load_data` zips two independently sorted directory listings. One stray file in either folder shifts every following pair and trains images against the wrong masks, silently. If you interrupt preprocessing partway, delete both output folders and start over rather than resuming.

**The checkpoint name in `visualize.py` does not match what training produces.** Training saves `deeplab_kidney_model_2d.h5` into the working directory. `visualize.py` loads `deeplab_kidney_model_2d_v8_80e.h5` from an absolute path, which is a file from an older run that is not in this repo and is not produced by the training script. Point that load call at whatever `.h5` your own training run wrote.

**No weights are committed.** You have to preprocess and train yourself to get a model to evaluate.

**No windowing or HU clipping.** CT intensities are not clipped to a soft-tissue window before saving, they just get whatever matplotlib's per-slice normalization does. That is probably the first thing I would change.

**Everything is single-file and top-level.** The scripts run their work at import time, there is no `main()` guard, and comments are in Turkish. `visualize.py` also carries six unused imports copied over from the training script (`layers`, `models`, `Adam`, `Metric`, `Model`, and `load_model`). The `load_model` one is worth deleting first: it comes from `keras.src.saving`, a private Keras path that may break on newer versions, and the file never calls it because the actual load goes through `tf.keras.models.load_model`.

## License

Not set yet. If you want to reuse this, open an issue and ask.
