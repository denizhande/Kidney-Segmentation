# Kidney Segmentation

2D kidney segmentation on abdominal CT slices from the KiTS19 dataset. I take the NIfTI volumes, slice them along the z axis into 128x128 PNG image/mask pairs, and train a small encoder decoder convnet in TensorFlow/Keras to predict a kidney mask per slice. There is a third script that loads a saved model and reports mean Dice, accuracy, precision, recall and Jaccard over the test split, plus a side by side ground truth vs prediction plot.

## What is in the repo

```
kidney_03_loadData_and_preprocessing.py   # step 1: NIfTI volumes -> 128x128 PNG slices
kidney_02_trainingModel_segmentation.py   # step 2: load the PNGs, train, save the .h5
visualize.py                              # step 3: load a saved .h5, score it, plot one example
```

That is the whole repo. The dataset and the trained weights are not in it.

## Run order (the file numbering is misleading)

Ignore the numbers in the filenames. The file named `_03` is the preprocessing step and has to run **before** the file named `_02`, because `_02` reads the PNGs that `_03` writes. I never got around to renaming them. The order that actually works:

```bash
python kidney_03_loadData_and_preprocessing.py
python kidney_02_trainingModel_segmentation.py
python visualize.py
```

## Data

KiTS19 is not included here. You have to download it yourself from the official repo at https://github.com/neheller/kits19. The preprocessing script expects one folder per case, each containing `imaging.nii.gz` and `segmentation.nii.gz`, which is how the official download is laid out. Cases missing either file get skipped with a printed message.

Preprocessing walks the z axis of every volume and drops any slice whose mask is entirely zero, so only slices with some annotated kidney end up in the dataset. Images are resized to 128x128 with bilinear interpolation, masks with nearest neighbour, and both are written out with matplotlib's `imsave` using a gray colormap.

## The model, and what it actually is

The function is called `create_deeplab_model` and the checkpoint is called `deeplab_kidney_model_2d.h5`, but the architecture it builds is not DeepLabV3+. Here is everything it contains:

- Conv2D 32, 3x3, ReLU, same padding -> MaxPooling 2x2
- Conv2D 64, 3x3, ReLU, same padding -> MaxPooling 2x2
- Conv2DTranspose 64, 3x3, stride 2
- Conv2DTranspose 32, 3x3, stride 2
- Conv2D 1, 1x1, sigmoid

That is a small plain encoder decoder. Real DeepLabV3+ has atrous spatial pyramid pooling on top of a dilated backbone, and none of that is in this code. There are no skip connections between the encoder and decoder either, and no batch normalisation. The name is left over from an earlier plan I did not follow through on. I am pointing this out because someone reading the filenames could reasonably assume otherwise, and any comparison against published DeepLab numbers would be meaningless.

Training setup as written: Adam at lr 0.001, binary crossentropy loss, 20 epochs, batch size 32, 80/20 train/test split with `random_state=42`. Metrics are Keras accuracy plus a custom `DiceCoefficient` metric class that rounds both mask and prediction, then averages the per update Dice score.

`visualize.py` recomputes the same load and the same split (same seed), then loops over the test set one image at a time, thresholds predictions at 0.5, and averages Dice, accuracy, precision (with `zero_division=1`), recall and Jaccard from scikit-learn. The per sample loop is slow on a full test set. The plotting block at the bottom uses `sample_index = 4`, change that to look at a different slice.

## Requirements

Python 3 with TensorFlow (Keras 2 style `.h5` saving), plus:

```bash
pip install tensorflow opencv-python nibabel numpy matplotlib scikit-learn
```

## Known rough edges

These are real and you will hit them, so here they are up front.

**Hardcoded absolute paths.** All three scripts have Windows paths baked in that point at my old machine, for example `C:\Users\DELL XPS 8910\PycharmProjects\pythonProject\kits19\...`. Every one of these has to be changed before anything runs. In `_03` it is `data_path`, `image_output_dir` and `mask_output_dir`. In `_02` and in `visualize.py` it is `image_path` and `mask_path`. There is no config file and no argparse.

**The checkpoint name in `visualize.py` does not match what training produces.** Training saves `deeplab_kidney_model_2d.h5` into the working directory. `visualize.py` loads `deeplab_kidney_model_2d_v8_80e.h5` from an absolute path, which is a file from an older run that is not in this repo and is not produced by the training script. Point that load call at whatever `.h5` your own training run wrote.

**No weights are committed.** You have to preprocess and train yourself to get a model to evaluate.

**Mask intensities go through a colormap.** Because the masks are saved with `imsave` and a gray colormap, values are rescaled per slice before being written to PNG, then divided by 255 on the way back in. If a slice contains more than one non-zero label the intermediate label can land near 0.5 and get rounded down. If you care about separating kidney from tumour rather than treating everything annotated as foreground, write the mask PNGs directly instead of going through matplotlib.

**No windowing or HU clipping.** CT intensities are not clipped to a soft tissue window before saving, they just get whatever matplotlib's per slice normalisation does. That is probably the first thing I would change.

**Everything is single-file and top-level.** The scripts run their work at import time, there is no `main()` guard, and comments are in Turkish. `visualize.py` also has an unused import from `keras.src.saving`, which is a private Keras path and may break on newer versions.

## License

Not set yet. If you want to reuse this, open an issue and ask.
