# Campus Image Location Prediction

Two fine-tuned `ConvNeXt-Tiny` models that predict, from a single campus photo, (1) which of 15 campus regions it was taken in and (2) its latitude/longitude. They are two separate models (one per task), not a single multi-task network.

This started as the Phase-2 project of the SMAI course at IIIT Hyderabad (see `Project_Phase_2.pdf`) and was later cleaned up. The coordinates in the dataset are the course's *scaled* units, not degrees, so location errors are reported in those units and relative to baselines.

## Features

*   **Latitude & Longitude regression**: MSE loss on robust-scaled coordinates.
*   **Region ID classification**: 15-way, cross-entropy loss.

## Getting Started

### Prerequisites

*   Python 3.8+
*   Conda or a virtual environment tool

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ashurudra09/campus-image-location-prediction.git
    cd campus-image-location-prediction
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the data and weights:**
    *   Download the [dataset](https://drive.google.com/drive/folders/1vtLNqk0N2GriYKcv6FUwiPQHSCBN69E-?usp=sharing) and place `labels_train.csv` and `images_train/` in the repo root (`labels_val.csv` and `images_val/` are already here), the same layout the notebooks read.
    *   Download the [pre-trained weights](https://drive.google.com/drive/folders/1j_WcoJo53CN_7BNkfpWw2KUbiK9SO1pm?usp=sharing) and place `best_region_model.pth` and `best_latlong_model.pth` in `models/`.

## Evaluation

```bash
python eval.py
```

`eval.py` loads the best checkpoints (if `labels_train.csv` is missing it falls back to the validation labels for the lat/long scaler and baselines, and says so in `metrics.json`) and writes `results/metrics.json` and `results/confusion_matrix.png`. It reports:

*   Region: top-1 accuracy, macro-F1, and the majority-class baseline accuracy.
*   Lat/long: mean/median/RMSE Euclidean error and per-axis MSE (the course metric), against two baselines (train-mean coordinate; centroid of the predicted region), plus the median-error reduction versus the mean baseline.

Evaluation uses a plain resize + normalize transform and the **full** validation set. The notebooks' original validation numbers should not be quoted: they were computed with random flip/rotation applied to validation images and, for lat/long, after dropping validation outliers. `eval.py` also reports the lat/long error on that filtered subset so the difference is visible.

About 2% of the validation coordinates (7 of 369) are IQR outliers. They dominate the mean error, RMSE and MSE, so **median error** is the more meaningful headline for lat/long; the other numbers are reported for completeness.

*Note on the shared lat/long checkpoint:* its keys are prefixed with `model.` (it was saved from a wrapper module, unlike the notebook, which saves the plain network), so it was probably produced by slightly different code than `latlong/Latlongp2.ipynb`. `eval.py` strips the prefix and loads it with strict key matching.

### Results (validation set, 369 images)

| Task | Metric | Result | Baseline |
| ---- | ------ | ------ | -------- |
| Region (15 classes) | Top-1 accuracy | **95.1%** | at most 9.8% (always guessing the most common region; upper bound computed on the validation set) |
| Region (15 classes) | Macro-F1 | **0.947** | n/a |
| Lat/long | Median error (dataset units) | **about 690** (590–850 depending on the scaler estimate, see below) | mean coordinate: 1358; predicted-region centroid: **303** (317 leave-one-out) |

Confusion matrix: `results/confusion_matrix.png` (the weakest regions are 9 and 11, mostly confused with neighbouring regions). Raw numbers: `results/metrics.json`.

**Lat/long is approximate.** `labels_train.csv` is not available (the dataset Drive link is dead), so `eval.py` fits the `RobustScaler` and the baselines on the validation labels instead (`latlong_reference` in `metrics.json` records this). The scaler is a little different from the training one, so the model's median error moves between about 590 and 850 when it is refit on two random halves of the validation labels; the baselines are fit on the data they are scored on, which flatters them slightly (the leave-one-out centroid baseline is 317 vs 303 in-sample). Treat these as indicative, not final.

What the numbers say, even with that uncertainty: the regressor cuts the median error of the mean-coordinate baseline by roughly 35-55% (37-57% across the scaler estimates), but **"classify the region, then return that region's centroid" has about half the error of the regressor**. Directly regressing scaled coordinates with MSE is therefore a weak objective here (it is also pulled around by the ~2% outlier coordinates). A natural next step is to predict a per-region offset from the centroid, or to train a robust-loss / multi-task model.

Caveat: the checkpoints were selected by validation loss on this same validation set, so these numbers are slightly optimistic; a separate held-out test split was not used.

## Methodology

Both models are ImageNet-pretrained `ConvNeXt-Tiny` networks, fine-tuned end to end (Adam, batch size 32, `ReduceLROnPlateau` with patience 1 and factor 0.5, dropout 0.5, stochastic depth 0.5, 40 epochs maximum). The checkpoint with the lowest validation loss is saved. The code lives in `latlong/Latlongp2.ipynb` and `regionID/RegionP2.ipynb`.

### Preprocessing
*   **Lat/long:** IQR outlier filtering on the training and validation labels in the notebook (keep values within `[Q1 - 1.5·IQR, Q3 + 1.5·IQR]`, applied to latitude then longitude). Coordinates are then scaled with `RobustScaler` fitted on the filtered training set.
*   **Region ID:** labels are shifted from 1-15 to 0-14 for training.

### Transforms (training)
Resize to 224x224, random horizontal flip, random rotation (±10°), ImageNet mean/std normalization. (The same transform is currently also used on validation inside the notebooks, which `eval.py` corrects.)

### Latlong model (`latlong/Latlongp2.ipynb`)
- The classifier is replaced by `Flatten → LayerNorm → Dropout(0.5) → Linear(768, 2)`.
- MSE loss, learning rate 1e-4.
- Training stops early when the learning rate falls below 5e-6.

### RegionID model (`regionID/RegionP2.ipynb`)
- `classifier[2]` is replaced by `Flatten → LayerNorm → Dropout(0.5) → Linear(768, 15)`. ConvNeXt's own classifier already starts with a LayerNorm, so this model has two LayerNorms in a row (kept as trained, so the saved weights load; `eval.py` rebuilds this head exactly).
- Cross-entropy loss, learning rate 2e-4.
- Training stops early when the learning rate falls below 1e-6.

---

### Link to Model Weights
[Google Drive](https://drive.google.com/drive/folders/1j_WcoJo53CN_7BNkfpWw2KUbiK9SO1pm?usp=sharing) (the folder also contains `best_angle_model.pth` from the unused direction task)

### Link to Dataset
[Google Drive](https://drive.google.com/drive/folders/1vtLNqk0N2GriYKcv6FUwiPQHSCBN69E-?usp=sharing)

---
