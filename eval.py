"""Evaluate the two fine-tuned ConvNeXt-Tiny models (region classifier, lat/long regressor) on the validation set.

Fixes the original notebook evaluation: validation uses a plain eval transform (no random flip/rotation),
the validation set is NOT outlier-filtered, and the saved *best* checkpoints are loaded. Baselines are included.

Run: python eval.py [--data .] [--models models]
  needs: <data>/labels_train.csv, labels_val.csv, images_train/, images_val/,
         <models>/best_region_model.pth, best_latlong_model.pth
Coordinates are the course dataset's scaled units (not degrees), so errors are in those units.
"""
import argparse
import json
import os

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import RobustScaler
from torchvision import transforms
from torchvision.models import convnext_tiny

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EVAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def region_model():
    """Head exactly as trained in RegionP2.ipynb: Flatten->LayerNorm->Dropout->Linear *inside* classifier[2]
    (so the net ends up with two LayerNorms). Needed to load the saved weights."""
    m = convnext_tiny()
    nf = m.classifier[2].in_features
    m.classifier[2] = nn.Sequential(nn.Flatten(), nn.LayerNorm(nf, eps=1e-6), nn.Dropout(0.5), nn.Linear(nf, 15))
    return m


def latlong_model():
    """Head exactly as trained in Latlongp2.ipynb: whole classifier replaced."""
    m = convnext_tiny()
    nf = m.classifier[2].in_features
    m.classifier = nn.Sequential(nn.Flatten(), nn.LayerNorm(nf, eps=1e-6), nn.Dropout(0.5), nn.Linear(nf, 2))
    return m


@torch.no_grad()
def predict(model, ckpt, files, root, batch=32):
    sd = torch.load(ckpt, map_location=device)
    if all(k.startswith("model.") for k in sd):  # the shared lat/long checkpoint was saved from a wrapper module
        sd = {k[len("model."):]: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    out = []
    for i in range(0, len(files), batch):
        x = torch.stack([EVAL_TF(Image.open(os.path.join(root, f)).convert("RGB")) for f in files[i:i + batch]])
        out.append(model(x.to(device)).cpu())
    return torch.cat(out).numpy()


def iqr_filter(df, col):
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    return df[(df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)]


def coord_errors(pred, true):
    d = np.linalg.norm(pred - true, axis=1)
    return {"mean_err": float(d.mean()), "median_err": float(np.median(d)), "rmse": float(np.sqrt((d ** 2).mean())),
            "mse_lat": float(((pred[:, 0] - true[:, 0]) ** 2).mean()),
            "mse_lon": float(((pred[:, 1] - true[:, 1]) ** 2).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".")
    ap.add_argument("--models", default="models")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0), np.random.seed(0)

    train_csv = os.path.join(a.data, "labels_train.csv")
    train = pd.read_csv(train_csv) if os.path.exists(train_csv) else None  # lat/long scaler + baselines need it
    val = pd.read_csv(os.path.join(a.data, "labels_val.csv"))
    val_files, val_root = val["filename"].tolist(), os.path.join(a.data, "images_val")
    res = {"n_val": len(val)}

    # --- region -------------------------------------------------------------------------
    y = val["Region_ID"].to_numpy() - 1  # training used 0-14
    logits = predict(region_model(), os.path.join(a.models, "best_region_model.pth"), val_files, val_root)
    pred_region = logits.argmax(1)
    # without train labels, the best constant guess on val itself is an upper bound on the majority baseline
    majority = int((train if train is not None else val)["Region_ID"].mode()[0]) - 1
    res["majority_baseline_source"] = "train" if train is not None else "val (upper bound; labels_train.csv not found)"
    res["region"] = {"top1": float((pred_region == y).mean()), "macro_f1": float(f1_score(y, pred_region, average="macro")),
                     "majority_baseline_top1": float((y == majority).mean())}
    cm = confusion_matrix(y, pred_region, labels=range(15))
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm / np.maximum(cm.sum(1, keepdims=True), 1), cmap="Blues")
    ax.set_xlabel("predicted region"), ax.set_ylabel("true region")
    ax.set_xticks(range(15), range(1, 16)), ax.set_yticks(range(15), range(1, 16))
    fig.savefig(os.path.join(a.out, "confusion_matrix.png"), dpi=120, bbox_inches="tight"), plt.close(fig)

    # --- lat/long (scaler and baselines come from the IQR-filtered train set, as in training) ----
    # If labels_train.csv is unavailable, fall back to the validation labels. The numbers are then APPROXIMATE:
    # the scaler differs slightly from the training one and the baselines are fit on the data they are scored on
    # (optimistic for the baselines, i.e. conservative for claims that the model beats them).
    ref = train if train is not None else val
    res["latlong_reference"] = "train" if train is not None else "val (approximate: labels_train.csv not found)"
    tr = iqr_filter(iqr_filter(ref, "latitude"), "longitude")
    scaler = RobustScaler().fit(tr[["latitude", "longitude"]])
    true = val[["latitude", "longitude"]].to_numpy()
    out = predict(latlong_model(), os.path.join(a.models, "best_latlong_model.pth"), val_files, val_root)
    pred = scaler.inverse_transform(out)
    keep = iqr_filter(iqr_filter(val, "latitude"), "longitude").index.to_numpy()  # what the old notebook evaluated on
    centroids = tr.groupby("Region_ID")[["latitude", "longitude"]].mean().reindex(range(1, 16)).to_numpy()
    mean_pred = np.tile(tr[["latitude", "longitude"]].mean().to_numpy(), (len(val), 1))
    res["latlong"] = {
        "model": coord_errors(pred, true),
        "model_on_iqr_filtered_val": coord_errors(pred[keep], true[keep]),
        "baseline_train_mean": coord_errors(mean_pred, true),
        "baseline_region_centroid_from_predicted_region": coord_errors(centroids[pred_region], true),
    }
    b, m = res["latlong"]["baseline_train_mean"]["median_err"], res["latlong"]["model"]["median_err"]
    res["latlong"]["median_err_reduction_vs_mean_baseline_pct"] = float(100 * (b - m) / b)
    res["latlong"]["n_val_iqr_filtered_out"] = int(len(val) - len(keep))
    if train is None:  # how much does the approximate scaler matter? refit it on two random halves of val
        idx = np.random.default_rng(0).permutation(len(val))
        halves = [val.iloc[idx[:len(idx) // 2]], val.iloc[idx[len(idx) // 2:]]]
        res["latlong"]["model_median_err_with_scaler_fit_on_val_halves"] = [
            coord_errors(RobustScaler().fit(iqr_filter(iqr_filter(h, "latitude"), "longitude")[["latitude", "longitude"]])
                         .inverse_transform(out), true)["median_err"] for h in halves]

    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
