# GPR-Anomaly-Detection

Ground Penetrating Radar (GPR) image anomaly detection using ResNet18 features, PCA dimensionality reduction, local dynamic modeling, and OneClassSVM.

## Overview

This project compares multiple approaches for unsupervised defect detection in GPR scan images:

- **Main pipeline**: ResNet18 feature extraction → PCA → local dynamic model (Ridge Regression) → OneClassSVM anomaly detection → segment merging
- **2D-ESN**: Echo State Network as an alternative dynamic modeling method
- **HOG+SVM**: Traditional HOG features + SVM baseline
- **YOLO**: End-to-end object detection baseline (YOLOv8n)

## Pipeline

```
GPR Images → Sliding Windows → ResNet18 Features → PCA → Dynamic Model (θ vectors) → OCSVM → Segments → Evaluation
```

1. **Window extraction**: Sliding windows (128×128, stride 32) from GPR scans
2. **Feature extraction**: ResNet18 features from each window
3. **Dimensionality reduction**: PCA to 32 dimensions
4. **Dynamic modeling**: Ridge regression fits $u_{t+1} = M \cdot u_t + c$ per window, producing θ parameter vectors
5. **Anomaly detection**: OneClassSVM on the θ vector space
6. **Post-processing**: Merge anomalous windows into continuous segments, evaluate against ground truth

## Project Structure

```
├── gpr_yolo_dataset/     # Dataset, window export, feature extraction scripts
│   ├── images/           # GPR scan images (YOLO format)
│   ├── labels/           # YOLO annotation labels
│   ├── export_window_dataset.py
│   └── export_resnet_features_all.py
├── analysis/             # Detection core (OCSVM training, dynamic model fitting)
│   └── detection.py
├── 2D-ESN/               # 2D Echo State Network baseline
├── HOG+SVM/              # HOG + SVM baseline
├── yolo/                 # YOLO detection + t-SNE visualization
├── yolo_baseline/        # YOLO training outputs
├── models/               # ResNet model definition
├── fit_all_theta_vectors.py      # Fit dynamic models for all windows
├── fit_pca_and_transform_features.py  # PCA fitting and feature transformation
├── eval_segment_from_ocsvm_preds.py   # Evaluate detection segments
├── run_gpr_pipeline_threshold_click.py # One-click pipeline runner
└── plot_*.py             # Visualization scripts (t-SNE, loss curves, heatmaps)
```

## Key Scripts

| Script | Purpose |
|--------|---------|
| `gpr_yolo_dataset/export_window_dataset.py` | Extract sliding windows from GPR images |
| `gpr_yolo_dataset/export_resnet_features_all.py` | Extract ResNet18 features for all windows |
| `fit_pca_and_transform_features.py` | Fit PCA and project features |
| `fit_all_theta_vectors.py` | Fit local dynamic models (Ridge regression) |
| `analysis/detection.py` | Train OCSVM, predict anomalies |
| `eval_segment_from_ocsvm_preds.py` | Merge predictions into segments and evaluate |
| `run_gpr_pipeline_threshold_click.py` | Run the full pipeline end-to-end |

## Requirements

- Python 3.10+
- PyTorch, torchvision
- scikit-learn
- NumPy, OpenCV
- Ultralytics YOLO (for YOLO baselines)

## Contributors

| Contributor | Role |
|-------------|------|
| [@chang-siqian](https://github.com/chang-siqian) | All the actual code, experiments, and suffering |
| [Claude Opus 4.7](https://claude.ai) | README, git setup, moral support |
