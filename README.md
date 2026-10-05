# Ship Detection in SAR Imagery

A YOLOv8s-P2 ship detector trained on LS-SSDD-v1.0 (Sentinel-1) and run on a real Sentinel-1 scene.

| | Precision | Recall | mAP50 |
|---|---|---|---|
| LS-SSDD test (scenes 11–15) | 0.80 | 0.70 | 0.75 |
| Real Sentinel-1 scene (xView3, 71 ships) | 0.85 | 0.89 | n/a (scored by point matching instead) |

## File structure

```
code/
  lsssdd_preprocess.ipynb       # 1. parse labels, normalise, split by scene, build YOLO dataset
  lsssdd_train_yolov8s.ipynb    # 2. train YOLOv8s / YOLOv8s-P2 (resumable), evaluate on test
  lsssdd_diagnose_recall.ipynb  # 3. recall by ship size, missed-ship gallery
  s1_gee_export.js              # optional: export a Sentinel-1 VV dB GeoTIFF from Google Earth Engine
  s1_xview3_inference.py        # 4. inference on a Sentinel-1 scene (CPU), scoring and overlays
  s1_xview3_inference.ipynb     #    same as the .py, as a notebook
report/
  figures/                      # figures used in the report
README.md
```

## How to run

### 1. Preprocess (Kaggle)
1. Create a Kaggle notebook from `lsssdd_preprocess.ipynb`.
2. Add the dataset `petrarodriguez/ls-ssdd-v1-0` as an input.
3. Click **Save & Run All**. The output folder is `lsssdd_yolo/`, containing images, labels and `data.yaml`. Stage figures go in `figures/`.

### 2. Train (Kaggle, GPU)
1. Create a notebook from `lsssdd_train_yolov8s.ipynb`, turn on a GPU, and add step 1's output as an input.
2. Set `MODEL = "yolov8s-p2.yaml"` and `BATCH = 8`. These are the settings for the final model.
3. Click **Save & Run All**. It takes about 1.5 hours.
4. The outputs are:
   - `runs/<run>/weights/best.pt`
   - `results.png`
   - `training.gif`
   - `test_metrics.txt`

To resume a stopped run, attach that run's own output as an input and run it again.

### 3. Inference on Sentinel-1 (laptop, CPU)
1. Download one xView3-SAR validation scene and `validation.csv` from https://iuu.xview.us (free account), then unpack the scene.
2. Install the requirements using setup.bat.

3. Put `best.pt`, `validation.csv` and the scene folder next to the script, then run:
   ```
   python s1_xview3_inference.py --scene 0157baf3866b2cf9v --labels validation.csv --weights best.pt
   ```
   Useful options:
   - `--region full` runs the whole scene (slow).
   - `--labels none` gives detections only, with no scores.
   - `--conf 0.3` changes the score threshold.
4. Results are written to `s1_results/`:
   - `metrics.txt`
   - `detections.csv`, with latitude and longitude for each detection
   - `overlay_region.png` and `overlay_zoom.png`
   - `intensity_settings.png`

## Notes
- The scene is converted from dB to 8-bit so it looks like the training tiles. The script compares 4 settings, and grey-45 worked best.
- Detections on land are removed using the scene's `bathymetry.tif`.
- Known limitations:
  - The model can't tell ships from offshore structures such as wind turbines.
  - It misses many ships under 12 px.
