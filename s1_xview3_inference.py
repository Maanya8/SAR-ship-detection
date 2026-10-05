"""
Ship detection on a real Sentinel-1 scene (xView3), run locally on CPU.

Usage (all arguments optional; defaults shown):
    python s1_xview3_inference.py --scene 0157baf3866b2cf9v --labels validation.csv --weights best.pt
    python s1_xview3_inference.py --region full            # whole scene (~1500 tiles, slow)
    python s1_xview3_inference.py --labels none            # detection only, no scores

Steps: windowed read of VV_dB.tif -> pick the densest sea crop -> dB to 8-bit (4 settings compared,
the domain-shift step) -> 800 px tiles with 100 px overlap -> YOLO on CPU -> NMS across tiles ->
land mask from bathymetry.tif -> point matching against xView3 labels (200 m) -> pictures + CSVs.

Outputs in --out: metrics.txt, intensity_comparison.csv, recall_by_length.csv, detections.csv (lat/lon),
scene_overview.png, intensity_settings.png, overlay_region.png, overlay_zoom.png
"""
from pathlib import Path
import math, time, gc
import numpy as np, pandas as pd, cv2, matplotlib.pyplot as plt, torch, torchvision
import argparse
import matplotlib
matplotlib.use("Agg")
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform as warp_transform
from ultralytics import YOLO, settings
settings.update({"raytune": False, "mlflow": False, "comet": False, "clearml": False, "dvc": False, "wandb": False, "sync": False})

# ---------------- Settings (override from the command line) ----------------
ap = argparse.ArgumentParser(description="Ship detection on an xView3 Sentinel-1 scene with a YOLO model, on CPU.")
ap.add_argument("--scene", default="0157baf3866b2cf9v", help="unpacked xView3 scene folder (contains VV_dB.tif)")
ap.add_argument("--labels", default="validation.csv", help="xView3 labels CSV, or 'none'")
ap.add_argument("--weights", default="best.pt")
ap.add_argument("--out", default="s1_results")
ap.add_argument("--region", default="auto", help="'auto' (densest sea crop), 'full', or row0,col0,height,width")
ap.add_argument("--crop", type=int, default=4800, help="crop size in px for --region auto")
ap.add_argument("--conf", type=float, default=0.25)
ap.add_argument("--no-land-mask", action="store_true")
args = ap.parse_args()

SCENE_DIR   = Path(args.scene)
LABELS_CSV  = None if args.labels.lower() == "none" else Path(args.labels)
WEIGHTS     = Path(args.weights)
OUT         = Path(args.out)
REGION      = args.region if args.region in ("auto", "full") else tuple(int(v) for v in args.region.split(","))
CROP_SIZE   = args.crop
CONF        = args.conf
USE_LAND_MASK = not args.no_land_mask

TILE, OVERLAP = 800, 100   # tile size matches training; overlap so ships on edges are seen whole
CONF_MIN    = min(0.05, CONF)   # keep lower scores too, for the AP curve
NMS_IOU     = 0.4      # merge duplicates across overlapping tiles
CHUNK       = 8        # tiles per predict call (keeps RAM low)
MATCH_PX    = 20       # 20 px = 200 m, the xView3 matching distance

# dB -> 8-bit settings to try. "amp": amplitude scaled so the sea median lands at the given grey level
# (bright ships saturate, like LS-SSDD's 8-bit tiles). "db": linear in dB between lo and hi.
INTENSITY_SETTINGS = [("amp", 30), ("amp", 45), ("amp", 70), ("db", (-25.0, 0.0))]
# ------------------------------------------------------

OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(max(1, (torch.get_num_threads())))
print("torch threads:", torch.get_num_threads())


# ======================================================================
# 1. Look at the scene
# ======================================================================

VV_PATH = SCENE_DIR / "VV_dB.tif"
BATHY_PATH = SCENE_DIR / "bathymetry.tif"
assert VV_PATH.exists(), f"{VV_PATH} not found. Files in {SCENE_DIR}: {sorted(p.name for p in SCENE_DIR.glob('*'))}"

src = rasterio.open(VV_PATH)
H, W = src.height, src.width
print(f"VV_dB.tif: {W} x {H} px, dtype {src.dtypes[0]}, CRS {src.crs}, nodata {src.nodata}")
print("pixel size:", src.res)

def valid_mask(a):
    m = np.isfinite(a) & (a > -60) & (a < 60) & (a != 0)
    if src.nodata is not None:
        m &= a != src.nodata
    return m

# Low-res overview of the whole scene (1/16 size) for picking the crop and for display
DEC = 16
ov = src.read(1, out_shape=(H // DEC, W // DEC)).astype(np.float32)
ov_valid = valid_mask(ov)
vals = ov[ov_valid]
print(f"valid pixels: {ov_valid.mean():.0%}   dB percentiles 1/50/99: {np.percentile(vals, [1, 50, 99]).round(1)}")

plt.figure(figsize=(8, 7))
plt.imshow(np.where(ov_valid, ov, np.nan), cmap="gray", vmin=-25, vmax=0)
plt.title("Whole scene, VV dB (1/16 size)"); plt.colorbar(shrink=0.7)
plt.savefig(OUT / "scene_overview.png", dpi=120, bbox_inches="tight"); plt.close()


# ======================================================================
# 2. Load the ground-truth labels for this scene
# ======================================================================

scene_id = SCENE_DIR.name
gt = ign = None
if LABELS_CSV is not None and Path(LABELS_CSV).exists():
    lab = pd.read_csv(LABELS_CSV)
    lab = lab[lab["scene_id"].astype(str) == scene_id].copy()
    print(f"{len(lab)} labels for scene {scene_id}")

    def as_bool(s):
        return s.map(lambda v: {"true": True, "false": False, "1": True, "0": False, "1.0": True, "0.0": False}.get(str(v).strip().lower(), np.nan))
    lab["vessel"] = as_bool(lab["is_vessel"]) if "is_vessel" in lab else True
    conf_col = lab["confidence"].astype(str).str.upper() if "confidence" in lab else pd.Series("HIGH", index=lab.index)

    # Ships to find: not marked as infrastructure, HIGH/MEDIUM confidence (the xView3 scoring rule).
    # Infrastructure and LOW-confidence labels are "ignore": a detection there is neither right nor wrong.
    is_ship = (lab["vessel"] != False) & conf_col.isin(["HIGH", "MEDIUM"])
    gt  = lab[is_ship].rename(columns={"detect_scene_row": "r", "detect_scene_column": "c"})
    ign = lab[~is_ship].rename(columns={"detect_scene_row": "r", "detect_scene_column": "c"})
    print(f"ships to find: {len(gt)}   ignore (infrastructure / low confidence): {len(ign)}")
    if "vessel_length_m" in gt:
        print("vessel length (m) quartiles:", gt["vessel_length_m"].describe()[["25%", "50%", "75%"]].round(0).to_dict())
else:
    print("No labels: running detection only, no scores.")


# ======================================================================
# 3. Choose the region
# ======================================================================

def pick_auto_region():
    cs = min(CROP_SIZE, H, W)
    F = 100                                   # 100 px grid cells
    gh, gw = math.ceil(H / F), math.ceil(W / F)
    k = cs // F
    valid_grid = cv2.resize(ov_valid.astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA)
    counts = np.zeros((gh, gw), np.float32)
    if gt is not None and len(gt):
        np.add.at(counts, ((gt["r"].values // F).astype(int).clip(0, gh - 1), (gt["c"].values // F).astype(int).clip(0, gw - 1)), 1)
    def box_sum(a):
        s = np.pad(a.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        return s[k:, k:] - s[:-k, k:] - s[k:, :-k] + s[:-k, :-k]
    n_ships, frac_valid = box_sum(counts), box_sum(valid_grid) / (k * k)
    score = np.where(frac_valid > 0.9, n_ships + frac_valid * 1e-3, -1)   # most ships among mostly-valid windows
    i, j = np.unravel_index(np.argmax(score), score.shape)
    return int(i * F), int(j * F), cs, cs

if REGION == "auto":
    R0, C0, RH, RW = pick_auto_region()
elif REGION == "full":
    R0, C0, RH, RW = 0, 0, H, W
else:
    R0, C0, RH, RW = REGION
RH, RW = min(RH, H - R0), min(RW, W - C0)
print(f"region: rows {R0}-{R0 + RH}, cols {C0}-{C0 + RW}  (about {RW * 10 / 1000:.0f} x {RH * 10 / 1000:.0f} km at 10 m pixels)")

def in_region(df):
    return df[(df.r >= R0) & (df.r < R0 + RH) & (df.c >= C0) & (df.c < C0 + RW)].copy()
gt_r  = in_region(gt)  if gt  is not None else None
ign_r = in_region(ign) if ign is not None else None
if gt_r is not None:
    print(f"ships in region: {len(gt_r)}   ignore labels in region: {len(ign_r)}")


# ======================================================================
# 4. dB to 8-bit (the domain-shift step)
# ======================================================================

reg_win = Window(C0, R0, RW, RH)
dec = max(1, max(RH, RW) // 2000)
small = src.read(1, window=reg_win, out_shape=(max(1, RH // dec), max(1, RW // dec))).astype(np.float32)
SEA_MED_DB = float(np.median(small[valid_mask(small)]))
print(f"sea median in region: {SEA_MED_DB:.1f} dB")

def to_u8(db, setting):
    kind, p = setting
    v = valid_mask(db)
    if kind == "amp":
        out = p * np.power(10.0, (db - SEA_MED_DB) / 20.0)
    else:
        lo, hi = p
        out = (db - lo) / (hi - lo) * 255.0
    out = np.clip(np.nan_to_num(out), 0, 255)
    out[~v] = 0
    return out.astype(np.uint8)

def setting_name(s):
    return f"amp{s[1]}" if s[0] == "amp" else f"db{s[1][0]:g}to{s[1][1]:g}"

# Show a 400 px patch around the brightest spot, under each setting
yy, xx = np.unravel_index(np.nanargmax(np.where(valid_mask(small), small, -99)), small.shape)
pr, pc = min(max(R0 + yy * dec - 200, R0), R0 + RH - 400), min(max(C0 + xx * dec - 200, C0), C0 + RW - 400)
patch = src.read(1, window=Window(pc, pr, 400, 400)).astype(np.float32)
fig, ax = plt.subplots(1, len(INTENSITY_SETTINGS) + 1, figsize=(4 * (len(INTENSITY_SETTINGS) + 1), 4))
ax[0].hist(patch[valid_mask(patch)].ravel(), 100); ax[0].set_title("patch dB histogram")
for a, s in zip(ax[1:], INTENSITY_SETTINGS):
    u = to_u8(patch, s); a.imshow(u, cmap="gray", vmin=0, vmax=255); a.set_title(f"{setting_name(s)}  median {np.median(u):.0f}"); a.axis("off")
plt.tight_layout(); plt.savefig(OUT / "intensity_settings.png", dpi=110, bbox_inches="tight"); plt.close()


# ======================================================================
# 5. Tile, predict on CPU, merge across tiles
# ======================================================================

model = YOLO(str(WEIGHTS))

def starts(n):
    if n <= TILE: return [0]
    s = list(range(0, n - TILE, TILE - OVERLAP)); s.append(n - TILE)
    return sorted(set(s))

tiles = [(R0 + r, C0 + c) for r in starts(RH) for c in starts(RW)]
print(f"{len(tiles)} tiles of {TILE} px")

def read_tile(r, c):
    h, w = min(TILE, R0 + RH - r), min(TILE, C0 + RW - c)
    a = np.zeros((TILE, TILE), np.float32)
    a[:h, :w] = src.read(1, window=Window(c, r, w, h))
    return a

def detect(setting):
    boxes, scores = [], []
    t0 = time.time()
    batch, offs = [], []
    def flush():
        res = model.predict(batch, imgsz=TILE, conf=CONF_MIN, device="cpu", verbose=False)
        for (r, c), rr in zip(offs, res):
            b = rr.boxes
            if len(b):
                xyxy = b.xyxy.numpy().copy(); xyxy[:, [0, 2]] += c; xyxy[:, [1, 3]] += r
                boxes.append(xyxy); scores.append(b.conf.numpy())
        batch.clear(); offs.clear()
    for r, c in tiles:
        db = read_tile(r, c)
        if valid_mask(db).mean() < 0.05: continue           # skip no-data tiles
        u = to_u8(db, setting)
        batch.append(np.dstack([u, u, u])); offs.append((r, c))
        if len(batch) == CHUNK: flush()
    if batch: flush()
    gc.collect()
    if not boxes:
        return pd.DataFrame(columns=["x1", "y1", "x2", "y2", "score", "r", "c"]), time.time() - t0
    B, S = torch.from_numpy(np.concatenate(boxes)), torch.from_numpy(np.concatenate(scores))
    keep = torchvision.ops.nms(B, S, NMS_IOU).numpy()
    d = pd.DataFrame(B.numpy()[keep], columns=["x1", "y1", "x2", "y2"]); d["score"] = S.numpy()[keep]
    d["r"], d["c"] = (d.y1 + d.y2) / 2, (d.x1 + d.x2) / 2
    return d, time.time() - t0


# ======================================================================
# 6. Land mask
# ======================================================================

def on_land(rows, cols):
    if not (USE_LAND_MASK and BATHY_PATH.exists()) or len(rows) == 0:
        return np.zeros(len(rows), bool)
    with rasterio.open(BATHY_PATH) as b:
        xs, ys = rasterio.transform.xy(src.transform, np.asarray(rows), np.asarray(cols))
        if b.crs != src.crs:
            xs, ys = warp_transform(src.crs, b.crs, xs, ys)
        z = np.array([v[0] for v in b.sample(list(zip(xs, ys)))], np.float32)
    return z > 0

if gt_r is not None:
    land_gt = on_land(gt_r.r.values, gt_r.c.values)
    print(f"labelled ships on 'land' (likely harbours, excluded from scoring): {land_gt.sum()}")
    gt_r = gt_r[~land_gt].copy()


# ======================================================================
# 7. Scoring (point matching, as in xView3)
# ======================================================================

def score_dets(d):
    # Greedy by score: each prediction takes the nearest unmatched ship within MATCH_PX.
    # Predictions near an 'ignore' label are dropped (neither right nor wrong).
    d = d.sort_values("score", ascending=False).reset_index(drop=True)
    g = gt_r[["r", "c"]].values if len(gt_r) else np.zeros((0, 2))
    ig = ign_r[["r", "c"]].values if ign_r is not None and len(ign_r) else np.zeros((0, 2))
    used = np.zeros(len(g), bool); tp = np.zeros(len(d), bool); keep = np.ones(len(d), bool); match = -np.ones(len(d), int)
    for i, (r, c) in enumerate(d[["r", "c"]].values):
        if len(g):
            dist = np.hypot(g[:, 0] - r, g[:, 1] - c); dist[used] = np.inf
            j = int(np.argmin(dist))
            if dist[j] <= MATCH_PX:
                used[j] = True; tp[i] = True; match[i] = j; continue
        if len(ig) and np.hypot(ig[:, 0] - r, ig[:, 1] - c).min() <= MATCH_PX:
            keep[i] = False
    d["tp"], d["match"] = tp, match
    d = d[keep].reset_index(drop=True)
    # AP over all scores >= CONF_MIN
    ctp, cfp = np.cumsum(d.tp), np.cumsum(~d.tp)
    rec = ctp / max(len(g), 1); prec = ctp / np.maximum(ctp + cfp, 1)
    mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[1], prec, [0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    ap = float(np.sum((mrec[1:] - mrec[:-1]) * mpre[1:]))
    a = d[d.score >= CONF]
    TP = int(a.tp.sum()); FP = len(a) - TP; FN = len(g) - TP
    P = TP / max(TP + FP, 1); R = TP / max(len(g), 1); F1 = 2 * P * R / max(P + R, 1e-9)
    return dict(P=P, R=R, F1=F1, AP=ap, TP=TP, FP=FP, FN=FN), d


# ======================================================================
# 8. Run every intensity setting and compare
# ======================================================================

rows, all_dets = [], {}
for s in INTENSITY_SETTINGS:
    d, secs = detect(s)
    land = on_land(d.r.values, d.c.values); d = d[~land].reset_index(drop=True)
    row = dict(setting=setting_name(s), detections=int((d.score >= CONF).sum()), seconds=round(secs, 1), land_dropped=int(land.sum()))
    if gt_r is not None:
        m, d = score_dets(d); row.update({k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()})
    rows.append(row); all_dets[setting_name(s)] = (s, d)
    print(row)
summary = pd.DataFrame(rows)
summary.to_csv(OUT / "intensity_comparison.csv", index=False)

key = "F1" if gt_r is not None else "detections"
best_name = summary.sort_values(key, ascending=False).iloc[0]["setting"]
BEST_SETTING, dets = all_dets[best_name]
print("chosen setting:", best_name, "(highest F1)" if gt_r is not None else "")
dets_c = dets[dets.score >= CONF].copy()

# lat/lon for each detection
xs, ys = rasterio.transform.xy(src.transform, dets_c.r.values, dets_c.c.values) if len(dets_c) else ([], [])
if len(dets_c) and src.crs and str(src.crs) != "EPSG:4326":
    xs, ys = warp_transform(src.crs, "EPSG:4326", xs, ys)
dets_c["lon"], dets_c["lat"] = xs, ys
px_m = 10.0
dets_c["length_m_est"] = (np.maximum(dets_c.x2 - dets_c.x1, dets_c.y2 - dets_c.y1) * px_m).round(0)
dets_c.drop(columns=[c for c in ["match"] if c in dets_c]).to_csv(OUT / "detections.csv", index=False)
print(f"{len(dets_c)} detections saved to {OUT / 'detections.csv'}")


# ======================================================================
# 9. Recall by vessel length
# ======================================================================

if gt_r is not None and "vessel_length_m" in gt_r and len(gt_r):
    found = np.zeros(len(gt_r), bool)
    hit = dets[(dets.score >= CONF) & dets.tp]["match"].values
    found[hit[hit >= 0]] = True
    L = gt_r["vessel_length_m"].values
    bins = [(0, 25), (25, 50), (50, 100), (100, 200), (200, 1e9)]
    rows = []
    for lo, hi in bins:
        sel = (L >= lo) & (L < hi)
        rows.append(dict(length_m=f"{lo}-{hi if hi < 1e9 else ''}", ships=int(sel.sum()), recall=round(found[sel].mean(), 2) if sel.sum() else None))
    unk = ~np.isfinite(L)
    rows.append(dict(length_m="unknown", ships=int(unk.sum()), recall=round(found[unk].mean(), 2) if unk.sum() else None))
    by_len = pd.DataFrame(rows); by_len.to_csv(OUT / "recall_by_length.csv", index=False); print(by_len.to_string(index=False))


# ======================================================================
# 10. Pictures: whole region, then zoomed panels
# ======================================================================

# Whole region at reduced size: green = labelled ship, red = detection, yellow = missed ship
img = to_u8(small, BEST_SETTING)
vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
def P(r, c): return int((c - C0) / dec), int((r - R0) / dec)
if gt_r is not None:
    found_idx = set(dets[(dets.score >= CONF) & dets.tp]["match"].values.tolist())
    for j, (r, c) in enumerate(gt_r[["r", "c"]].values):
        cv2.circle(vis, P(r, c), 6, (0, 255, 0) if j in found_idx else (0, 255, 255), 1)
for x1, y1, x2, y2 in dets_c[["x1", "y1", "x2", "y2"]].values:
    a, b = P(y1, x1); e, f = P(y2, x2); cv2.rectangle(vis, (a - 2, b - 2), (e + 2, f + 2), (0, 0, 255), 1)
cv2.imwrite(str(OUT / "overlay_region.png"), vis)
plt.figure(figsize=(11, 11)); plt.imshow(vis[..., ::-1]); plt.axis("off")
plt.title(f"{scene_id}: red = detection, green = labelled ship found, yellow = labelled ship missed"); plt.close()

# Zoomed 600 px panels around the busiest spots, full resolution
Z = 600
pts = dets_c[["r", "c"]].values if len(dets_c) else (gt_r[["r", "c"]].values if gt_r is not None else np.zeros((0, 2)))
centres = []
for r, c in pts[np.argsort(-np.array([((np.abs(pts - p) < Z / 2).all(1)).sum() for p in pts]))] if len(pts) else []:
    if all(abs(r - a) > Z or abs(c - b) > Z for a, b in centres):
        centres.append((r, c))
    if len(centres) == 4: break
if centres:
    fig, ax = plt.subplots(1, len(centres), figsize=(5 * len(centres), 5))
    ax = np.atleast_1d(ax)
    for a, (r, c) in zip(ax, centres):
        r0 = int(min(max(r - Z / 2, R0), R0 + RH - Z)); c0 = int(min(max(c - Z / 2, C0), C0 + RW - Z))
        u = cv2.cvtColor(to_u8(src.read(1, window=Window(c0, r0, Z, Z)).astype(np.float32), BEST_SETTING), cv2.COLOR_GRAY2BGR)
        if gt_r is not None:
            for j, (gr, gc_) in enumerate(gt_r[["r", "c"]].values):
                if r0 <= gr < r0 + Z and c0 <= gc_ < c0 + Z:
                    cv2.circle(u, (int(gc_ - c0), int(gr - r0)), 14, (0, 255, 0) if j in found_idx else (0, 255, 255), 2)
        for x1, y1, x2, y2, s in dets_c[["x1", "y1", "x2", "y2", "score"]].values:
            if r0 <= (y1 + y2) / 2 < r0 + Z and c0 <= (x1 + x2) / 2 < c0 + Z:
                cv2.rectangle(u, (int(x1 - c0), int(y1 - r0)), (int(x2 - c0), int(y2 - r0)), (0, 0, 255), 2)
                cv2.putText(u, f"{s:.2f}", (int(x1 - c0), int(y1 - r0) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        a.imshow(u[..., ::-1]); a.set_title(f"rows {r0}, cols {c0}"); a.axis("off")
    plt.tight_layout(); plt.savefig(OUT / "overlay_zoom.png", dpi=120, bbox_inches="tight"); plt.close()


# ======================================================================
# 11. Summary
# ======================================================================

lines = [f"scene {scene_id}  region rows {R0}-{R0 + RH} cols {C0}-{C0 + RW}  tiles {len(tiles)}",
         f"sea median {SEA_MED_DB:.1f} dB   chosen intensity setting: {best_name}   conf {CONF}   match {MATCH_PX} px",
         "", summary.to_string(index=False)]
if gt_r is not None:
    lines += ["", "Note: the intensity setting was chosen on these same labels, so treat the chosen row as slightly optimistic."]
(OUT / "metrics.txt").write_text("\n".join(lines))
print("\n".join(lines))
print("\nSaved:", sorted(p.name for p in OUT.iterdir()))
