# -*- coding: utf-8 -*-
"""
SAM-3 multi-concept inference on a single tile raster → GeoJSON (EPSG:4326)
Updated v4.8.9 (2025-12-18)

Robustness fixes v4.8.9:
- Removed geometry simplification (no simplify_m; no EPSG:3857 projection step).
- Dissolve per concept with unary_union so polygons of the same concept do not overlap.
- Strict JSON output (allow_nan=False) to prevent Infinity/NaN tokens.
- Exclusion mask still treats any value >0 as excluded.
- Geodesic area handles Polygon & MultiPolygon with holes.

"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import math
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import rasterio
from rasterio.features import shapes as rio_shapes
from rasterio.enums import Resampling
from shapely.geometry import shape as shp_shape, mapping as shp_mapping
from shapely.ops import transform as shp_transform, unary_union
from shapely.geometry import Polygon, MultiPolygon
from pyproj import CRS, Transformer, Geod

# Try to import make_valid (Shapely 2.x). Fallback to buffer(0)
try:
    from shapely.validation import make_valid as _make_valid
except Exception:
    _make_valid = None

# Ultralytics SAM-3 predictor
from ultralytics.models.sam.predict import SAM3SemanticPredictor

# ---- constants ----
GEOD = Geod(ellps="WGS84")
WGS84 = CRS.from_epsg(4326)

# ------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in_tif", required=True)
    p.add_argument("--out_geojson", required=True)
    p.add_argument("--concepts_json", required=True,
                   help="JSON dict: {key:{name,prompt,min_area_m2,min_area_px}}")
    p.add_argument("--model", required=True)
    p.add_argument("--bpe_path", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--half", action="store_true")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.28)
    p.add_argument("--tile_id", type=int, default=-1)
    p.add_argument("--exclude_mask_tif", default=None, help="uint8 mask tile; exclude where value>0")
    p.add_argument("--src_epsg", type=int, default=None,
                   help="Override source EPSG when input raster CRS is missing/unknown (e.g., 3826 for TWD97 TM2 zone 121)")
    p.add_argument("--no_plot", action="store_true")
    p.add_argument("--no_save", action="store_true")
    p.add_argument("--log_json", default=None, help="write run log to this path")
    return p.parse_args()

# ------------------
def ensure_rgb_uint8(arr_bhw: np.ndarray) -> np.ndarray:
    if arr_bhw.ndim != 3:
        raise ValueError("Expected (bands,h,w) array from rasterio.read()")
    b, h, w = arr_bhw.shape
    if b >= 3:
        rgb = np.stack([arr_bhw[0], arr_bhw[1], arr_bhw[2]], axis=-1)
    else:
        g = arr_bhw[0]
        rgb = np.stack([g, g, g], axis=-1)
    rgb = rgb.astype(np.float32, copy=False)
    if np.isnan(rgb).any():
        rgb = np.nan_to_num(rgb, nan=0.0)
    try:
        cap = float(np.nanpercentile(rgb, 99.0)) if rgb.size else 0.0
    except Exception:
        cap = 0.0
    if cap <= 0:
        rgb = np.clip(rgb, 0, 255)
        out = rgb.astype(np.uint8)
    else:
        rgb = np.clip(rgb, 0, cap)
        rgb = (rgb / cap) * 255.0
        out = np.clip(rgb, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(out)

# ------------------
def read_exclusion_mask(path: Optional[str], out_hw: Tuple[int, int]) -> Optional[np.ndarray]:
    if not path:
        return None
    with rasterio.open(path) as src:
        m = src.read(1, out_shape=out_hw, resampling=Resampling.nearest)
    return (m.astype(np.float32) > 0)

# ------------------
def build_predictor(model_path: str, bpe_path: str, device: str, imgsz: int,
                   half: bool, no_plot: bool, no_save: bool, conf: float):
    overrides = dict(task="segment", mode="predict", model=model_path,
                     device=device, imgsz=imgsz, half=bool(half), verbose=False, conf=conf)
    if no_plot:
        overrides.update(dict(show=False))
    if no_save:
        overrides.update(dict(save=False, save_txt=False, save_conf=False, save_crop=False))
    try:
        pred = SAM3SemanticPredictor(overrides=overrides, bpe_path=bpe_path)
    except TypeError:
        pred = SAM3SemanticPredictor(overrides=overrides)
    for obj in (pred, getattr(pred, 'args', None)):
        if obj is None:
            continue
        if hasattr(obj, 'bpe_path'):
            try:
                setattr(obj, 'bpe_path', bpe_path)
            except Exception:
                pass
    if hasattr(pred, "args"):
        for k, v in {"verbose": False, "show": False, "save": False,
                     "save_txt": False, "save_conf": False, "save_crop": False, "conf": conf}.items():
            if hasattr(pred.args, k):
                try:
                    setattr(pred.args, k, v)
                except Exception:
                    pass
    return pred

# ------------------
def extract_masks_and_scores(result):
    r = result[0] if isinstance(result, (list, tuple)) and len(result) else result
    masks = None
    scores = None
    if getattr(r, "masks", None) is not None and getattr(r.masks, "data", None) is not None:
        masks = r.masks.data
        try:
            masks = masks.detach().float().cpu().numpy()
        except Exception:
            masks = np.asarray(masks, dtype=np.float32)
        if masks.ndim == 2:
            masks = masks[None, ...]
        elif masks.ndim != 3:
            masks = None
    if getattr(r, "boxes", None) is not None and getattr(r.boxes, "conf", None) is not None:
        try:
            scores = r.boxes.conf.detach().float().cpu().numpy().tolist()
        except Exception:
            try:
                scores = list(r.boxes.conf)
            except Exception:
                scores = None
    if masks is not None and scores is None:
        scores = [None] * int(masks.shape[0])
    return masks, scores

# ------------------
def _has_nonfinite_coords(g) -> bool:
    try:
        geoms = list(g.geoms) if hasattr(g, 'geoms') else [g]
        for gg in geoms:
            rings = [gg.exterior] + list(getattr(gg, 'interiors', []))
            for ring in rings:
                for x, y in ring.coords:
                    if not (math.isfinite(x) and math.isfinite(y)):
                        return True
        return False
    except Exception:
        return True

# ------------------
def _make_geometry_valid(g):
    try:
        if g.is_valid:
            return g
    except Exception:
        pass
    if _make_valid is not None:
        try:
            return _make_valid(g)
        except Exception:
            pass
    try:
        return g.buffer(0)
    except Exception:
        return g

# ------------------
def _clamp_lat_wgs84(g):
    """Clamp latitudes to [-85,85] to ensure valid WebMercator projection."""
    try:
        return shp_transform(lambda x, y, z=None: (x, max(min(y, 85.0), -85.0)), g)
    except Exception:
        return g

# ------------------
def to_wgs84_geom(geom: Dict[str, Any], src_crs: Optional[CRS]) -> Dict[str, Any]:
    """Reproject a GeoJSON geometry dict to EPSG:4326.
    IMPORTANT: do NOT clamp latitudes before reprojection when the source CRS is projected
    (e.g., TWD97 TM2 EPSG:3826), otherwise coordinates in meters will be destroyed.
    """
    if not src_crs or CRS.from_user_input(src_crs) == WGS84:
        return geom

    src_crs = CRS.from_user_input(src_crs)
    tf = Transformer.from_crs(src_crs, WGS84, always_xy=True)

    g = shp_shape(geom)
    g = _make_geometry_valid(g)

    # Reproject first
    g2 = shp_transform(lambda x, y, z=None: tf.transform(x, y), g)

    # Clamp only in WGS84 degrees (optional safety for any downstream web-mercator ops)
    g2 = _clamp_lat_wgs84(g2)

    return shp_mapping(g2)

# ------------------
def _ring_area_m2(lonlat_ring: List[Tuple[float, float]]) -> float:
    if len(lonlat_ring) < 4:
        return 0.0
    lons, lats = zip(*lonlat_ring)
    area, _ = GEOD.polygon_area_perimeter(lons, lats)
    return float(area)


def polygon_area_m2_wgs84(geom4326: Dict[str, Any]) -> float:
    try:
        g = shp_shape(geom4326)
        total = 0.0
        if isinstance(g, Polygon):
            total += _ring_area_m2(list(g.exterior.coords))
            for hole in g.interiors:
                total += _ring_area_m2(list(hole.coords))
        elif isinstance(g, MultiPolygon):
            for poly in g.geoms:
                total += _ring_area_m2(list(poly.exterior.coords))
                for hole in poly.interiors:
                    total += _ring_area_m2(list(hole.coords))
        else:
            coords = geom4326.get("coordinates", [])
            if not coords:
                return 0.0
            ring = coords[0]
            total += _ring_area_m2(ring)
        return abs(float(total))
    except Exception:
        return 0.0

# ------------------
def pixel_m2_from_transform(src_transform: rasterio.Affine, src_crs: Optional[CRS]) -> Optional[float]:
    if src_transform is None or src_crs is None:
        return None
    try:
        a = abs(src_transform.a)
        e = abs(src_transform.e)
        if CRS.from_user_input(src_crs).is_geographic:
            return None
        return float(a * e)
    except Exception:
        return None

# ------------------
def main():
    args = parse_args()
    t0 = time.time()
    log_obj = {"tile_id": args.tile_id, "in_tif": args.in_tif, "out_geojson": args.out_geojson}

    with open(args.concepts_json, "r", encoding="utf-8") as f:
        concepts = json.load(f)
    if not isinstance(concepts, dict) or not concepts:
        raise RuntimeError("concepts_json must be a non-empty dict")

    predictor = build_predictor(args.model, args.bpe_path, args.device, args.imgsz,
                                args.half, args.no_plot, args.no_save, args.conf)
    # v4.8.9: accumulate geometries per concept and dissolve to remove overlaps
    concept_acc: Dict[str, Dict[str, Any]] = {}
    src_name = os.path.basename(args.in_tif)

    with rasterio.open(args.in_tif) as src:
        arr = src.read()
        rgb = ensure_rgb_uint8(arr)
        predictor.set_image(rgb)

        excl = read_exclusion_mask(args.exclude_mask_tif, (rgb.shape[0], rgb.shape[1]))
        src_crs = src.crs
        # If CRS is missing/unknown, allow user override via --src_epsg
        if (src_crs is None or (hasattr(src_crs, 'to_epsg') and src_crs.to_epsg() is None)) and args.src_epsg:
            try:
                src_crs = CRS.from_epsg(int(args.src_epsg))
            except Exception:
                src_crs = src.crs
        src_transform = src.transform
        px_m2 = pixel_m2_from_transform(src_transform, src_crs)

        for key, cfg in concepts.items():
            prompt = str(cfg.get("prompt", key))
            out_name = str(cfg.get("name", f"sam3_{key}"))
            min_area_m2 = float(cfg.get("min_area_m2", 0) or 0)
            min_area_px = int(cfg.get("min_area_px", 0) or 0)
            # (simplify removed in v4.8.9)
            try:
                res = predictor(text=[prompt])
            except TypeError:
                res = predictor(text=[prompt], bpe_path=args.bpe_path)
            masks, scores = extract_masks_and_scores(res)
            if masks is None or int(masks.shape[0]) == 0:
                continue

            for i in range(int(masks.shape[0])):
                m = masks[i]
                if excl is not None:
                    m = np.where(excl, 0.0, m)
                mask_bin = (m > 0.5).astype(np.uint8)

                for geom, val in rio_shapes(mask_bin, mask=None, transform=src_transform):
                    if val != 1:
                        continue
                    geom4326 = to_wgs84_geom(geom, CRS.from_user_input(src_crs) if src_crs else None)
                    area_m2 = polygon_area_m2_wgs84(geom4326)

                    if min_area_m2 > 0 and area_m2 < min_area_m2:
                        continue
                    if min_area_m2 == 0 and min_area_px > 0:
                        if px_m2 is not None:
                            if area_m2 < (min_area_px * px_m2):
                                continue
                        else:
                            if int(np.sum(mask_bin)) < min_area_px:
                                continue

                    # v4.8.9: no geometry simplify; keep original WGS84 geometry
                    geom_simpl = geom4326
                    # sanity check (skip empty/non-finite geometries)
                    try:
                        gchk = shp_shape(geom_simpl)
                        if gchk.is_empty or _has_nonfinite_coords(gchk):
                            continue
                    except Exception:
                        continue                    # Accumulate per-concept geometries (avoid overlaps via unary_union later)
                    sc = float(scores[i]) if scores is not None and i < len(scores) and scores[i] is not None else None
                    try:
                        g_obj = shp_shape(geom_simpl)
                        g_obj = _make_geometry_valid(g_obj)
                        if g_obj.is_empty or _has_nonfinite_coords(g_obj):
                            continue
                    except Exception:
                        continue
                    acc = concept_acc.setdefault(out_name, {
                        "concept": out_name,
                        "concept_key": str(key),
                        "prompt": prompt,
                        "scores": [],
                        "geoms": []
                    })
                    acc["geoms"].append(g_obj)
                    if sc is not None:
                        acc["scores"].append(sc)    # Build final (non-overlapping) features per concept via unary_union
    features: List[Dict[str, Any]] = []
    for cname, acc in concept_acc.items():
        geoms = acc.get("geoms", [])
        if not geoms:
            continue
        try:
            u = unary_union(geoms)
            u = _make_geometry_valid(u)
        except Exception:
            # If union fails, fall back to individual valid geometries (may still overlap)
            u = geoms

        # Normalize to list of polygons
        parts = []
        if isinstance(u, (list, tuple)):
            parts = list(u)
        else:
            try:
                if hasattr(u, "geoms"):
                    parts = [g for g in u.geoms]
                else:
                    parts = [u]
            except Exception:
                parts = []

        score_max = max(acc.get("scores", []), default=None)

        for part in parts:
            try:
                part = _make_geometry_valid(part)
                if part.is_empty or _has_nonfinite_coords(part):
                    continue
            except Exception:
                continue

            geom_out = shp_mapping(part)
            area_m2 = polygon_area_m2_wgs84(geom_out)
            props = {
                "concept": acc.get("concept"),
                "concept_key": acc.get("concept_key"),
                "prompt": acc.get("prompt"),
                "score": score_max,
                "area_m2": float(area_m2),
                "tile_id": int(args.tile_id),
                "source": src_name,
            }
            features.append({"type": "Feature", "geometry": geom_out, "properties": props})

    out_fc = {"type": "FeatureCollection", "name": f"sam3_tile_{args.tile_id:06d}", "features": features}
    os.makedirs(os.path.dirname(args.out_geojson), exist_ok=True)
    with open(args.out_geojson, "w", encoding="utf-8") as f:
        json.dump(out_fc, f, ensure_ascii=False, allow_nan=False)

    log_obj["n_features"] = len(features)
    log_obj["seconds"] = round(time.time() - t0, 3)
    if args.log_json:
        os.makedirs(os.path.dirname(args.log_json), exist_ok=True)
        with open(args.log_json, "w", encoding="utf-8") as f:
            json.dump(log_obj, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"OK tile_id={args.tile_id} features={len(features)} seconds={log_obj['seconds']}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
