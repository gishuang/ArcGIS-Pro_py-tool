# -*- coding: utf-8 -*-
"""
SAM-3 Extraction Toolbox (.pyt) — v4.8.9
ArcGIS Pro Python Toolbox
多分類文字提示分割：逐格(tile)推論 → 合併 GeoJSON → 匯入 GDB 並依概念拆分

修正與對齊重點：
- 版本標示/alias 對齊 v4.8.9
- 預設推論腳本路徑小拼字修正 (sam3_tools)
- 僅在投影座標(公尺)時計算 min_area_px；經緯度以 m² 為主、px 為 0
- run 資料夾名稱對齊 v4.8.9
- 合併後若有 features 才繼續匯入 GDB（修正不必要的提前 return）

注意：此工具箱會呼叫外部 Python (建議 GPU conda env) 執行 sam3_infer_multiconcept_tile_geojson_v4_8_9.py。
"""
from __future__ import annotations
import os, json, math, time, datetime, subprocess
from typing import Dict, Any, List, Optional
import arcpy

# ---- Defaults (請依實際環境調整) ----
DEFAULT_EXT_PYTHON = r"C:\\miniconda3\\envs\\sam3-gpu\\python.exe"
DEFAULT_INFER_PY   = r"C:\\sam3_tools\\sam3_infer_multiconcept_tile_geojson_v4_8_9.py"  # 修正 sma3 -> sam3
DEFAULT_MODEL      = r"C:\\sam3_assets\\sam3.pt"
DEFAULT_BPE        = r"C:\\sam3_assets\\bpe_simple_vocab_16e6.txt.gz"

# ---- Utilities ----
def _now_tag():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _unique_fc_path(out_gdb: str, base_name: str) -> str:
    """Return an available FC path in out_gdb. If base exists (or is locked), append _001, _002, ..."""
    # base_name is a feature class name (no path)
    cand = os.path.join(out_gdb, base_name)
    if not arcpy.Exists(cand):
        return cand
    # Try delete first (if not locked)
    try:
        arcpy.management.Delete(cand)
        return cand
    except Exception:
        pass
    for k in range(1, 1000):
        name2 = f"{base_name}_{k:03d}"
        cand2 = os.path.join(out_gdb, name2)
        if not arcpy.Exists(cand2):
            return cand2
    raise RuntimeError(f"Unable to find available output name under {out_gdb} for base {base_name}")

def _write_text(path: str, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", errors="replace") as f:
        f.write(text)



def _is_finite_number(x) -> bool:
    try:
        return isinstance(x, (int, float)) and math.isfinite(x)
    except Exception:
        return False


def _coords_all_finite(coords) -> bool:
    """Recursively validate GeoJSON coordinates."""
    if coords is None:
        return False
    # Leaf: [x, y] or [x, y, z]
    if isinstance(coords, (list, tuple)) and coords and all(not isinstance(c, (list, tuple)) for c in coords):
        return all(_is_finite_number(c) for c in coords)
    # Nested lists
    if isinstance(coords, (list, tuple)):
        return all(_coords_all_finite(c) for c in coords)
    return False


def _sanitize_geojson(gj: dict):
    """
    Remove features with non-finite coordinates (NaN/Inf) and ensure output is strict JSON.
    Returns (sanitized_geojson, dropped_count).
    """
    dropped = 0
    out = dict(gj)
    feats = gj.get("features") or []
    clean_feats = []
    for f in feats:
        try:
            geom = f.get("geometry") if isinstance(f, dict) else None
            if not geom or not isinstance(geom, dict):
                dropped += 1
                continue
            coords = geom.get("coordinates", None)
            if not _coords_all_finite(coords):
                dropped += 1
                continue
            clean_feats.append(f)
        except Exception:
            dropped += 1
    out["features"] = clean_feats
    return out, dropped
def _clean_env():
    env = os.environ.copy()
    for k in ["PYTHONHOME", "PYTHONPATH", "PYTHONUSERBASE", "CONDA_DEFAULT_ENV", "CONDA_SHLVL"]:
        env.pop(k, None)
    return env

def _env_for_python(python_exe: str) -> Dict[str, str]:
    env = _clean_env()
    env_root = os.path.dirname(python_exe)
    lib_bin = os.path.join(env_root, "Library", "bin")
    scripts = os.path.join(env_root, "Scripts")
    old_path = env.get("PATH", "")
    prepend = [p for p in [lib_bin, scripts, env_root] if os.path.isdir(p)]
    env["PATH"] = os.pathsep.join(prepend + [old_path])
    # 性能/相容性環境變數
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_THREADING_LAYER"] = "GNU"
    env["MKL_SERVICE_FORCE_INTEL"] = "1"
    env["PYTHONUTF8"] = "1"
    return env

class Toolbox(object):
    def __init__(self):
        self.label = "SAM-3 Extraction Toolbox (v4.8.9)"
        self.alias = "sam3_extract_v487"
        self.tools = [SAM3Extract]

class SAM3Extract(object):
    def __init__(self):
        self.label = "SAM-3 Multi-Concept Extraction (Tile GeoJSON + Merge + GDB) v4.8.9"
        self.description = "Run SAM-3 external inference per tile, merge GeoJSON, import to GDB FCs."
        self.canRunInBackground = False

    def getParameterInfo(self):
        p = []
        p.append(arcpy.Parameter(
            displayName="Input Raster / Mosaic / Image Service Layer",
            name="in_raster",
            datatype="GPRasterLayer",
            parameterType="Required",
            direction="Input"
        ))
        p.append(arcpy.Parameter(
            displayName="AOI (optional polygon feature/layer)",
            name="aoi",
            datatype="GPFeatureLayer",
            parameterType="Optional",
            direction="Input"
        ))
        p.append(arcpy.Parameter(
            displayName="Water Exclusion Mask Raster (optional; water=1 excluded, others=0)",
            name="water_mask",
            datatype="GPRasterLayer",
            parameterType="Optional",
            direction="Input"
        ))
        p.append(arcpy.Parameter(
            displayName="Output Workspace (GDB)",
            name="out_gdb",
            datatype="DEWorkspace",
            parameterType="Required",
            direction="Input"
        ))
        p.append(arcpy.Parameter(
            displayName="Output Feature Class Prefix (manual input; e.g., sam3run1)",
            name="out_prefix",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        ))
        p[-1].value = "sam3"
        p.append(arcpy.Parameter(
            displayName="Output Name Suffix (optional; blank = auto timestamp)",
            name="out_suffix",
            datatype="GPString",
            parameterType="Optional",
            direction="Input"
        ))
        p.append(arcpy.Parameter(
            displayName="Output Folder (run artifacts)",
            name="out_folder",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input"
        ))
        p.append(arcpy.Parameter(
            displayName="External Python (conda env; GPU recommended)",
            name="ext_python",
            datatype="DEFile",
            parameterType="Required",
            direction="Input"
        ))
        p[-1].value = DEFAULT_EXT_PYTHON
        p.append(arcpy.Parameter(
            displayName="Infer script (tile GeoJSON)",
            name="infer_py",
            datatype="DEFile",
            parameterType="Required",
            direction="Input"
        ))
        p[-1].value = DEFAULT_INFER_PY
        p.append(arcpy.Parameter(
            displayName="SAM-3 model weights (sam3.pt)",
            name="model",
            datatype="DEFile",
            parameterType="Required",
            direction="Input"
        ))
        p[-1].value = DEFAULT_MODEL
        p.append(arcpy.Parameter(
            displayName="BPE vocab path",
            name="bpe",
            datatype="DEFile",
            parameterType="Required",
            direction="Input"
        ))
        p[-1].value = DEFAULT_BPE
        p.append(arcpy.Parameter(
            displayName="Device (auto/cuda/cuda:0/cpu)",
            name="device",
            datatype="GPString",
            parameterType="Optional",
            direction="Input"
        ))
        p[-1].value = "cuda:0"
        p.append(arcpy.Parameter(
            displayName="Use FP16 half (GPU only)",
            name="half",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input"
        ))
        p[-1].value = True
        p.append(arcpy.Parameter(
            displayName="imgsz (Ultralytics inference, px)",
            name="imgsz",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input"
        ))
        p[-1].value = 640
        p.append(arcpy.Parameter(
            displayName="Tile Size (px)",
            name="tile_px",
            datatype="GPLong",
            parameterType="Required",
            direction="Input"
        ))
        p[-1].value = 768
        p.append(arcpy.Parameter(
            displayName="Overlap (px)",
            name="overlap_px",
            datatype="GPLong",
            parameterType="Required",
            direction="Input"
        ))
        p[-1].value = 96
        p.append(arcpy.Parameter(
            displayName="Processing Cell Size (map units; blank = use source)",
            name="cellsize",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input"
        ))
        p.append(arcpy.Parameter(
            displayName="Confidence (SAM-3 conf)",
            name="conf",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input"
        ))
        p[-1].value = 0.28
        p.append(arcpy.Parameter(
            displayName="Extra prompts (optional; comma separated)",
            name="extra_prompts",
            datatype="GPString",
            parameterType="Optional",
            direction="Input"
        ))
        p[-1].value = ""


        # Source EPSG override (use when tile GeoTIFF has no CRS; e.g., 3826 for TWD97 TM2 zone 121)
        p.append(arcpy.Parameter(
            displayName="Source EPSG override (use when tile has no CRS; e.g., 3826 for TWD97 TM2 zone 121)",
            name="src_epsg",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input"
        ))
        p[-1].value = 3826
        return p

    def _pretty_cmd(self, cmd: List[str]) -> str:
        def q(s: str) -> str:
            if any(ch in s for ch in [' ', '\t', '"']):
                return '"' + s.replace('"', r'\\"') + '"'
            return s
        return " ".join(q(str(x)) for x in cmd)

    def _run_external(self, cmd: List[str], messages, python_exe_for_env: str, log_path: str) -> int:
        env = _env_for_python(python_exe_for_env)
        messages.addMessage("External CMD:\n" + self._pretty_cmd(cmd))
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True, env=env)
        dt = time.time() - t0
        full = (
            "CMD:\n" + self._pretty_cmd(cmd) + "\n\n"
            f"RETURN CODE: {p.returncode}\n"
            f"SECONDS: {dt:.3f}\n\n"
            "STDOUT:\n" + (p.stdout or "") + "\n\n"
            "STDERR:\n" + (p.stderr or "") + "\n"
        )
        _write_text(log_path, full)
        if p.stdout and p.stdout.strip():
            s = p.stdout.strip()
            messages.addMessage(s if len(s) <= 6000 else (s[:6000] + "\n... (stdout truncated; see log)"))
        if p.stderr and p.stderr.strip():
            s = p.stderr.strip()
            messages.addWarningMessage(s if len(s) <= 8000 else (s[:8000] + "\n... (stderr truncated; see log)"))
        if p.returncode != 0:
            messages.addWarningMessage(f"External failed (rc={p.returncode}). Log: {log_path}")
        return p.returncode

    def execute(self, parameters, messages):
        in_raster    = parameters[0].valueAsText
        aoi          = parameters[1].valueAsText
        water_mask   = parameters[2].valueAsText
        out_gdb      = parameters[3].valueAsText
        out_prefix   = (parameters[4].valueAsText or "").strip()
        out_suffix   = (parameters[5].valueAsText or "").strip()
        out_folder   = parameters[6].valueAsText
        ext_python   = parameters[7].valueAsText
        infer_py     = parameters[8].valueAsText
        model        = parameters[9].valueAsText
        bpe          = parameters[10].valueAsText
        device       = (parameters[11].valueAsText or "auto").strip()
        half         = bool(parameters[12].value) if parameters[12].value is not None else False
        imgsz        = int(parameters[13].value) if parameters[13].value is not None else 640
        tile_px      = int(parameters[14].value)
        overlap_px   = int(parameters[15].value)
        cellsize     = parameters[16].value
        conf         = float(parameters[17].value) if parameters[17].value is not None else 0.28
        extra_prompts= (parameters[18].valueAsText or "").strip()
        src_epsg     = parameters[19].value

        if not out_prefix:
            raise RuntimeError("Output Feature Class Prefix (out_prefix) is required.")

        # 驗證基本路徑
        for label, path in [("ext_python", ext_python), ("infer_py", infer_py), ("model", model), ("bpe", bpe)]:
            if not path or not os.path.exists(path):
                raise RuntimeError(f"Missing or not found: {label} = {path}")
        os.makedirs(out_folder, exist_ok=True)
        arcpy.env.overwriteOutput = True

        # 來源解析度（cellsize）
        ras = arcpy.Raster(in_raster)
        if cellsize is None:
            try:
                cellsize = float(abs(ras.meanCellWidth))
            except Exception:
                cellsize = 0.5
        cellsize = float(cellsize)

        # 檢查座標系是否為投影（公尺）
        sr = arcpy.Describe(in_raster).spatialReference
        meters_per_unit = float(getattr(sr, "metersPerUnit", 0) or 0)
        is_projected_m = (getattr(sr, "type", None) == "Projected") and meters_per_unit > 0

        # ---- Concepts dict（可依需求調整） ----
        concepts: Dict[str, Dict[str, Any]] = {
            "road":     {"name": "sam3_road",     "prompt": "road surface, road markings",                      "min_area_m2": 30},
            "building": {"name": "sam3_building", "prompt": "buildings, rooftops",                               "min_area_m2": 30},
            "tree":     {"name": "sam3_tree",     "prompt": "trees, individual tree crowns",                    "min_area_m2": 5},
            "forest":   {"name": "sam3_forest",   "prompt": "forest canopy, dense trees",                        "min_area_m2": 200},
            "solar":    {"name": "sam3_solar",    "prompt": "solar panels, photovoltaic farm",                   "min_area_m2": 20},
            "wind":     {"name": "sam3_wind",     "prompt": "wind turbine, wind farm",                           "min_area_m2": 80},
            "tank":     {"name": "sam3_tank",     "prompt": "oil storage tank, fuel tank, tank farm",            "min_area_m2": 60},
            "airplane": {"name": "sam3_airplane", "prompt": "airplane, aircraft on ground",                      "min_area_m2": 80},
            "ship":     {"name": "sam3_ship",     "prompt": "ship, vessel, boat",                                "min_area_m2": 50},
            "car":      {"name": "sam3_car",      "prompt": "car, vehicle, truck",                               "min_area_m2": 8},
            # 新增：農田/農地/魚塭/水塘
            "farmland": {
                "name": "sam3_farmland",
                "prompt": "farmland, cropland, agricultural field, cultivated land, crop rows, paddy field",
                "min_area_m2": 300},
            "agricultural_land": {
                "name": "sam3_agri_land",
                "prompt": "agricultural land, orchards, greenhouses, farm fields",
                "min_area_m2": 200},
            "fishpond": {
                "name": "sam3_fishpond",
                "prompt": "fish pond, aquaculture pond, shrimp pond, rectangular ponds, aerators",
                "min_area_m2": 80},
            "water_pond": {
                "name": "sam3_water_pond",
                "prompt": "water pond, small reservoir, retention basin, farm pond",
                "min_area_m2": 60},
        }

        # Extra prompts → custom concepts
        if extra_prompts:
            parts = [p.strip() for p in extra_prompts.split(",") if p.strip()]
            for i, pr in enumerate(parts, start=1):
                key = f"custom_{i}"
                concepts[key] = {"name": f"sam3_custom_{i:02d}", "prompt": pr, "min_area_m2": 5}

        # 僅在投影座標(公尺)時計算 min_area_px；否則為 0
        for k, cfg in concepts.items():
            m2 = float(cfg.get("min_area_m2", 0) or 0)
            if is_projected_m and cellsize and cellsize > 0 and m2 > 0:
                cfg["min_area_px"] = int(max(0, round(m2 / (cellsize * cellsize))))
            else:
                cfg["min_area_px"] = 0

        # ---- Run folders ----
        run_root   = os.path.join(out_folder, f"sam3_v485_{_now_tag()}")  # 對齊 v4.8.9
        tiles_dir  = os.path.join(run_root, "tiles")
        geojson_dir= os.path.join(run_root, "tile_geojson")
        logs_dir   = os.path.join(run_root, "logs")
        os.makedirs(tiles_dir, exist_ok=True)
        os.makedirs(geojson_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)

        concepts_json = os.path.join(run_root, "concepts.json")
        _write_text(concepts_json, json.dumps(concepts, ensure_ascii=False, indent=2))
        messages.addMessage(f"Run folder: {run_root}")
        messages.addMessage(f"cellsize={cellsize}, tile_px={tile_px}, overlap_px={overlap_px}, imgsz={imgsz}, conf={conf}, device={device}, half={half}")

        # ---- Processing extent (AOI or raster) ----
        proc_extent = arcpy.Describe(aoi).extent if aoi else arcpy.Describe(in_raster).extent
        xmin, ymin, xmax, ymax = proc_extent.XMin, proc_extent.YMin, proc_extent.XMax, proc_extent.YMax
        tile_m = tile_px * cellsize
        step_m = max(cellsize, (tile_px - overlap_px) * cellsize)
        ncols = int(math.ceil((xmax - xmin) / step_m))
        nrows = int(math.ceil((ymax - ymin) / step_m))
        total_tiles = nrows * ncols
        messages.addMessage(f"Extent tiles: {nrows}x{ncols} = {total_tiles}")

        def export_tile(path_out: str, ext):
            arcpy.env.extent = ext
            arcpy.management.CopyRaster(in_raster, path_out, format="TIFF")
            arcpy.env.extent = None

        def export_mask(path_out: str, ext):
            arcpy.env.extent = ext
            arcpy.management.CopyRaster(water_mask, path_out, format="TIFF", pixel_type="8_BIT_UNSIGNED")
            arcpy.env.extent = None
            # 若需強制二值化（>0→1），可在安裝有 SA 的環境下開啟以下程式碼：
            try:
                arcpy.sa.Con(arcpy.Raster(path_out) > 0, 1, 0).save(path_out)
            except Exception:
                pass  # 無 SA 或失敗則略過，推論端可採用 >0 視為排除

        tile_geojson_paths: List[str] = []
        tile_index = 0
        t_all0 = time.time()

        for r in range(nrows):
            y0 = ymin + r * step_m
            y1 = min(y0 + tile_m, ymax)
            for c in range(ncols):
                x0 = xmin + c * step_m
                x1 = min(x0 + tile_m, xmax)
                ext = arcpy.Extent(x0, y0, x1, y1)

                tile_tif = os.path.join(tiles_dir, f"tile_{tile_index:06d}.tif")
                export_tile(tile_tif, ext)

                excl_tif = None
                if water_mask:
                    excl_tif = os.path.join(tiles_dir, f"water_{tile_index:06d}.tif")
                    export_mask(excl_tif, ext)

                out_geojson = os.path.join(geojson_dir, f"tile_{tile_index:06d}.geojson")
                log_path    = os.path.join(logs_dir,  f"tile_{tile_index:06d}.log")
                log_json    = os.path.join(logs_dir,  f"tile_{tile_index:06d}.json")

                cmd = [
                    ext_python, "-E", "-s", infer_py,
                    "--in_tif", tile_tif,
                    "--out_geojson", out_geojson,
                    "--concepts_json", concepts_json,
                    "--model", model,
                    "--bpe_path", bpe,
                    "--device", device,
                    "--imgsz", str(imgsz),
                    "--conf", str(conf),
                    "--tile_id", str(tile_index),
                    "--no_plot",
                    "--no_save",
                    "--log_json", log_json,
                ]
                if half:
                    cmd.append("--half")
                if excl_tif:
                    cmd += ["--exclude_mask_tif", excl_tif]

                if src_epsg not in (None, ""):
                    try:
                        epsg = int(src_epsg)
                        if epsg > 0:
                            cmd += ["--src_epsg", str(epsg)]
                    except Exception:
                        pass

                rc = self._run_external(cmd, messages, python_exe_for_env=ext_python, log_path=log_path)
                if rc == 0 and os.path.exists(out_geojson):
                    tile_geojson_paths.append(out_geojson)
                else:
                    messages.addWarningMessage(f"Tile {tile_index:06d} failed or missing GeoJSON. See log: {log_path}")
                tile_index += 1

        # ---- Merge tile GeoJSON ----
        merged_geojson = os.path.join(run_root, "merged.geojson")
        merged = {"type": "FeatureCollection", "name": "sam3_merged", "features": []}
        for gp in tile_geojson_paths:
            try:
                with open(gp, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                    feats = obj.get("features", [])
                    if feats:
                        merged["features"].extend(feats)
            except Exception:
                continue
        merged, dropped = _sanitize_geojson(merged)
        if dropped:
            messages.addWarningMessage(f"Sanitized merged GeoJSON: dropped {dropped} invalid feature(s) with non-finite coordinates.")
        _write_text(merged_geojson, json.dumps(merged, ensure_ascii=False, allow_nan=False))
        messages.addMessage(f"Merged GeoJSON: {merged_geojson} (features={len(merged['features'])})")
        if not merged["features"]:
            messages.addWarningMessage("No features found in merged GeoJSON. Check per-tile logs / AOI / prompts.")
            return  # 無要素則不再匯入 GDB

        # ---- Import to GDB and split by concept ----
        run_tag = out_suffix if out_suffix else _now_tag()
        temp_fc = os.path.join(out_gdb, f"{out_prefix}_all_{run_tag}")
        arcpy.conversion.JSONToFeatures(merged_geojson, temp_fc)

        # 依 'concept' 欄位拆分成多個 FC
        out_fcs: List[str] = []
        unique_concepts = sorted({row[0] for row in arcpy.da.SearchCursor(temp_fc, ["concept"]) if row and row[0]})
        for concept in unique_concepts:
            # sanitize name
            name = concept.replace("sam3_", "").replace("-", "_").replace(" ", "_")
            base_fc_name = f"{out_prefix}_{name}_{run_tag}"
            out_fc = _unique_fc_path(out_gdb, base_fc_name)
            where = "concept = '{}'".format(concept.replace("'", "''"))
            lyr_name = f"lyr_{name}"
            lyr = arcpy.management.MakeFeatureLayer(temp_fc, lyr_name, where)[0]
            arcpy.management.CopyFeatures(lyr, out_fc)
            out_fcs.append(out_fc)
            messages.addMessage(f"Output FC: {out_fc}")

        total_sec = time.time() - t_all0
        messages.addMessage(f"Done. Total seconds: {total_sec:.1f}. Run folder: {run_root}")