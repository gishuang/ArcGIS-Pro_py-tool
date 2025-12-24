# ArcGIS-Pro_py-tool
ArcGIS Pro python


# SAM-3 (ArcGIS + GeoJSON)

This repo provides scripts and ArcGIS Pro tools to extract **farmland parcels**, **fishponds**, and **water ponds**...... from aerial imagery using **Ultralytics SAM-3**.  
Two outputs are supported:

1. **Polygons** (Polygon/MultiPolygon): dissolved non-overlapping parcels; includes **`crs: EPSG:4326`** in GeoJSON for compatibility.

> Imagery resolution: **5 cm ~ 1 m**.  
> Minimum area threshold: **200 m²** (WGS84 geodesic area).

---
# Prerequisites

- **Miniconda** environment (Windows recommended for ArcGIS Pro).
- **GPU** with CUDA is recommended for speed; CPU is supported but slower.
- **Ultralytics SAM-3** weights:
  - `sam3.pt` (download per Ultralytics/Meta instructions)
  - Optional BPE vocab: `bpe_simple_vocab_16e6.txt.gz`

Place the files here (configurable in scripts):
C:\sam3_assets\sam3.pt
C:\sam3_assets\bpe_simple_vocab_16e6.txt.gz

c:\sam3_tools\sam3_infer_multiconcept_tile_geojson_v4_8_9.py

ArcGIS Pro


GeoJSON → Feature Class

Use GeoJSON to Feature Class tool.
For polygons, ensure target is Polygon; for boundaries, ensure target is Polyline.
Map coordinate system: GCS WGS 1984 (EPSG:4326).



Polygon to Line (optional)

If you started from polygons and need edges, use Polygon To Line in ArcGIS.



.pyt Tool

arcgis/sam3_extract_v4_8_9_manualname.pyt (to be provided) wraps these scripts into an ArcGIS Pro toolbox.
It will expose parameters (input folder, output path, tile size, overlap, area threshold).

## Repository Structure



License：MIT License Copyright (c) 2025
Permission is hereby granted, free of charge, to any person obtaining a copy

Acknowledgements：
https://docs.ultralytics.com/models/sam-3/
ESRI ArcGIS Pro workflows (GeoJSON to Feature)

