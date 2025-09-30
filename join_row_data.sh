#!/usr/bin/env bash
set -euo pipefail

# === Paths ===
RASTER_DIR="/home/gomosak/conafor_archivo/segmentacion/data_row"
CUTLINE_GPKG="/home/gomosak/conafor_archivo/segmentacion/entrenamiento_recortar.gpkg"
CUTLINE_LAYER="$(ogrinfo -q "$CUTLINE_GPKG" | head -n 1 | awk '{print $2}')"  # detecta nombre de capa
OUT_DIR="/home/gomosak/conafor_archivo/segmentacion/cnn/data"

# === Outputs ===
LIST_FILE="${OUT_DIR}/tif_list.txt"
VRT_OUT="${OUT_DIR}/mosaic.vrt"
CLIP_OUT="${OUT_DIR}/mosaic_clip_tmp.tif"
FINAL_OUT="${OUT_DIR}/mosaic_clip.tif"

mkdir -p "$OUT_DIR"

echo ">> Listing GeoTIFF files..."
find "$RASTER_DIR" -type f \( -iname "*.tif" -o -iname "*.tiff" \) -print0 | sort -z | xargs -0 -I{} echo "{}" > "$LIST_FILE"
[[ -s "$LIST_FILE" ]] || { echo "No .tif found in $RASTER_DIR"; exit 1; }

echo ">> Building VRT (virtual mosaic)..."
gdalbuildvrt -input_file_list "$LIST_FILE" "$VRT_OUT"

echo ">> Clipping with polygon from GPKG..."
gdalwarp \
  -cutline "$CUTLINE_GPKG" \
  -cl "$CUTLINE_LAYER" \
  -crop_to_cutline \
  -dstnodata 0 \
  -multi -wo NUM_THREADS=ALL_CPUS -wm 2048 \
  -of GTiff \
  -co COMPRESS=DEFLATE -co PREDICTOR=2 -co TILED=YES -co BIGTIFF=YES \
  "$VRT_OUT" "$CLIP_OUT"

echo ">> Optimizing final GeoTIFF..."
gdal_translate "$CLIP_OUT" "$FINAL_OUT" \
  -co COMPRESS=DEFLATE -co PREDICTOR=2 -co TILED=YES -co BIGTIFF=YES

echo ">> Building overviews..."
gdaladdo -r average "$FINAL_OUT" 2 4 8 16 32

echo "✅ Done!"
echo "VRT:   $VRT_OUT"
echo "CLIP:  $FINAL_OUT"
