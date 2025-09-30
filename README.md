

# Segmentación Multibanda — Pipeline completo (Datos → Tiles → Entrenamiento)

Este repositorio implementa un **pipeline de segmentación semántica** para datos de teledetección multibanda (25 canales), desde la **generación del stack** (Google Earth Engine), creación de **tiles** `.npy`, hasta el **entrenamiento** con manejo de **clases minoritarias** (oversampling, crops garantizados y pérdidas ponderadas).

---

## 🧭 Flujo general

1) **Stack multibanda (25 bandas) en GEE** → exportado como GeoTIFF.  
2) **Tiler C++ (GDAL + cnpy)**: corta la imagen y su máscara en tiles fijos y los guarda como **NumPy**:
   - Imagen: `(C, tileH, tileW)` float32.
   - Máscara: `(tileH, tileW)` uint8 (con remapeo de clases).
3) **Preprocesado** en Python: estadísticas de bandas y pesos por clase.
4) **Entrenamiento** (PyTorch + SMP): Unet++ (variantes *inflated* / *projection*), **oversampling agresivo**, **crops centrados** en minoritarias, **CutMix condicionado**, **pérdidas con pesos manuales** y métricas IoU.

---

## 📂 Estructura del repo (clave)

```
├── band_stats.npz                    # medias/std por banda (C=25)
├── class_weights.json                # pesos por clase (balanceo base)
├── compute_band_stats.py             # script para generar band_stats.npz
├── compute_class_weights.py          # script para generar class_weights.json
├── data_loading.py                   # Dataset (crops garantizados, CutMix, albumentations)
├── loss_functions.py                 # CE/Focal/Tversky/Lovasz + OHEM (opcional)
├── metrics.py                        # iou_per_class, mean_iou + métricas fastai
├── model_creation.py                 # build_model('inflated'|'projection'), Unet++
├── experiments_runner.py             # orquestador de experimentos
├── samplers.py                       # WeightedRandomSampler y MinorityOversampler
├── tile_raster.cpp                   # tiler GDAL + cnpy (C++)
├── results/                          # métricas y checkpoints
└── (opcional) scripts GEE            # exportación del stack (25 bandas)
```

---

## 🛰️ 1) Stack multibanda (GEE)

El script GEE compone el año **2020** para un AOI definido por códigos **WRS-2** y genera **25 bandas**:

- **Sentinel‑2 SR** (mediana): `B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12` (reflectancia 0–1).
- **Índices**: `NDVI, EVI, NDWI (Green–NIR), NBR`.
- **Terreno** (SRTM 30 m): `elev, slope_deg, aspect_deg, curvature, rugosity, TWI`.
- **Agua (JRC 2020)**: `water_perm, water_seas`.
- **LST (MODIS, °C)**: `LST_C` (promedio anual).
- **Humedad suelo (SMAP 0–1)**: `SMAP_SoilMoist` (promedio anual con QA).
- **Canopy height 2020 (10 m)**: `CanopyHeight_10m`.

**Salida**: GeoTIFF reproyectado (p. ej., `EPSG:6372`, 30 m). *(Recomendación: puedes omitir `.reproject()` en el pipeline y fijar `crs/scale` solo en el `Export`.)*

---

## 🧱 2) Tiler (C++ con GDAL + cnpy)

**`tile_raster.cpp`** lee `data.tif` (C bandas) y `mask.tif` (1 banda), corta una rejilla de `tile_w × tile_h` y guarda:

- `out_img_dir/{id}.npy`: **(C, tileH, tileW)** float32 (NaNs → 0).
- `out_msk_dir/{id}.npy`: **(tileH, tileW)** uint8 (remapeada).

### Remapeo de clases (ejemplo del código)

| Valor original | Clase remapeada |
|----------------|------------------|
| −5             | 0                |
| 2,3,6,12,14    | 1                |
| 29,290         | 2                |
| 28,280,21,23,25,26,27 | 3        |
| 32             | 4                |
| 30             | 5                |
| 31             | 6                |

> Los tiles con **máscara 0 en todo el tile** se **descartan** (reduce vacíos).

### Compilación
```bash
g++ -std=c++17 tile_raster.cpp -o tile_raster   $(gdal-config --cflags --libs) -lcnpy -lz
```

### Uso
```bash
./tile_raster data.tif mask.tif out_img_dir out_msk_dir 512 512
```

---

## 📊 3) Preprocesado (Python)

### Estadísticas por banda (para normalizar)
```bash
python compute_band_stats.py --img_dir /path/img_data --out_path band_stats.npz --sample 200
```

### Pesos por clase (balanceo base)
```bash
python compute_class_weights.py   --mask_dir /path/img_mask --num_classes 7   --out_json class_weights.json --scheme median_freq
```
- Esquemas: `median_freq`, `inv_log`, `inv`.  
- Los pesos se normalizan a media ≈ 1. Luego se **multiplican** por **boosts manuales** para minoritarias.

---

## 🏋️ 4) Entrenamiento (experiments_runner.py)

### Modelos
- **`inflated`**: Unet++ con primera conv **inflada** a 25 canales (mantiene *encoder_weights* con réplica de filtros RGB).
- **`projection`**: Conv **1×1** (25→3) + Unet++ preentrenado (aprovecha `encoder_weights='imagenet'`).

### Dataset y augmentations
- **Crops garantizados** hacia minoritarias con reintentos (`minority_center_prob` ~ 0.95, `min_minor_pixels`).
- **CutMix condicionado**: inserta parches desde tiles que contienen minoritarias.
- **Channel dropout** multibanda.
- **Albumentations**: flips/rotaciones, `Affine`, `GridDistortion`, ruido gaussiano (según versión) y **Normalize** (con `band_stats.npz`).

### Samplers
- `random`: muestreo uniforme (baseline).
- `weighted`: favorece tiles con minoritarias según `rare_score`.
- `minority_oversampler`: **agresivo** (duplica probabilidades y alarga época con `epoch_mult`).

### Pérdidas
- `ce`, `focal`, `tversky`, `cb_focal_tversky_lovasz` (combinación).  
- **Pesos**: `class_weights.json` × **boosts manuales** (por defecto `{4:6, 5:7, 6:8}`).  
- OHEM opcional vía `ohem_topk` en CE/Focal.

### Ejecutar
```bash
# Ejemplo A: sampler weighted
python experiments_runner.py   --img_dir /path/img_data --mask_dir /path/img_mask   --band_stats band_stats.npz --class_weights class_weights.json   --minority 4,5,6 --val_ratio 0.2   --epochs 80 --batch 12 --lr 1e-3   --sampler weighted

# Ejemplo B: oversampler agresivo
python experiments_runner.py   --img_dir /path/img_data --mask_dir /path/img_mask   --band_stats band_stats.npz --class_weights class_weights.json   --minority 4,5,6 --val_ratio 0.2   --epochs 80 --batch 12 --lr 1e-3   --sampler minority_oversampler

# Smoke test (rápido): una semilla y una variante
python experiments_runner.py ... --runs_quick
```

**Salida** en `results/`:
- `metrics_log.csv` y `metrics_log.xlsx` (hojas: `history`, `best_by_run`, `group_summary`).
- `ckpt/VARIANTE-...-best.pth` (mejor checkpoint por run).

---

## 🧪 Métricas

- `iou_per_class` (PyTorch puro), `mean_iou`, y métricas fastai (`MeanIoU`, `PerClassIoU`).  
- En `metrics_log.*`: columnas `epoch`, `mIoU`, `IoU_0..IoU_6`, `run_name`, etc.

---

## ⚙️ Requisitos (sugerencia)

- **Python ≥ 3.10**, CUDA si hay GPU.
- PyTorch (instala según tu CUDA): https://pytorch.org
- Paquetes:
  ```bash
  pip install segmentation-models-pytorch albumentations opencv-python               numpy pandas openpyxl fastai
  ```
- (Opcional) **Earth Engine** para generar el stack:
  ```bash
  pip install earthengine-api
  python -c "import ee; ee.Authenticate(); ee.Initialize()"
  ```
- (Para el tiler C++) **GDAL**, **cnpy**, **zlib** (instala vía tu gestor de paquetes y compila como arriba).

---

## 🛠️ Solución de problemas

- **Albumentations `Affine`:** evita `shear=None` y parámetros deprecados; ya está configurado en `data_loading.py`.
- **Multiprocessing + Lambda:** se reemplazó por un transform custom (`GaussianNoiseManual`).  
- **`ImportError: iou_per_class`**: usa `metrics.py` provisto (incluye la función PyTorch).  
- **`ImportError: build_model`**: usa `model_creation.py` con la función `build_model(...)`.  
- **`band_stats` vs canales**: asegúrate que la longitud de `mean/std` coincida con `in_channels` (25).  
- **Datasets vacíos**: el tiler descarta tiles todo‑fondo; si necesitas negativos, modifica el tiler para guardar 1/N tiles vacíos.

---

## 📜 Licencia

Este código se distribuye **tal cual**, sin garantías. Adapta la licencia según las políticas de tu organización.

---

## ✍️ Créditos rápidos

- **SMP** (segmentation_models_pytorch) para Unet++.  
- **Albumentations** para augmentations.  
- **GDAL + cnpy** para tiling `.npy`.  
- **Google Earth Engine** para componer el stack multibanda.