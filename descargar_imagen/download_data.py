import ee, time

# --- Auth & init (si ya autenticó antes, basta Initialize) ---
try:
    ee.Initialize(project='ee-kgomez0800-images')
except Exception:
    ee.Authenticate()
    ee.Initialize(project='ee-kgomez0800-images')

# -----------------------
# 0) Parámetros
# -----------------------
year = 2020
start = f'{year}-01-01'
end   = f'{year}-12-31'
out_crs = 'EPSG:6372'   # México LCC, cambia a 'EPSG:4326' si prefieres
out_scale = 30          # m (SRTM a 30m; puedes 10m si quieres alinear con S2)
export_desc = f"mexico_stack_{year}_WRSPR_033041_034041"
drive_folder = "gee_exports"

# -----------------------
# 1) AOI: dos frames por código WRSPR
# -----------------------
codes = ["032039","032038","032040","033038","033039","033040","033041","034039"]  # strings con ceros
wrs = ee.FeatureCollection('users/pablo_avila/Conafor/WRS2_descending_Mex')
sel = wrs.filter(ee.Filter.inList('WRSPR', codes))
aoi = sel.geometry()

# -----------------------
# 2) Sentinel-2 SR (bandas y nubes)
# -----------------------
s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
      .filterDate(start, end)
      .filterBounds(aoi)
      .map(lambda img:
           img.updateMask(img.select('QA60').bitwiseAnd(1<<10).eq(0))  # sin nubes
              .updateMask(img.select('QA60').bitwiseAnd(1<<11).eq(0))) # sin cirros
     )

def scale_s2(img):
    scaled = img.select(['B.*']).multiply(1e-4)  # reflectancia 0–1
    return img.addBands(scaled, overwrite=True)

s2 = s2.map(scale_s2)

bands_keep = ["B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12"]
s2_median = s2.median().select(bands_keep)

def add_indices(img):
    b = {bn: img.select(bn) for bn in bands_keep}
    ndvi = b['B8'].subtract(b['B4']).divide(b['B8'].add(b['B4'])).rename('NDVI')
    evi = (b['B8'].subtract(b['B4']).multiply(2.5)
           .divide(b['B8'].add(b['B4'].multiply(6)).subtract(b['B2'].multiply(7.5)).add(1.0))
          ).rename('EVI')
    ndwi = b['B3'].subtract(b['B8']).divide(b['B3'].add(b['B8'])).rename('NDWI')  # Green–NIR
    nbr  = b['B8'].subtract(b['B12']).divide(b['B8'].add(b['B12'])).rename('NBR')
    return img.addBands([ndvi, evi, ndwi, nbr])

s2_prod = add_indices(s2_median)

# -----------------------
# 3) DEM y derivados
# -----------------------
dem = ee.Image('USGS/SRTMGL1_003').clip(aoi)
terrain = ee.Algorithms.Terrain(dem)
slope = terrain.select('slope').rename('slope_deg')
aspect = terrain.select('aspect').rename('aspect_deg')

# Curvatura (Laplaciano simple)
kernel = ee.Kernel.fixed(3,3, [[0,1,0],[1,-4,1],[0,1,0]], -1, -1, False)
curvature = dem.convolve(kernel).rename('curvature')

# Rugosidad: desviación estándar local (3x3)
rugosity = dem.reduceNeighborhood(
    reducer=ee.Reducer.stdDev(),
    kernel=ee.Kernel.square(radius=1)
).rename('rugosity')

# TWI ≈ ln(As / tan(beta)), usando upa de MERIT Hydro
merit = ee.Image('MERIT/Hydro/v1_0_1').clip(aoi)
cellsize = ee.Number(out_scale)  # m
As = merit.select('upa').multiply(1e6).divide(cellsize)  # km2 -> m2 / m
tan_beta = slope.expression('tan(b)', {'b': slope.multiply(3.14159265/180)}).rename('tan_slope')
twi = As.divide(tan_beta.max(1e-6)).log().rename('TWI')

# -----------------------
# 4) Agua 2020 (JRC)
# -----------------------
jrc = (ee.ImageCollection('JRC/GSW1_4/YearlyHistory')
       .filter(ee.Filter.eq('year', year))
       .first()
       .select('waterClass')
       .clip(aoi))
water_perm = jrc.eq(3).rename('water_perm')
water_seas = jrc.eq(2).rename('water_seas')

# -----------------------
# 5) LST MODIS (°C) 2020 promedio
# -----------------------
lst = (ee.ImageCollection('MODIS/061/MOD11A2')
       .filterDate(start, end)
       .filterBounds(aoi)
       .select('LST_Day_1km')
       .mean()
       .multiply(0.02).subtract(273.15)  # K->°C
       .rename('LST_C')
       .clip(aoi)
       .resample('bilinear'))

# -----------------------
# 6) SMAP (vol. fracc. 0–1) 2020 promedio, con QA
# -----------------------
smap_col = (ee.ImageCollection('NASA/SMAP/SPL3SMP_E/005')
            .filterDate(start, end)
            .filterBounds(aoi))

def smap_daily_mean(img):
    sm = img.select(['soil_moisture_am','soil_moisture_pm'])
    qa_ok = img.select('retrieval_qual_flag_am').eq(0).And(
            img.select('retrieval_qual_flag_pm').eq(0))
    sm = sm.updateMask(qa_ok)
    sm_mean = sm.reduce(ee.Reducer.mean()).rename('SMAP_SoilMoist')
    return sm_mean.updateMask(sm_mean.gte(0).And(sm_mean.lte(1)))

smap = (smap_col.map(smap_daily_mean)
        .mean()
        .clip(aoi)
        .resample('bilinear'))

# -----------------------
# 7) Altura de dosel 2020 (10 m)
# -----------------------
canopy = (ee.Image('users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1')
          .select(0).rename('CanopyHeight_10m')
          .clip(aoi))

# -----------------------
# 8) Reproyección, remuestreo y STACK
# -----------------------
def prep(img):
    return img.reproject(crs=out_crs, scale=out_scale).clip(aoi)

# Agua: usa directamente las máscaras (default resampling = nearest)
water_perm_f = water_perm.toFloat()
water_seas_f = water_seas.toFloat()

stack = ee.Image.cat(
    prep(s2_prod.toFloat()),                 # S2 + índices
    prep(dem.rename('elev').toFloat()),
    prep(slope.toFloat()), prep(aspect.toFloat()),
    prep(curvature.toFloat()), prep(rugosity.toFloat()), prep(twi.toFloat()),
    prep(water_perm_f), prep(water_seas_f),
    prep(lst.toFloat()), prep(smap.toFloat()),
    prep(canopy.toFloat())
)

final_band_names = ["B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12",
                    "NDVI","EVI","NDWI","NBR",
                    "elev","slope_deg","aspect_deg",
                    "curvature","rugosity","TWI",
                    "water_perm","water_seas",
                    "LST_C","SMAP_SoilMoist",
                    "CanopyHeight_10m"]
stack = stack.rename(final_band_names)

# Por si alguna operación generó float64, fuerza a float32:
stack = stack.toFloat()

# -----------------------
# 9) Export a Google Drive
# -----------------------
task = ee.batch.Export.image.toDrive(
    image=stack,
    description=export_desc,
    folder=drive_folder,
    fileNamePrefix=export_desc,
    region=aoi,           # geometría de los dos frames
    scale=out_scale,
    crs=out_crs,
    maxPixels=1e13
)
task.start()
print("Export iniciado:", export_desc)

# (Opcional) Monitorear el estado:
while task.active():
    print('Procesando...')
    time.sleep(30)
print('Estado final:', task.status())
