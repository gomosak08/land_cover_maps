#include "gdal_priv.h"
#include "cpl_conv.h"
#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>
#include <filesystem>
#include <cmath>
#include "cnpy.h"

using std::string;

static const std::unordered_map<int,int> VALUE_MAP = {
    {-5,0}, {2,1}, {3,1}, {6,1}, {12,1}, {28,3}, {29,2},
    {30,5}, {31,6}, {32,4}, {280,3}, {14,1}, {21,3},
    {23,3}, {25,3}, {26,3}, {27,3}, {290,2}
};

inline uint8_t remap(int v){
    auto it = VALUE_MAP.find(v);
    return (it==VALUE_MAP.end()) ? 0 : static_cast<uint8_t>(it->second);
}

int main(int argc, char** argv){
    if(argc < 7){
        std::cerr << "Usage: " << argv[0]
                  << " data.tif mask.tif out_img_dir out_msk_dir tile_w tile_h\n";
        return 1;
    }
    string dataPath = argv[1];
    string maskPath = argv[2];
    string outImgDir = argv[3];
    string outMskDir = argv[4];
    const int tileW = std::stoi(argv[5]);
    const int tileH = std::stoi(argv[6]);

    std::filesystem::create_directories(outImgDir);
    std::filesystem::create_directories(outMskDir);

    GDALAllRegister();

    GDALDataset* imgDs = static_cast<GDALDataset*>(GDALOpen(dataPath.c_str(), GA_ReadOnly));
    GDALDataset* mskDs = static_cast<GDALDataset*>(GDALOpen(maskPath.c_str(), GA_ReadOnly));
    if(!imgDs || !mskDs){
        std::cerr << "Cannot open input datasets\n";
        if(imgDs) GDALClose(imgDs);
        if(mskDs) GDALClose(mskDs);
        return 2;
    }

    const int Wimg = imgDs->GetRasterXSize();
    const int Himg = imgDs->GetRasterYSize();
    const int C    = imgDs->GetRasterCount();

    const int Wmsk = mskDs->GetRasterXSize();
    const int Hmsk = mskDs->GetRasterYSize();

    // Región común (evita fallos si no coinciden tamaños globales)
    const int W = std::min(Wimg, Wmsk);
    const int H = std::min(Himg, Hmsk);
    if(W <= 0 || H <= 0){
        std::cerr << "No common overlap between image and mask.\n";
        GDALClose(imgDs); GDALClose(mskDs);
        return 3;
    }
    if(Wimg != Wmsk || Himg != Hmsk){
        std::cerr << "[WARN] Image and mask sizes differ. Using common overlap: "
                  << "W=" << W << ", H=" << H << "\n";
    }

    // Buffers por tile (PAD con ceros para siempre guardar tileW x tileH)
    std::vector<float>   imgTile(C * tileW * tileH, 0.0f);   // (C, tileH, tileW)
    std::vector<uint8_t> mskTile(tileW * tileH, 0);          // (tileH, tileW)

    // Buffers temporales para la porción válida (w x h)
    std::vector<float> imgTmp;  // se redimensiona por banda
    std::vector<int>   mskTmp;  // lectura cruda de máscara antes de remap

    const int cols = (W + tileW - 1) / tileW;
    const int rows = (H + tileH - 1) / tileH;

    int tileId = 0, saved = 0;

    for(int r=0; r<rows; ++r){
        for(int cidx=0; cidx<cols; ++cidx){
            const int x0 = cidx * tileW;
            const int y0 = r * tileH;
            const int w = std::min(tileW, W - x0);
            const int h = std::min(tileH, H - y0);
            if(w<=0 || h<=0) { ++tileId; continue; }

            // Limpia/pone a cero los buffers padded
            std::fill(imgTile.begin(), imgTile.end(), 0.0f);
            std::fill(mskTile.begin(), mskTile.end(), uint8_t(0));

            // --- Leer máscara (banda 1) SOLO en la región válida (w,h)
            mskTmp.assign(w*h, 0);
            CPLErr err = mskDs->GetRasterBand(1)->RasterIO(
                GF_Read, x0, y0, w, h,
                mskTmp.data(), w, h, GDT_Int32,
                0, 0, nullptr
            );
            if(err != CE_None){
                std::cerr << "Mask read error at tile " << tileId << "\n";
                ++tileId; continue;
            }

            // Remap y copia dentro del tile padded (top-left)
            bool allZero = true;
            for(int yy=0; yy<h; ++yy){
                for(int xx=0; xx<w; ++xx){
                    const int srcIdx = yy*w + xx;
                    const int dstIdx = yy*tileW + xx;
                    uint8_t v = remap(mskTmp[srcIdx]);
                    mskTile[dstIdx] = v;
                    if(v != 0) allZero = false;
                }
            }
            if(allZero){ ++tileId; continue; }  // descarta tiles vacíos

            // --- Leer imagen banda por banda y copiar con padding
            for(int b=1; b<=C; ++b){
                imgTmp.assign(w*h, 0.0f);
                err = imgDs->GetRasterBand(b)->RasterIO(
                    GF_Read, x0, y0, w, h,
                    imgTmp.data(), w, h, GDT_Float32,
                    0, 0, nullptr
                );
                if(err != CE_None){
                    std::cerr << "Image read error (band " << b << ") at tile " << tileId << "\n";
                    break;
                }

                // dst apuntando al inicio de la banda b-1 en el tile padded
                float* dstBand = imgTile.data() + (b-1)*tileW*tileH;

                // Copiar con NaN->0
                for(int yy=0; yy<h; ++yy){
                    for(int xx=0; xx<w; ++xx){
                        const int srcIdx = yy*w + xx;
                        const int dstIdx = yy*tileW + xx;
                        float v = imgTmp[srcIdx];
                        if(std::isnan(v)) v = 0.0f;
                        dstBand[dstIdx] = v;
                    }
                }
            }

            // --- Guardar .npy (sin pickle)
            {
                // Imagen: shape (C, tileH, tileW), dtype float32
                std::vector<size_t> shape_img = { (size_t)C, (size_t)tileH, (size_t)tileW };
                const std::string imgNpy = outImgDir + "/" + std::to_string(tileId) + ".npy";
                cnpy::npy_save(imgNpy, imgTile.data(), shape_img, "w");

                // Máscara: shape (tileH, tileW), dtype uint8
                std::vector<size_t> shape_msk = { (size_t)tileH, (size_t)tileW };
                const std::string mskNpy = outMskDir + "/" + std::to_string(tileId) + ".npy";
                cnpy::npy_save(mskNpy, mskTile.data(), shape_msk, "w");
            }

            ++saved;
            ++tileId;
        }
    }

    std::cout << "Done. tiles_total=" << tileId << " tiles_saved=" << saved << "\n";
    GDALClose(imgDs); GDALClose(mskDs);
    return 0;
}

/*
Ejemplo:
./tile_raster \
  /home/gomosak/conafor_archivo/segmentacion/cnn/data/mosaic_clip.tif \
  /home/gomosak/conafor_archivo/segmentacion/cnn/data/mask.tif \
  /home/gomosak/conafor_archivo/segmentacion/cnn/img_data \
  /home/gomosak/conafor_archivo/segmentacion/cnn/img_mask \
  512 512

*/
