# model_creation.py
from __future__ import annotations
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

# ------------------------------------------------------
# Variantes de modelo para multibanda (25 canales)
# ------------------------------------------------------

class ProjectionUNetPP(nn.Module):
    """
    Proyección 1x1: in_channels -> 3, luego Unet++ con encoder preentrenado.
    Ventaja: aprovecha encoder_weights='imagenet' aunque tus entradas tengan 25 canales.
    """
    def __init__(self, in_channels: int, num_classes: int,
                 encoder_name: str = "resnet34", encoder_weights: str = "imagenet"):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
            activation=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(self.proj(x))


class InflatedUNetPP(nn.Module):
    """
    Inflar primera conv del encoder a in_channels.
    - Si encoder_weights='imagenet', infla conv1 replicando la media de sus filtros a los 25 canales.
    - Si encoder_weights=None, simplemente construye in_channels directamente.
    """
    def __init__(self, in_channels: int, num_classes: int,
                 encoder_name: str = "resnet34", encoder_weights: str = "imagenet"):
        super().__init__()

        if encoder_weights is None:
            # Caso simple: el encoder ya nace con in_channels = 25 (sin preentrenar)
            self.model = smp.UnetPlusPlus(
                encoder_name=encoder_name,
                encoder_weights=None,
                in_channels=in_channels,
                classes=num_classes,
                activation=None,
            )
        else:
            # Crear con 3 canales y luego inflar la primera conv del encoder
            self.model = smp.UnetPlusPlus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=3,
                classes=num_classes,
                activation=None,
            )
            self._inflate_first_conv_to_n_channels(in_channels)

    def _inflate_first_conv_to_n_channels(self, new_in_channels: int):
        """
        Localiza la primera conv del encoder y la reemplaza por una nueva con new_in_channels.
        Inicializa pesos repitiendo la media de los 3 canales preentrenados.
        """
        enc = self.model.encoder
        conv = None

        # rutas comunes según backbone
        for attr in ["conv1", "stem.conv1", "conv_stem"]:
            try:
                # soporte para 'stem.conv1'
                target = enc
                for part in attr.split("."):
                    target = getattr(target, part)
                if isinstance(target, nn.Conv2d):
                    conv = (attr, target)
                    break
            except Exception:
                continue

        # ConvNeXt / otros: buscar la primera nn.Conv2d con in_channels==3
        if conv is None:
            try:
                for name, module in enc.named_modules():
                    if isinstance(module, nn.Conv2d) and module.in_channels == 3:
                        conv = (name, module)
                        break
            except Exception:
                pass

        if conv is None:
            raise ValueError("No se encontró la primera conv del encoder para inflar.")

        name_path, old = conv
        W = old.weight.data  # [out_c, 3, kH, kW]
        new = nn.Conv2d(
            in_channels=new_in_channels,
            out_channels=old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=(old.bias is not None),
        )

        with torch.no_grad():
            # Promedia sobre el eje de entrada (canales RGB) y repite a new_in_channels
            meanW = W.mean(dim=1, keepdim=True)  # [out_c, 1, kH, kW]
            new.weight[:] = meanW.repeat(1, new_in_channels, 1, 1)
            if old.bias is not None and new.bias is not None:
                new.bias[:] = old.bias

        # Colocar la nueva conv en el encoder, respetando la ruta (p.ej. 'stem.conv1')
        target_parent = enc
        parts = name_path.split(".")
        for p in parts[:-1]:
            target_parent = getattr(target_parent, p)
        setattr(target_parent, parts[-1], new)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

# ------------------------------------------------------
# API principal usada por experiments_runner.py
# ------------------------------------------------------

def build_model(variant: str, in_channels: int, num_classes: int,
                encoder_name: str = "resnet34", encoder_weights: str = "imagenet") -> nn.Module:
    """
    Crea el modelo según la variante:
      - 'projection': conv1x1 (in_channels->3) + Unet++ (preentrenado).
      - 'inflated'  : Unet++ y conv inicial inflada a in_channels (si hay preentrenamiento).
    """
    variant = variant.lower()
    if variant == "projection":
        return ProjectionUNetPP(in_channels, num_classes, encoder_name=encoder_name, encoder_weights=encoder_weights)
    elif variant == "inflated":
        return InflatedUNetPP(in_channels, num_classes, encoder_name=encoder_name, encoder_weights=encoder_weights)
    else:
        raise ValueError(f"Variante desconocida: {variant}")

# ------------------------------------------------------
# Compatibilidad opcional con fastai (si lo necesitas)
# ------------------------------------------------------

def create_model(encoder_name: str, in_channels: int, classes: int):
    """
    Compatibilidad con tu código previo: crea Unet++ directo sin variantes.
    Útil si sigues usando fastai Learner en otro flujo.
    """
    return smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=None,  # importante si pasas in_channels != 3 sin inflar
        in_channels=in_channels,
        classes=classes,
        activation=None
    )

def create_learner(model, loss_func, opt_func, db, metrics):
    # Solo si aún usas fastai en otro script.
    try:
        from fastai.vision.all import Learner
        return Learner(db, model, loss_func=loss_func, opt_func=opt_func, metrics=metrics)
    except Exception as e:
        raise RuntimeError("Fastai no está instalado o no disponible. "
                           "Usa tu loop PyTorch (experiments_runner.py).") from e
