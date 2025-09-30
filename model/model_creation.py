import segmentation_models_pytorch as smp
from fastai.vision.all import *

def create_model(encoder_name,in_channels,classes):
    """
    Creates a U-Net model using the segmentation models library.

    Parameters:
    - `encoder_name` (str): The name of the encoder to use for the U-Net model.
    - `in_channels` (int): The number of input channels for the model.
    - `classes` (int): The number of output classes for the model.

    Returns:
    - `torch.nn.Module`: The created U-Net model moved to the GPU.
    """
    return smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=None,   # important for 25 bands
        in_channels=in_channels,
        classes=classes,
        activation=None
    )
    
def create_learner(model,loss_func,opt_func,db,metrics):
    """
    Creates a fastai Learner object for training.

    Parameters:
    - `model` (torch.nn.Module): The model to be trained.
    - `loss_func` (callable): The loss function to use during training.
    - `opt_func` (callable): The optimizer function to use during training.
    - `db` (DataLoaders): The dataloaders for training and validation data.
    - `metrics` (list of callables): The list of metrics to evaluate during training.

    Returns:
    - `Learner`: The fastai Learner object.
    """
    return Learner(db, model, loss_func=loss_func, opt_func=opt_func, metrics=metrics)


def modify_first_conv_for_inchannels(encoder: nn.Module, in_channels: int) -> nn.Module:
    """
    Replace the first Conv2d of common torchvision backbones so it accepts `in_channels` instead of 3.

    Supported examples:
      - ResNet family: encoder.conv1
      - MobileNetV2:   encoder.features[0][0]
      - EfficientNetV2 (timm-like): encoder.conv_stem
      - ConvNeXt variants (stem inside features[0])

    If your backbone differs, adapt the attribute path accordingly.
    """
    # 1) ResNet family
    if hasattr(encoder, "conv1") and isinstance(encoder.conv1, nn.Conv2d):
        old = encoder.conv1
        encoder.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=(old.bias is not None),
        )
        nn.init.kaiming_normal_(encoder.conv1.weight, mode="fan_out", nonlinearity="relu")
        return encoder

    # 2) MobileNetV2
    try:
        first = encoder.features[0][0]
        if isinstance(first, nn.Conv2d):
            old = first
            encoder.features[0][0] = nn.Conv2d(
                in_channels=in_channels,
                out_channels=old.out_channels,
                kernel_size=old.kernel_size,
                stride=old.stride,
                padding=old.padding,
                bias=(old.bias is not None),
            )
            nn.init.kaiming_normal_(encoder.features[0][0].weight, mode="fan_out", nonlinearity="relu")
            return encoder
    except Exception:
        pass

    # 3) EfficientNetV2 (timm-like) stem
    if hasattr(encoder, "conv_stem") and isinstance(encoder.conv_stem, nn.Conv2d):
        old = encoder.conv_stem
        encoder.conv_stem = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=(old.bias is not None),
        )
        nn.init.kaiming_normal_(encoder.conv_stem.weight, mode="fan_out", nonlinearity="relu")
        return encoder

    # 4) ConvNeXt-like stems (adjust if your variant differs)
    try:
        # Many convnext implementations have the stem under features[0]
        for name, module in encoder.features[0]._modules.items():
            if isinstance(module, nn.Conv2d):
                old = module
                new_conv = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=old.out_channels,
                    kernel_size=old.kernel_size,
                    stride=old.stride,
                    padding=old.padding,
                    bias=(old.bias is not None),
                )
                nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
                encoder.features[0]._modules[name] = new_conv
                return encoder
    except Exception:
        pass

    raise ValueError("Could not locate the first Conv2d in the given encoder; tell me your backbone and I’ll adapt it.")