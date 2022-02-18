import kornia.augmentation as kaug
from torchvision import transforms as tv_transforms


class GetItemTransform:
    def __call__(self, x):
        return x[0, :, :, :]


_IMAGE_NORMALIZATIONS = dict(
    imagenet=tv_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
)


def get_backbone_normalization_transforms(backbone_arch):
    if backbone_arch.startswith("resnet"):
        return _IMAGE_NORMALIZATIONS["imagenet"]
    else:
        raise NotImplementedError(backbone_arch)


def get_train_augmentation_transforms(
    p_flip_vertical=0.5,
    p_flip_horizontal=0.5,
    max_rotation=10.0,
    max_zoom=1.1,
    max_warp=0.2,
    p_affine=0.75,
    max_lighting=0.2,
    p_lighting=0.75,
):
    """
    Build a set of pytorch image Transforms to use during training:

        p_flip_vertical: probability of a vertical flip
        p_flip_horizontal: probability of a horizontal flip
        max_rotation: maximum rotation angle in degrees
        max_zoom: maximum zoom level
        max_warp: perspective warping scale (from 0.0 to 1.0)
        p_affine: probility of rotation, zoom and perspective warping
        max_lighting: maximum scaling of brightness and contrast
    """
    return [
        kaug.RandomVerticalFlip(p=p_flip_vertical),
        kaug.RandomHorizontalFlip(p=p_flip_horizontal),
        kaug.RandomAffine(p=p_affine, degrees=max_rotation, scale=(1.0, max_zoom)),
        kaug.RandomPerspective(p=p_affine, distortion_scale=max_warp),
        kaug.ColorJitter(p=p_lighting, brightness=max_lighting, contrast=max_lighting),
        # TODO: the kornia transforms work on batches and so by default they
        # add a batch dimension. Until I work out how to apply transform by
        # batches (and only while training) I will just keep this here to
        # remove the batch dimension again
        GetItemTransform(),
    ]


def get_transforms(step, normalize_for_arch):
    arch_norm_transform = get_backbone_normalization_transforms(
        backbone_arch=normalize_for_arch
    )

    transforms = [
        arch_norm_transform,
    ]

    if step == "train":
        transforms += get_train_augmentation_transforms()
    elif step == "predict":
        # we don't include any extra transforms when doing inference
        pass
    else:
        raise NotImplementedError(step)

    return tv_transforms.Compose(transforms)
