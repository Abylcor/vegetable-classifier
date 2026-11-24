"""Общее для обучения и предсказания: классы, загрузка картинки, преобразования, модель."""
from pathlib import Path

import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms as T

CLASSES = ["Capsicum", "Carrot", "Cucumber", "Potato", "Raddish", "Squash"]  # порядок = номер выхода сети
IMG_SIZE = 224
RESIZE = 256
MEAN = (0.485, 0.456, 0.406)  # нормализация ImageNet
STD = (0.229, 0.224, 0.225)
ARCHS = ("efficientnet_b0", "resnet18", "mobilenet_v3_large")
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_image(path):
    """RGB-картинка с поворотом по EXIF."""
    img = Image.open(path)
    if img.format == "JPEG":
        img.draft("RGB", (2 * RESIZE, 2 * RESIZE))
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return img.convert("RGB")


def eval_transform():
    return T.Compose([
        T.Resize(RESIZE),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


def train_transform(kind="full"):
    if kind == "light":
        return T.Compose([
            T.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])
    return T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(0.35, 1.0), ratio=(0.75, 1.33)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.03)], p=0.8),
        T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.2),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
        T.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])


def build_model(arch, num_classes=len(CLASSES), pretrained=True):
    """pretrained=True скачивает веса ImageNet (нужен интернет)."""
    if arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    elif arch == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"Неизвестная архитектура {arch}; доступны: {ARCHS}")
    return m


def head_module(model, arch):
    return model.fc if arch == "resnet18" else model.classifier


def gradcam_layer(model, arch):
    return model.layer4[-1] if arch == "resnet18" else model.features[-1]


def build_file_index(root):
    """'Класс/имя_файла' -> путь на диске."""
    index = {}
    for p in Path(root).rglob("*"):
        if p.suffix.lower() in IMG_EXTENSIONS and p.parent.name in CLASSES:
            index.setdefault(f"{p.parent.name}/{p.name}", p)
    return index
