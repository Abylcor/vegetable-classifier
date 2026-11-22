# -*- coding: utf-8 -*-
"""
common.py - всё, что должно быть одинаковым при обучении и при предсказании:
список классов, загрузка картинки, преобразования, сборка модели.

Если предобработка в train.py и predict.py хоть чем-то отличается, модель на
предсказании видит "не те" картинки, поэтому она живёт в одном месте.
"""
from pathlib import Path

import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms as T

CLASSES = ["Capsicum", "Carrot", "Cucumber", "Potato", "Raddish", "Squash"]  # порядок = номер выхода сети
IMG_SIZE = 224            # вход сети
RESIZE = 256              # сначала короткая сторона -> 256, потом центр 224x224 (стандарт ImageNet)
MEAN = (0.485, 0.456, 0.406)  # нормализация, с которой обучали предобученные веса ImageNet
STD = (0.229, 0.224, 0.225)
ARCHS = ("efficientnet_b0", "resnet18", "mobilenet_v3_large")
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_image(path):
    """Открывает файл и возвращает RGB-картинку в правильной ориентации.

    * draft: большие JPEG (фото 9-16 Мп) декодируются сразу в уменьшенном виде, в разы быстрее;
    * exif_transpose: у 175 из 185 фото датасета поворот записан в EXIF, без этого они
      попадут в сеть боком (стандартный загрузчик torchvision EXIF не учитывает);
    * RGBA/палитра/серые/CMYK -> RGB, прозрачность кладём на белый фон.
    """
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
    """Для проверки и предсказания: без случайности."""
    return T.Compose([
        T.Resize(RESIZE),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


def train_transform(kind="full"):
    """Аугментации против того, чем фото из магазина отличаются от наших кадров.

    full:
      RandomResizedCrop  - другой масштаб и кадрирование (в магазине овощ может быть мелким);
      flips              - съёмка сверху, у овоща на столе нет "верха";
      ColorJitter        - другой свет и баланс белого; оттенок (hue) почти не трогаем,
                           цвет - главный признак класса (морковь оранжевая, огурец зелёный);
      GaussianBlur       - кадры видео смазаны, фото резкие: учим не зависеть от резкости;
      RandomErasing      - перекрытия: ценники, руки, упаковка.
    light: только кадрирование и отражение - для сравнения в отчёте.
    """
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
    """Предобученная на ImageNet сеть с новой "головой" на 6 классов.
    pretrained=True скачивает веса ImageNet (нужен интернет, только при обучении).
    В predict.py будет pretrained=False + загрузка своих весов из файла."""
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
    """Новая часть сети (классификатор), которую учим с нуля."""
    return model.fc if arch == "resnet18" else model.classifier


def gradcam_layer(model, arch):
    """Последний свёрточный слой - для Grad-CAM (карта 7x7 при входе 224)."""
    return model.layer4[-1] if arch == "resnet18" else model.features[-1]


def build_file_index(root):
    """'Класс/имя_файла' -> путь. Нужен, если на Kaggle папки вложены иначе, чем у вас локально.
    Копии в Dataset/ и final_dataset/ побайтово одинаковые, поэтому годится любая."""
    index = {}
    for p in Path(root).rglob("*"):
        if p.suffix.lower() in IMG_EXTENSIONS and p.parent.name in CLASSES:
            index.setdefault(f"{p.parent.name}/{p.name}", p)
    return index
