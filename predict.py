#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict.py - предсказание классов для всех фото в папке. Только CPU, без интернета.

  python predict.py --input путь/к/папке --output preds.csv

Результат - CSV с колонками filename,label,confidence:
  filename    путь файла относительно --input (для обычной папки - просто имя файла)
  label       Capsicum | Carrot | Cucumber | Potato | Raddish | Squash, либо unknown:
              файл не удалось прочитать, или уверенность ниже --unknown-threshold (если задан)
  confidence  вероятность выбранного класса от 0 до 1 (для нечитаемого файла 0)

Как выполнены требования ТЗ:
  * CPU: модель всегда считается на процессоре, даже если в машине есть видеокарта;
  * без интернета: архитектура создаётся без скачивания весов ImageNet, свои веса читаются из файла;
  * один битый или неподдерживаемый файл не останавливает работу: он попадает в CSV как unknown
    с confidence 0, а причина печатается в консоль;
  * предобработка та же, что при обучении (common.py): поворот по EXIF, RGB, resize 256, центр 224.
"""
import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import torch

from common import CLASSES, build_model, eval_transform, load_image

try:  # фото с iPhone в HEIC читаются, только если установлен pillow-heif (необязательно)
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff",
                    ".heic", ".heif"}
DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "weights" / "model.pt"


def parse_args():
    p = argparse.ArgumentParser(description="Классификация овощей на фото (CPU).")
    p.add_argument("--input", required=True, help="папка с фото (вложенные папки тоже обрабатываются)")
    p.add_argument("--output", required=True, help="куда записать CSV")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="файл весов (по умолчанию weights/model.pt)")
    p.add_argument("--batch", type=int, default=16, help="сколько фото считать за раз")
    p.add_argument("--threads", type=int, default=0, help="потоков CPU (0 = как решит PyTorch)")
    p.add_argument("--unknown-threshold", type=float, default=0.0,
                   help="если уверенность ниже, ответ unknown (0 = не использовать)")
    p.add_argument("--benchmark", type=int, default=0,
                   help="дополнительно замерить чистое время сети на одном фото, N повторов")
    return p.parse_args()


def list_files(root):
    images, skipped = [], []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith("."):
            images.append(p)
        else:
            skipped.append(p)       # .txt, .DS_Store, ._файлы macOS и т.п.
    return images, skipped


def load_model(weights_path):
    ck = torch.load(weights_path, map_location="cpu", weights_only=True)
    if list(ck["classes"]) != CLASSES:
        sys.exit(f"Классы в весах {ck['classes']} не совпадают с {CLASSES}")
    model = build_model(ck["arch"], pretrained=False)   # pretrained=False: ничего не скачиваем
    model.load_state_dict(ck["state_dict"])
    return model.eval(), ck["arch"]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
        sys.stderr.reconfigure(errors="replace")
    args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    in_dir = Path(args.input)
    if not in_dir.is_dir():
        sys.exit(f"Папка не найдена: {in_dir}")
    if not Path(args.weights).is_file():
        sys.exit(f"Нет файла весов: {args.weights} (см. README: веса лежат в weights/model.pt)")

    t_load = time.perf_counter()
    model, arch = load_model(args.weights)
    t_load = time.perf_counter() - t_load
    transform = eval_transform()
    files, skipped = list_files(in_dir)
    print(f"Модель {arch} загружена за {t_load:.1f} c; CPU-потоков: {torch.get_num_threads()}")
    print(f"Фото: {len(files)}" + (f"; пропущено файлов не-изображений: {len(skipped)}" if skipped else ""))

    results, bad = {}, 0
    batch_x, batch_names = [], []
    first_x = None

    def run_batch():
        with torch.inference_mode():
            probs = torch.softmax(model(torch.stack(batch_x)), dim=1)
        conf, idx = probs.max(dim=1)
        for name, c, i in zip(batch_names, conf.tolist(), idx.tolist()):
            results[name] = (CLASSES[i] if c >= args.unknown_threshold else "unknown", c)
        batch_x.clear()
        batch_names.clear()

    t0 = time.perf_counter()
    for path in files:
        name = path.relative_to(in_dir).as_posix()
        try:
            x = transform(load_image(path))
        except Exception as e:          # битый, обрезанный, не-картинка, неподдерживаемый формат
            results[name] = ("unknown", 0.0)
            bad += 1
            print(f"! не удалось прочитать {name}: {type(e).__name__}: {e}"[:220], file=sys.stderr)
            continue
        if first_x is None:
            first_x = x
        batch_x.append(x)
        batch_names.append(name)
        if len(batch_x) >= args.batch:
            run_batch()
    if batch_x:
        run_batch()
    elapsed = time.perf_counter() - t0

    out = Path(args.output)
    if out.parent != Path(""):
        out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label", "confidence"])
        for name in sorted(results):
            label, conf = results[name]
            w.writerow([name, label, f"{conf:.4f}"])

    n = len(files)
    print(f"Готово: {n} фото ({bad} нечитаемых) за {elapsed:.1f} c"
          + (f" = {1000 * elapsed / n:.1f} мс на фото, включая чтение и предобработку" if n else ""))
    print(f"CSV: {out}")

    if args.benchmark and first_x is not None:     # чистое время сети на одном фото
        x1 = first_x.unsqueeze(0)
        with torch.inference_mode():
            for _ in range(5):
                model(x1)
            times = []
            for _ in range(args.benchmark):
                t = time.perf_counter()
                model(x1)
                times.append(1000 * (time.perf_counter() - t))
        print(f"Сеть на 1 фото (batch=1, {args.benchmark} повторов): медиана {statistics.median(times):.1f} мс, "
              f"min {min(times):.1f} мс")


if __name__ == "__main__":
    main()
