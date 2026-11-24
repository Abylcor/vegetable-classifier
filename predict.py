"""Предсказание классов для всех фото в папке: только CPU, без интернета.

python predict.py --input папка --output preds.csv [--onnx] [--unknown-threshold 0.45]
CSV: filename, label (один из 6 классов или unknown), confidence.
"""
import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import torch

from common import CLASSES, build_model, eval_transform, load_image

try:  # HEIC с iPhone читается, если установлен pillow-heif
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff",
                    ".heic", ".heif"}
DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "weights" / "model.pt"
DEFAULT_ONNX = DEFAULT_WEIGHTS.with_suffix(".onnx")
# порог подобран choose_threshold.py
DEFAULT_UNKNOWN_THRESHOLD = 0.45


def parse_args():
    p = argparse.ArgumentParser(description="Классификация овощей на фото (CPU).")
    p.add_argument("--input", required=True, help="папка с фото (вложенные папки тоже обрабатываются)")
    p.add_argument("--output", required=True, help="куда записать CSV")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="файл весов (по умолчанию weights/model.pt)")
    p.add_argument("--onnx", nargs="?", const=str(DEFAULT_ONNX), default=None, metavar="ФАЙЛ",
                   help="считать через ONNX Runtime (по умолчанию weights/model.onnx); "
                        "нужен pip install onnxruntime")
    p.add_argument("--batch", type=int, default=16, help="сколько фото считать за раз")
    p.add_argument("--threads", type=int, default=0, help="потоков CPU (0 = как решит библиотека)")
    p.add_argument("--unknown-threshold", type=float, default=DEFAULT_UNKNOWN_THRESHOLD,
                   help=f"если уверенность ниже, ответ unknown (по умолчанию {DEFAULT_UNKNOWN_THRESHOLD}, "
                        "0 = не использовать)")
    p.add_argument("--benchmark", type=int, default=0,
                   help="дополнительно замерить чистое время сети на одном фото, N повторов")
    args = p.parse_args()
    if not 0 <= args.unknown_threshold <= 1:
        p.error("--unknown-threshold должен быть от 0 до 1")
    return args


def list_files(root):
    images, skipped = [], []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith("."):
            images.append(p)
        else:
            skipped.append(p)  # .txt, .DS_Store, ._файлы macOS и т.п.
    return images, skipped


def load_model(weights_path):
    ck = torch.load(weights_path, map_location="cpu", weights_only=True)
    if list(ck["classes"]) != CLASSES:
        sys.exit(f"Классы в весах {ck['classes']} не совпадают с {CLASSES}")
    model = build_model(ck["arch"], pretrained=False)  # pretrained=False: ничего не скачиваем
    model.load_state_dict(ck["state_dict"])
    return model.eval(), ck["arch"]


def load_onnx(path, threads):
    try:
        import onnxruntime as ort
    except ImportError:
        sys.exit("Для --onnx нужен onnxruntime: pip install onnxruntime")
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    meta = sess.get_modelmeta().custom_metadata_map
    if meta.get("classes", "").split(",") != CLASSES:
        sys.exit(f"Классы в {path} ({meta.get('classes')}) не совпадают с {CLASSES}")
    name = sess.get_inputs()[0].name
    return (lambda x: torch.from_numpy(sess.run(None, {name: x.numpy()})[0])), meta.get("arch", "?")


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
    weights = Path(args.onnx or args.weights)
    if not weights.is_file():
        hint = "python export_onnx.py" if args.onnx else "см. README: веса лежат в weights/model.pt"
        sys.exit(f"Нет файла весов: {weights} ({hint})")

    t_load = time.perf_counter()
    if args.onnx:
        forward, arch = load_onnx(weights, args.threads)
        engine = f"ONNX Runtime, CPU-потоков: {args.threads or 'авто'}"
    else:
        forward, arch = load_model(weights)
        engine = f"PyTorch, CPU-потоков: {torch.get_num_threads()}"
    t_load = time.perf_counter() - t_load
    transform = eval_transform()
    files, skipped = list_files(in_dir)
    print(f"Модель {arch} ({engine}) загружена за {t_load:.1f} c")
    print(f"Фото: {len(files)}" + (f"; пропущено файлов не-изображений: {len(skipped)}" if skipped else ""))

    results, bad, low = {}, 0, 0
    batch_x, batch_names = [], []
    first_x = None

    def run_batch():
        nonlocal low
        with torch.inference_mode():
            probs = torch.softmax(forward(torch.stack(batch_x)), dim=1)
        conf, idx = probs.max(dim=1)
        for name, c, i in zip(batch_names, conf.tolist(), idx.tolist()):
            if c >= args.unknown_threshold:
                results[name] = (CLASSES[i], c)
            else:
                results[name] = ("unknown", c)
                low += 1
        batch_x.clear()
        batch_names.clear()

    t0 = time.perf_counter()
    for path in files:
        name = path.relative_to(in_dir).as_posix()
        try:
            x = transform(load_image(path))
        except Exception as e:  # битый или неподдерживаемый файл
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
    if args.unknown_threshold > 0:
        print(f"unknown из-за низкой уверенности (порог {args.unknown_threshold:g}): {low}")
    print(f"CSV: {out}")

    if args.benchmark and first_x is not None:  # чистое время сети на одном фото
        x1 = first_x.unsqueeze(0)
        with torch.inference_mode():
            for _ in range(5):
                forward(x1)
            times = []
            for _ in range(args.benchmark):
                t = time.perf_counter()
                forward(x1)
                times.append(1000 * (time.perf_counter() - t))
        print(f"Сеть на 1 фото (batch=1, {args.benchmark} повторов): медиана {statistics.median(times):.1f} мс, "
              f"min {min(times):.1f} мс")


if __name__ == "__main__":
    main()
