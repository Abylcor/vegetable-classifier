#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_onnx.py - экспорт модели в ONNX, проверка совпадения ответов и сравнение скорости на CPU.

  pip install -r requirements-onnx.txt
  python export_onnx.py                              # weights/model.pt -> weights/model.onnx
  python export_onnx.py --check-dir папка/с/фото     # сверить ответы ещё и на этих фото

Что делает:
  1. Загружает weights/model.pt так же, как predict.py, и сохраняет сеть в ONNX
     с переменным размером пачки. В файл записываются классы и параметры предобработки:
     predict.py --onnx проверяет, что классы совпадают.
  2. Сверяет ONNX Runtime с PyTorch: на случайных тензорах и на настоящих фото (--check-dir).
  3. Замеряет время сети на CPU: одинаковое число потоков, пачка 1 и 16, прогрев,
     медиана по повторам. Предобработка в замер не входит: она одинаковая у обоих.
     Таблица печатается в консоль и пишется в results/onnx_benchmark.json.
"""
import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

from common import CLASSES, IMG_SIZE, MEAN, RESIZE, STD, eval_transform, load_image
from predict import DEFAULT_WEIGHTS, list_files, load_model

HERE = Path(__file__).resolve().parent
DEFAULT_ONNX = DEFAULT_WEIGHTS.with_suffix(".onnx")


def parse_args():
    p = argparse.ArgumentParser(description="Экспорт в ONNX и сравнение с PyTorch на CPU.")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="веса PyTorch (weights/model.pt)")
    p.add_argument("--output", default=str(DEFAULT_ONNX), help="куда сохранить ONNX (weights/model.onnx)")
    p.add_argument("--check-dir", action="append", default=[],
                   help="папка с фото для сверки ответов (можно указать несколько раз)")
    p.add_argument("--threads", type=int, default=os.cpu_count() or 1,
                   help="потоков CPU, одинаково для обоих движков (по умолчанию все)")
    p.add_argument("--runs", type=int, default=100, help="повторов замера для пачки 1")
    p.add_argument("--rounds", type=int, default=3,
                   help="сколько раз по очереди повторить замер обоих движков (берём медиану)")
    p.add_argument("--report", default=str(HERE / "results" / "onnx_benchmark.json"))
    return p.parse_args()


def cpu_name():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def export(model, path):
    """Новый экспортёр (torch.export) с переменной пачкой; если его нет - старый (TorchScript)."""
    x = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
    names = dict(input_names=["image"], output_names=["logits"])
    try:
        batch = torch.export.Dim("batch", min=1, max=4096)
        torch.onnx.export(model, (x,), str(path), dynamo=True, external_data=False,
                          dynamic_shapes=({0: batch},), **names)
        return "torch.export (dynamo=True)"
    except Exception as e:  # старый torch или нет onnxscript
        print(f"Новый экспортёр не сработал ({type(e).__name__}: {str(e)[:200]}), пробую старый.",
              file=sys.stderr)
        torch.onnx.export(model, (x,), str(path), dynamo=False, opset_version=17,
                          dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, **names)
        return "TorchScript (dynamo=False)"


def add_metadata(path, arch):
    import onnx
    m = onnx.load(str(path))
    meta = {"classes": ",".join(CLASSES), "arch": arch, "img_size": str(IMG_SIZE), "resize": str(RESIZE),
            "mean": ",".join(map(str, MEAN)), "std": ",".join(map(str, STD)),
            "preprocessing": "common.py: EXIF, RGB, resize 256, center crop 224, ImageNet normalize"}
    del m.metadata_props[:]
    for k, v in meta.items():
        prop = m.metadata_props.add()
        prop.key, prop.value = k, v
    onnx.checker.check_model(m)
    onnx.save(m, str(path))
    return m.opset_import[0].version


def session(path, threads):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def compare(model, sess, x):
    """Логиты и вероятности обоих движков на одной пачке."""
    with torch.inference_mode():
        a = model(x).numpy()
    b = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0]
    return a, b


def check_on_images(model, sess, dirs):
    transform = eval_transform()
    files = [f for d in dirs for f in list_files(Path(d))[0]]
    pa, pb, bad = [], [], 0
    for i in range(0, len(files), 16):
        xs = []
        for f in files[i:i + 16]:
            try:
                xs.append(transform(load_image(f)))
            except Exception:
                bad += 1
        if xs:
            a, b = compare(model, sess, torch.stack(xs))
            pa.append(softmax(a))
            pb.append(softmax(b))
    if not pa:
        return None
    pa, pb = np.concatenate(pa), np.concatenate(pb)
    return {"photos": int(len(pa)), "unreadable": bad,
            "same_label": int((pa.argmax(1) == pb.argmax(1)).sum()),
            "max_prob_diff": float(np.abs(pa - pb).max())}


def timed(fn, runs, warmup=10):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        times.append(1000 * (time.perf_counter() - t))
    return statistics.median(times)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    args = parse_args()
    torch.set_num_threads(args.threads)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    model, arch = load_model(args.weights)
    how = export(model, out)
    opset = add_metadata(out, arch)
    print(f"ONNX: {out} ({out.stat().st_size / 2**20:.1f} МБ), экспорт: {how}, opset {opset}")

    sess = session(out, args.threads)
    x = torch.randn(8, 3, IMG_SIZE, IMG_SIZE, generator=torch.Generator().manual_seed(0))
    a, b = compare(model, sess, x)
    # на шуме логиты огромные (десятки), поэтому сравниваем относительную разницу
    random_check = {"batch": 8, "same_label": int((a.argmax(1) == b.argmax(1)).sum()),
                    "rel_logit_diff": float(np.abs(a - b).max() / np.abs(a).max()),
                    "max_prob_diff": float(np.abs(softmax(a) - softmax(b)).max())}
    print(f"Случайные тензоры: одинаковый ответ у {random_check['same_label']} из 8, "
          f"разница вероятностей до {random_check['max_prob_diff']:.1e}")
    image_check = check_on_images(model, sess, args.check_dir) if args.check_dir else None
    if image_check:
        print(f"Фото: {image_check['photos']}, одинаковый ответ у {image_check['same_label']}, "
              f"разница вероятностей до {image_check['max_prob_diff']:.1e}")

    name = sess.get_inputs()[0].name
    speed = {}
    for b in (1, 16):
        xb = torch.randn(b, 3, IMG_SIZE, IMG_SIZE)
        xn = xb.numpy()
        runs = args.runs if b == 1 else max(10, args.runs // 8)
        t_torch, t_onnx = [], []
        for _ in range(args.rounds):          # по очереди, чтобы фоновая нагрузка влияла на обоих одинаково
            with torch.inference_mode():
                t_torch.append(timed(lambda: model(xb), runs))
            t_onnx.append(timed(lambda: sess.run(None, {name: xn}), runs))
        speed[b] = {"pytorch_ms": statistics.median(t_torch) / b, "onnx_ms": statistics.median(t_onnx) / b}

    print(f"\nВремя сети на одно фото, CPU {cpu_name()}, потоков: {args.threads}")
    print(f"{'пачка':>6} {'PyTorch, мс':>12} {'ONNX Runtime, мс':>17} {'быстрее в':>10}")
    for b, s in speed.items():
        print(f"{b:>6} {s['pytorch_ms']:>12.1f} {s['onnx_ms']:>17.1f} {s['pytorch_ms'] / s['onnx_ms']:>9.2f}x")

    import onnxruntime
    report = {
        "cpu": cpu_name(), "threads": args.threads, "os": platform.platform(),
        "python": platform.python_version(), "torch": torch.__version__,
        "onnxruntime": onnxruntime.__version__, "export": how, "opset": opset,
        "size_mb": {"pt": round(Path(args.weights).stat().st_size / 2**20, 1),
                    "onnx": round(out.stat().st_size / 2**20, 1)},
        "check_random": random_check, "check_images": image_check,
        "ms_per_photo": {str(b): {k: round(v, 2) for k, v in s.items()} for b, s in speed.items()},
        "note": "только сеть, без чтения и предобработки; медиана по повторам после прогрева",
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Отчёт: {args.report}")


if __name__ == "__main__":
    main()
