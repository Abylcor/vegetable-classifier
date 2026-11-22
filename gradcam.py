#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gradcam.py - Grad-CAM: на какую часть картинки смотрит модель, когда принимает решение.

Берём предсказания на тестовых фолдах (runs/fold0, runs/fold1) и для каждого примера
используем ту модель, которая НЕ видела его видео (модель своего фолда). Выбираем:
  * самые уверенные ошибки (не больше --per-group из одного видео, чтобы примеры были разными);
  * по одному правильному примеру каждого класса с типичной (медианной) уверенностью.
Для ошибок рисуем две карты: за предсказанный класс и за правильный.

Как работает Grad-CAM: берём выход последнего свёрточного слоя (1280 карт 7x7 у EfficientNet-B0)
и градиент оценки нужного класса по этим картам. Среднее градиента по карте = её важность для
класса; взвешенная сумма карт, обрезанная снизу нулём (ReLU), и есть тепловая карта.

Запуск (Kaggle):
  python gradcam.py --data /kaggle/input --runs runs --out runs/gradcam
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torchvision import transforms as T  # noqa: E402

from common import (CLASSES, IMG_SIZE, RESIZE, build_file_index, build_model, eval_transform,  # noqa: E402
                    gradcam_layer, load_image)


class GradCAM:
    def __init__(self, model, layer):
        self.model = model
        self.acts = self.grads = None
        layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, inputs, output):
        self.acts = output
        output.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self.grads = grad

    def __call__(self, x, class_idx):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        logits[0, class_idx].backward()
        weights = self.grads.mean(dim=(2, 3), keepdim=True)              # важность каждой карты
        cam = F.relu((weights * self.acts).sum(dim=1, keepdim=True))     # взвешенная сумма
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy(), torch.softmax(logits, dim=1)[0].detach().cpu().numpy()


def load_fold_model(run_dir, device):
    ck = torch.load(Path(run_dir) / "model.pt", map_location=device, weights_only=True)
    model = build_model(ck["arch"], pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model, ck["arch"]


def pick_examples(rows, n_errors, n_correct, per_group):
    errors = sorted([r for r in rows if r["label"] != r["pred"]], key=lambda r: -float(r["confidence"]))
    chosen, used = [], {}
    for r in errors:
        if used.get(r["group"], 0) < per_group:
            chosen.append(r)
            used[r["group"]] = used.get(r["group"], 0) + 1
        if len(chosen) == n_errors:
            break
    correct = [r for r in rows if r["label"] == r["pred"]]
    for c in CLASSES[:n_correct]:
        part = [r for r in correct if r["label"] == c]
        if part:
            med = float(np.median([float(r["confidence"]) for r in part]))
            chosen.append(min(part, key=lambda r: (abs(float(r["confidence"]) - med), r["path"])))
    return chosen


def main():
    ap = argparse.ArgumentParser(description="Grad-CAM на примерах из тестовых фолдов.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--folds", default="fold0,fold1", help="папки запусков с preds.csv и model.pt")
    ap.add_argument("--out", default="runs/gradcam")
    ap.add_argument("--n-errors", type=int, default=6)
    ap.add_argument("--n-correct", type=int, default=6)
    ap.add_argument("--per-group", type=int, default=1, help="ошибок из одного видео, не больше")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    for run in args.folds.split(","):
        with open(Path(args.runs) / run / "preds.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                r["run"] = run
                rows.append(r)
    examples = pick_examples(rows, args.n_errors, args.n_correct, args.per_group)

    index = None
    for r in examples:
        p = Path(args.data) / r["path"]
        if not p.exists():
            index = index or build_file_index(args.data)
            p = index[f"{r['label']}/{Path(r['path']).name}"]
        r["file"] = p

    show = T.Compose([T.Resize(RESIZE), T.CenterCrop(IMG_SIZE)])   # то, что видит сеть, без нормализации
    to_input = eval_transform()
    models, cams = {}, {}
    n = len(examples)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3.1 * n))
    axes = np.atleast_2d(axes)
    log = []
    for i, r in enumerate(examples):
        if r["run"] not in models:
            model, arch = load_fold_model(Path(args.runs) / r["run"], device)
            models[r["run"]] = GradCAM(model, gradcam_layer(model, arch))
        cam_fn = models[r["run"]]
        img = load_image(r["file"])
        x = to_input(img).unsqueeze(0).to(device)
        pred_i, true_i = CLASSES.index(r["pred"]), CLASSES.index(r["label"])
        cam_pred, probs = cam_fn(x, pred_i)
        is_error = pred_i != true_i
        cam_true = cam_fn(x, true_i)[0] if is_error else None
        view = np.asarray(show(img))

        color = "red" if is_error else "green"
        video = r["group"].split("/")[-1]
        axes[i, 0].imshow(view)
        axes[i, 0].set_title(f"{i + 1}. true: {r['label']}  ({video})", fontsize=9, color=color)
        axes[i, 1].imshow(view)
        axes[i, 1].imshow(cam_pred, cmap="jet", alpha=0.45)
        axes[i, 1].set_title(f"CAM pred: {r['pred']} ({probs[pred_i]:.2f})", fontsize=9, color=color)
        if is_error:
            axes[i, 2].imshow(view)
            axes[i, 2].imshow(cam_true, cmap="jet", alpha=0.45)
            axes[i, 2].set_title(f"CAM true: {r['label']} ({probs[true_i]:.2f})", fontsize=9)
        for a in axes[i]:
            a.axis("off")
        log.append({"n": i + 1, "path": r["path"], "true": r["label"], "pred": r["pred"],
                    "confidence": round(float(probs[pred_i]), 4), "model": r["run"], "group": r["group"],
                    "error": int(is_error)})
    fig.suptitle("Grad-CAM: red = error, green = correct (model of the fold that did not see this video)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / (3.1 * n)))   # место под общий заголовок
    fig.savefig(out / "gradcam_grid.png", dpi=80)
    with open(out / "examples.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0].keys()))
        w.writeheader()
        w.writerows(log)
    print(f"{len(log)} примеров ({sum(e['error'] for e in log)} ошибок) -> {out / 'gradcam_grid.png'}")
    for e in log:
        print(f"{e['n']:>2}. {'ОШИБКА ' if e['error'] else 'верно  '} {e['true']:<9} -> {e['pred']:<9} "
              f"{e['confidence']:.2f}  {e['group']}")


if __name__ == "__main__":
    main()
