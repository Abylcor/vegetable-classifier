
"""
train.py - дообучение предобученной сети (transfer learning) по таблице разбиения.

Примеры (Kaggle, GPU):
  python train.py --data /kaggle/input --splits splits.csv --test-fold 0 --out runs/fold0
  python train.py --data /kaggle/input --splits splits.csv --test-fold 1 --out runs/fold1
  python train.py --data /kaggle/input --splits splits.csv --test-fold none --out runs/final
  python train.py --data /kaggle/input --splits author_split.csv --train-folds train --test-fold test --out runs/author

Схема обучения (одинаковая для всех запусков, выбрана заранее, на тестовом фолде не подбирается):
  1) "голова": веса ImageNet заморожены, учим только новый классификатор (--epochs-head эпох);
  2) "вся сеть": размораживаем всё и дообучаем с маленьким LR и косинусным затуханием (--epochs-full эпох).
Отдельной валидации и early stopping нет: у Cucumber, Potato и Raddish всего по 2 видео,
"чистую" валидацию из новых видео взять неоткуда, а подбирать эпоху по тестовому фолду - утечка.

В папку --out пишутся:
  model.pt      веса + архитектура + классы (для predict.py)
  log.csv       loss и accuracy на обучении по эпохам
  preds.csv     предсказания на тестовом фолде с вероятностями всех классов
  metrics.json  accuracy, macro-F1, precision/recall/F1 по классам, матрица ошибок
"""
import argparse
import csv
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from common import (ARCHS, CLASSES, IMG_SIZE, MEAN, RESIZE, STD, build_file_index, build_model,
                    eval_transform, head_module, load_image, train_transform)


class ImageDataset(Dataset):
    def __init__(self, files, labels, transform):
        self.files, self.labels, self.transform = files, labels, transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return self.transform(load_image(self.files[i])), self.labels[i]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(_):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def read_split_rows(splits_csv, data_root):
    """Читает таблицу разбиения и находит каждый файл на диске."""
    with open(splits_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    index, missing = None, []
    for r in rows:
        p = Path(data_root) / r["path"]
        if not p.exists():
            if index is None:
                print("Пути из таблицы не найдены напрямую, ищу файлы по имени класса и файла...", flush=True)
                index = build_file_index(data_root)
            p = index.get(f"{r['label']}/{Path(r['path']).name}")
        if p is None:
            missing.append(r["path"])
        r["file"] = p
    if missing:
        sys.exit(f"Не найдено {len(missing)} файлов, например {missing[:3]}. Проверьте --data.")
    return rows


def drop_unreadable(rows):
    """Открываем каждый файл до обучения: один битый файл не должен ронять обучение посреди эпохи.
    В нашем датасете таких нет (аудит открыл все файлы), но проверка дешёвая."""
    good, bad = [], []
    for r in rows:
        try:
            load_image(r["file"]).load()
            good.append(r)
        except Exception as e:
            bad.append((r["path"], f"{type(e).__name__}: {e}"[:120]))
    if bad:
        print(f"! Пропускаю {len(bad)} нечитаемых файлов:")
        for path, err in bad[:10]:
            print(f"   {path}: {err}")
    return good


def limit_per_class(rows, n, seed):
    """Для быстрых проверок кода: не больше n фото каждого класса."""
    rng = random.Random(seed)
    out = []
    for c in CLASSES:
        part = [r for r in rows if r["label"] == c]
        rng.shuffle(part)
        out += part[:n]
    return out


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, frozen_bn=()):
    model.train()
    for m in frozen_bn:          # замороженная часть: BatchNorm не обновляет статистики
        m.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        loss_sum += loss.item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        total += len(y)
    return loss_sum / total, correct / total


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    chunks = [torch.softmax(model(x.to(device)), dim=1).cpu() for x, _ in loader]
    return torch.cat(chunks).numpy()


def compute_metrics(y_true, y_pred, k):
    """accuracy, precision/recall/F1 по каждому классу, macro-F1 и матрица ошибок.
    macro-F1 = среднее F1 по классам без весов: маленький Raddish весит столько же,
    сколько большой Squash, поэтому провал на редком классе не спрячется."""
    cm = np.zeros((k, k), dtype=int)                 # строки - истинный класс, столбцы - предсказанный
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    tp = np.diag(cm).astype(float)
    precision = np.divide(tp, cm.sum(0), out=np.zeros(k), where=cm.sum(0) > 0)
    recall = np.divide(tp, cm.sum(1), out=np.zeros(k), where=cm.sum(1) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(k), where=(precision + recall) > 0)
    present = cm.sum(1) > 0                          # классы, которые есть в тесте
    return {
        "accuracy": float(tp.sum() / cm.sum()),
        "macro_f1": float(f1[present].mean()),
        "per_class": {CLASSES[i]: {"precision": round(float(precision[i]), 4), "recall": round(float(recall[i]), 4),
                                   "f1": round(float(f1[i]), 4), "support": int(cm[i].sum())} for i in range(k)},
        "confusion_matrix": cm.tolist(),
    }


def print_metrics(m):
    print(f"\naccuracy = {m['accuracy']:.4f}   macro-F1 = {m['macro_f1']:.4f}")
    print(f"{'class':<10}{'precision':>10}{'recall':>8}{'f1':>8}{'support':>9}")
    for c, v in m["per_class"].items():
        print(f"{c:<10}{v['precision']:>10.3f}{v['recall']:>8.3f}{v['f1']:>8.3f}{v['support']:>9}")
    print("\nМатрица ошибок (строка = истинный класс, столбец = предсказание):")
    print(" " * 10 + "".join(f"{c[:8]:>9}" for c in CLASSES))
    for c, row in zip(CLASSES, m["confusion_matrix"]):
        print(f"{c:<10}" + "".join(f"{v:>9}" for v in row))


def parse_args():
    p = argparse.ArgumentParser(description="Transfer learning: классификатор овощей.")
    p.add_argument("--data", required=True, help="корень с датасетом (на Kaggle можно /kaggle/input)")
    p.add_argument("--splits", required=True, help="splits.csv (наше разбиение) или author_split.csv")
    p.add_argument("--test-fold", required=True, help="фолд для проверки; 'none' = учить на всём без проверки")
    p.add_argument("--train-folds", default=None, help="через запятую; по умолчанию все, кроме тестового")
    p.add_argument("--out", required=True, help="папка для результатов")
    p.add_argument("--arch", default="efficientnet_b0", choices=ARCHS)
    p.add_argument("--epochs-head", type=int, default=2)
    p.add_argument("--epochs-full", type=int, default=6)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--lr-full", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--wd", type=float, default=0.01, help="weight decay для AdamW")
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--aug", default="full", choices=["full", "light"])
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--no-pretrained", action="store_true", help="без весов ImageNet (только для проверки кода)")
    p.add_argument("--limit", type=int, default=0, help="макс. фото на класс (только для быстрой проверки кода)")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
                          (args.device if args.device != "auto" else "cpu"))
    print(f"Устройство: {device}" + ("" if device.type == "cuda" else "  (на CPU обучение медленное; на Kaggle включите GPU)"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = read_split_rows(args.splits, args.data)
    folds = sorted({r["fold"] for r in rows})
    test_fold = None if args.test_fold.lower() == "none" else args.test_fold
    if test_fold is not None and test_fold not in folds:
        sys.exit(f"Фолда {test_fold} нет в таблице; есть: {folds}")
    train_folds = args.train_folds.split(",") if args.train_folds else [f for f in folds if f != test_fold]
    train_rows = [r for r in rows if r["fold"] in train_folds]
    test_rows = [r for r in rows if r["fold"] == test_fold] if test_fold is not None else []
    if args.limit:
        train_rows = limit_per_class(train_rows, args.limit, args.seed)
        test_rows = limit_per_class(test_rows, args.limit, args.seed)
    t = time.time()
    train_rows, test_rows = drop_unreadable(train_rows), drop_unreadable(test_rows)
    print(f"Проверил {len(train_rows) + len(test_rows)} файлов за {time.time() - t:.0f} c")
    # одна и та же группа (видео) в train и test = утечка; в нашем разбиении должно быть 0
    shared = {r.get("group") for r in train_rows} & {r.get("group") for r in test_rows}
    print(f"Групп (видео/сцен) одновременно в train и test: {len(shared)}")

    y_index = {c: i for i, c in enumerate(CLASSES)}
    train_y = [y_index[r["label"]] for r in train_rows]
    test_y = [y_index[r["label"]] for r in test_rows]
    counts = Counter(train_y)
    print(f"train: фолды {train_folds}, {len(train_rows)} фото; test: фолд {test_fold}, {len(test_rows)} фото")
    print("train по классам:", {CLASSES[k]: counts[k] for k in range(len(CLASSES))})
    if len(counts) < len(CLASSES):
        sys.exit("В обучающей части нет какого-то класса - так учить нельзя.")

    g = torch.Generator()
    g.manual_seed(args.seed)
    loader_kw = dict(num_workers=args.workers, pin_memory=device.type == "cuda", worker_init_fn=seed_worker)
    train_loader = DataLoader(ImageDataset([r["file"] for r in train_rows], train_y, train_transform(args.aug)),
                              batch_size=args.batch, shuffle=True, drop_last=False, generator=g, **loader_kw)

    model = build_model(args.arch, pretrained=not args.no_pretrained).to(device)
    # веса классов: редкие классы весят больше, иначе модели выгодно чаще говорить "Squash"
    w = torch.tensor([len(train_y) / (len(CLASSES) * counts[k]) for k in range(len(CLASSES))], dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=w.to(device), label_smoothing=args.label_smoothing)

    history = []

    def log(phase, epoch, loss, acc, lr, sec):
        history.append({"phase": phase, "epoch": epoch, "train_loss": round(loss, 4), "train_acc": round(acc, 4),
                        "lr": lr, "seconds": round(sec, 1)})
        print(f"[{phase}] epoch {epoch}: train loss {loss:.4f}, train acc {acc:.4f}, lr {lr:.2e}, {sec:.0f} c",
              flush=True)

    # Этап 1: учим только голову, остальная сеть заморожена
    head = head_module(model, args.arch)
    head_ids = {id(p) for p in head.parameters()}
    for p in model.parameters():
        p.requires_grad = id(p) in head_ids
    head_mods = {id(m) for m in head.modules()}
    frozen_bn = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm) and id(m) not in head_mods]
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr_head, weight_decay=args.wd)
    for e in range(1, args.epochs_head + 1):
        t = time.time()
        loss, acc = train_one_epoch(model, train_loader, criterion, opt, None, device, frozen_bn)
        log("head", e, loss, acc, args.lr_head, time.time() - t)

    # Этап 2: размораживаем всю сеть, маленький LR с косинусным затуханием
    for p in model.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_full, weight_decay=args.wd)
    steps = max(1, args.epochs_full * len(train_loader))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    for e in range(1, args.epochs_full + 1):
        t = time.time()
        lr_start = opt.param_groups[0]["lr"]
        loss, acc = train_one_epoch(model, train_loader, criterion, opt, sched, device)
        log("full", e, loss, acc, lr_start, time.time() - t)

    torch.save({"arch": args.arch, "classes": CLASSES, "img_size": IMG_SIZE, "resize": RESIZE,
                "mean": list(MEAN), "std": list(STD), "state_dict": model.state_dict(),
                "train_args": {k: str(v) for k, v in vars(args).items()}}, out / "model.pt")
    with open(out / "log.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(history[0].keys()) if history else ["phase"])
        wr.writeheader()
        wr.writerows(history)

    result = {"arch": args.arch, "test_fold": test_fold, "train_folds": train_folds,
              "n_train": len(train_rows), "n_test": len(test_rows), "history": history,
              "args": {k: str(v) for k, v in vars(args).items()}}
    if test_rows:
        test_loader = DataLoader(ImageDataset([r["file"] for r in test_rows], test_y, eval_transform()),
                                 batch_size=args.batch * 2, shuffle=False, **loader_kw)
        probs = predict_probs(model, test_loader, device)
        pred = probs.argmax(1)
        with open(out / "preds.csv", "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["path", "label", "group", "pred", "confidence"] + [f"p_{c}" for c in CLASSES])
            for r, pr, p in zip(test_rows, pred, probs):
                wr.writerow([r["path"], r["label"], r.get("group", ""), CLASSES[pr], round(float(p.max()), 4)]
                            + [round(float(v), 4) for v in p])
        m = compute_metrics(test_y, pred.tolist(), len(CLASSES))
        print_metrics(m)
        result.update(m)
    (out / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nСохранено в {out}/: model.pt, log.csv" + (", preds.csv, metrics.json" if test_rows else ", metrics.json"))


if __name__ == "__main__":
    main()
