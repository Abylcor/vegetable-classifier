#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_splits.py - честное разбиение датасета овощей на фолды по группам (видео).

Почему так (по результатам audit_dataset.py):
  * 3694 из 3879 фото - кадры 19 видео (имя YYYYMMDD_HHMMSS_frame_N.jpg), соседние кадры
    почти одинаковые. Если кадры одного видео попадут и в train, и в test, тест
    проверяет запоминание, а не обобщение. Поэтому единица разбиения - видео.
  * Остальные 185 - фото (YYYYMMDD_HHMMSS.jpg), обычно снятые в ту же минуту, что и видео
    (та же сцена). Фото присоединяем к ближайшему по времени видео того же класса,
    если разница <= --photo-gap секунд, иначе фото образуют группу "класс + день".
  * final_dataset/ не используем: это копии тех же файлов, разложенные с пересечениями.

Фолды строит StratifiedGroupKFold: группа целиком в одном фолде, пропорции классов
по фолдам примерно равны. Обучение на фолдах != k, проверка на фолде k; метрики
считаем по предсказаниям out-of-fold (каждое фото предсказано моделью, не видевшей его видео).

Запуск:
  python make_splits.py --data путь/к/data_raw --out splits.csv
  python make_splits.py --data путь/к/data_raw --out splits.csv --pairs audit_out/near_dup_pairs.csv
"""
import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

CLASSES = ["Capsicum", "Carrot", "Cucumber", "Potato", "Raddish", "Squash"]
NAME_RE = re.compile(r"^(\d{8})_(\d{6})(?:_frame_(\d+))?\.jpe?g$", re.IGNORECASE)
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_images(data_dir):
    """Все изображения в папках классов, кроме всего, что лежит внутри final_dataset."""
    items = []
    for p in sorted(data_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMG_EXT:
            continue
        parts = [s.lower() for s in p.relative_to(data_dir).parts]
        if "final_dataset" in parts:
            continue
        if p.parent.name in CLASSES:
            items.append({"path": p.relative_to(data_dir).as_posix(), "label": p.parent.name,
                          "filename": p.name})
    return items


def assign_groups(items, photo_gap):
    videos = defaultdict(dict)                       # label -> {video_id: datetime}
    for it in items:
        m = NAME_RE.match(it["filename"])
        it["ts"] = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S") if m else None
        it["kind"] = "frame" if (m and m.group(3) is not None) else ("photo" if m else "other")
        if it["kind"] == "frame":
            vid = f"{m.group(1)}_{m.group(2)}"
            it["group"] = f"{it['label']}/video_{vid}"
            videos[it["label"]][vid] = it["ts"]
    for it in items:
        if it["kind"] == "photo":
            cand = [(abs((ts - it["ts"]).total_seconds()), vid) for vid, ts in videos[it["label"]].items()]
            gap, vid = min(cand) if cand else (float("inf"), None)
            it["group"] = (f"{it['label']}/video_{vid}" if gap <= photo_gap
                           else f"{it['label']}/photos_{it['ts']:%Y%m%d}")
        elif it["kind"] == "other":                  # имя не по шаблону: отдельная группа
            it["group"] = f"{it['label']}/file_{it['filename']}"
    return items


def assign_folds(items, n_folds, seed):
    try:
        from sklearn.model_selection import StratifiedGroupKFold
    except ImportError:
        sys.exit("Нужен scikit-learn:  python -m pip install scikit-learn")
    y = [it["label"] for it in items]
    g = [it["group"] for it in items]
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, test_idx) in enumerate(sgkf.split(X=[[0]] * len(items), y=y, groups=g)):
        for i in test_idx:
            items[i]["fold"] = k
    return items


def report(items, n_folds):
    print("\nИзображений / групп по классам и фолдам:")
    print("class".ljust(10) + "".join(f"fold{k}".rjust(14) for k in range(n_folds)))
    ok = True
    for c in CLASSES:
        row = c.ljust(10)
        for k in range(n_folds):
            sel = [it for it in items if it["label"] == c and it["fold"] == k]
            row += f"{len(sel):>7} / {len({it['group'] for it in sel}):<4}"
            ok &= len(sel) > 0
        print(row)
    folds_per_group = defaultdict(set)
    for it in items:
        folds_per_group[it["group"]].add(it["fold"])
    leaked = [g for g, f in folds_per_group.items() if len(f) > 1]
    print(f"Групп всего: {len(folds_per_group)}; групп в нескольких фолдах: {len(leaked)}")
    if not ok:
        print("! В каком-то фолде нет одного из классов: метрики этого фолда будут неполными.")
    return ok and not leaked


def check_pairs(items, pairs_csv, max_dist):
    """Независимая проверка по pHash из аудита: нет ли почти-дубликатов в разных фолдах."""
    fold_of = {f"{it['label']}/{it['filename']}": it["fold"] for it in items}
    key = lambda rel: "/".join(rel.split("/")[-2:])
    crossing, total = Counter(), 0
    with open(pairs_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["transform"] != "same" or int(r["dist"]) > max_dist:
                continue
            fa, fb = fold_of.get(key(r["a_relpath"])), fold_of.get(key(r["b_relpath"]))
            if fa is None or fb is None or r["a_relpath"].split("/")[0].lower() == "final_dataset":
                continue
            total += 1
            if fa != fb:
                crossing[(r["a_cls"], r["b_cls"])] += 1
    print(f"\nПроверка по pHash (та же ориентация, d<={max_dist}): пар {total}, "
          f"из них в разных фолдах {sum(crossing.values())}")
    for (a, b), n in crossing.most_common():
        print(f"  {a} - {b}: {n}" + ("  (разные классы: проверьте глазами)" if a != b else ""))


def main():
    ap = argparse.ArgumentParser(description="Разбиение на фолды по видео (group k-fold).")
    ap.add_argument("--data", required=True, help="папка датасета (data_raw); final_dataset внутри игнорируется")
    ap.add_argument("--out", default="splits.csv")
    ap.add_argument("--n-folds", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--photo-gap", type=int, default=300, help="сек: фото ближе к видео - в его группу")
    ap.add_argument("--pairs", help="near_dup_pairs.csv из audit_dataset.py для проверки")
    ap.add_argument("--pairs-max-dist", type=int, default=8)
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    data_dir = Path(args.data).expanduser().resolve()
    items = find_images(data_dir)
    if not items:
        sys.exit(f"Не нашёл изображений в папках классов внутри {data_dir}")
    print(f"Изображений: {len(items)}; по классам: {dict(Counter(it['label'] for it in items))}")
    items = assign_groups(items, args.photo_gap)
    print("Типы файлов:", dict(Counter(it["kind"] for it in items)))
    items = assign_folds(items, args.n_folds, args.seed)
    ok = report(items, args.n_folds)
    if args.pairs:
        check_pairs(items, args.pairs, args.pairs_max_dist)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "group", "kind", "fold"])
        for it in sorted(items, key=lambda it: it["path"]):
            w.writerow([it["path"], it["label"], it["group"], it["kind"], it["fold"]])
    print(f"\nЗаписал {args.out}" + ("" if ok else "  (есть предупреждения выше)"))


if __name__ == "__main__":
    main()
