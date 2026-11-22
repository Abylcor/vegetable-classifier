#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_dataset.py - аудит датасета изображений (без обучения моделей).

Скрипт только читает файлы и ничего не меняет в датасете. Результат: папка --out
и архив <out>.zip на несколько МБ, который удобно отправить на проверку.

Что считается:
  inventory.csv      по строке на файл: корень/сплит/класс из пути, формат, размеры,
                     EXIF, оценка качества JPEG, MD5, pHash (+7 вариантов для отражений
                     и поворотов на 90/180/270), простые признаки съёмки (яркость,
                     контраст, насыщенность, фон по краям кадра, резкость), ближайшие
                     соседи по pHash (в своём корне, в другом сплите, в другом корне).
  near_dup_pairs.csv пары почти-дубликатов внутри каждого корня (pHash-расстояние <= --pair-max-dist).
  sheets/*.jpg       контрольные листы: случайные примеры по классам, пары на разных
                     расстояниях (чтобы выбрать порог глазами), пары train/test,
                     пары из разных классов, крупнейшие кластеры почти-дубликатов.
  sheets_index.csv   какие файлы изображены на листах.
  summary.txt/.json  сводка.

Зависимости: Pillow, numpy.
Запуск:
  python audit_dataset.py --data путь/к/распакованному/датасету --out audit_out
"""
import argparse
import csv
import hashlib
import io
import json
import os
import random
import re
import shutil
import statistics
import sys
import time
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

RESAMPLING = getattr(Image, "Resampling", Image)  # Pillow < 9.1 не имеет Image.Resampling

CLASSES = ["Capsicum", "Carrot", "Cucumber", "Potato", "Raddish", "Squash"]
CLASS_ALIASES = {c.lower(): c for c in CLASSES}
CLASS_ALIASES["radish"] = "Raddish"
SPLIT_ALIASES = {"train": "train", "training": "train", "test": "test", "testing": "test",
                 "val": "val", "valid": "val", "validation": "val"}

FEAT_SIDE = 256   # длинная сторона картинки, на которой считаются признаки съёмки
HASH_SIDE = 32    # pHash: серая 32x32 -> DCT -> левый верхний блок 8x8 (как в imagehash.phash)
HASH_LOW = 8
CHUNK = 1024      # сколько запросов за раз в матричном подсчёте расстояний
SEED = 42
TRANSFORMS = ["same", "flip_lr", "flip_tb", "rot180", "transpose", "rot90", "rot270", "transverse"]

STD_LUMA_QTABLE = [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
                   14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
                   18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
                   49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99]

COLUMNS = [
    "relpath", "root", "split", "cls", "cls_known", "filename", "name_template", "ext",
    "bytes", "ok", "error", "warning", "format", "mode", "width", "height",
    "oriented_w", "oriented_h", "exif_orientation", "n_frames", "has_alpha", "progressive",
    "jpeg_q_est", "exif_make", "exif_model", "exif_software", "exif_datetime",
    "md5", "md5_dups_same_root", "md5_copies_other_roots", "phash", "phash_variants",
    "mean_r", "mean_g", "mean_b", "luma_mean", "luma_std", "sat_mean",
    "border_luma_mean", "border_luma_std", "border_white_frac", "border_dark_frac", "lap_var",
    "nn_same_root_dist", "nn_same_root_relpath", "nn_other_split_dist", "nn_other_split_relpath",
    "nn_other_root_dist", "nn_other_root_relpath", "near_cluster", "near_cluster_size",
]


# --------------------------------------------------------------------------------------
# Разбор пути и имени файла
# --------------------------------------------------------------------------------------
def parse_relpath(rel):
    """root = верхняя папка (Dataset, final_dataset ...), split = train/test/val по любой
    папке в пути (иначе 'all'), cls = папка с именем одного из 6 классов (иначе родительская)."""
    dirs = PurePosixPath(rel).parts[:-1]
    top = dirs[0] if dirs else "."
    root = "." if (not dirs or top.lower() in CLASS_ALIASES or top.lower() in SPLIT_ALIASES) else top
    split, cls = "all", None
    for d in dirs:
        split = SPLIT_ALIASES.get(d.lower(), split)
        cls = CLASS_ALIASES.get(d.lower(), cls)
    known = cls is not None
    if cls is None:
        cls = dirs[-1] if (len(dirs) > 1 or root == ".") and dirs else "-"
    return root, split, cls, int(known)


_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
_LONGHEX = re.compile(r"(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{12,}", re.I)
_DIGITS = re.compile(r"\d+")


def name_template(filename):
    """Шаблон имени: цифры -> '#', uuid/длинный hex -> метка. 'IMG_20230101_1234.jpg' -> 'IMG_#_#.jpg'.
    По шаблонам видно, откуда фото: камера, скачано из поиска, кадры видео, аугментации..."""
    stem, ext = os.path.splitext(filename)
    t = _UUID.sub("<uuid>", stem)
    t = _LONGHEX.sub("<hex>", t)
    t = _DIGITS.sub("#", t)
    return t + ext.lower()


# --------------------------------------------------------------------------------------
# Анализ одного файла (выполняется в отдельных процессах)
# --------------------------------------------------------------------------------------
def _dct_matrix(n):
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    return np.cos(np.pi * k * (2 * i + 1) / (2 * n))


DCT = _dct_matrix(HASH_SIDE)
_SIGNS = (-1.0) ** np.arange(HASH_LOW)
SIGN_LR = np.tile(_SIGNS, (HASH_LOW, 1))   # (-1)^v, v - горизонтальная частота
SIGN_TB = SIGN_LR.T                         # (-1)^u, u - вертикальная частота


def phash_variants(rgb):
    """pHash как в imagehash.phash, плюс хеши 7 отражённых/повёрнутых версий картинки.
    Отражение слева-направо меняет знак DCT-коэффициентов с нечётной горизонтальной
    частотой, транспонирование картинки транспонирует DCT. Поэтому все 8 вариантов
    считаются из одного DCT, без повторного ресайза."""
    gray = rgb.convert("L").resize((HASH_SIDE, HASH_SIDE), RESAMPLING.LANCZOS)
    low = (DCT @ np.asarray(gray, dtype=np.float64) @ DCT.T)[:HASH_LOW, :HASH_LOW]
    base = [low, low * SIGN_LR, low * SIGN_TB, low * SIGN_LR * SIGN_TB]
    # порядок соответствует TRANSFORMS
    variants = base[:4] + [base[0].T, base[1].T, base[2].T, base[3].T]
    out = []
    for m in variants:
        bits = (m > np.median(m)).ravel()
        out.append(np.packbits(bits).tobytes().hex())
    return out


def to_rgb(img):
    """Любой режим -> RGB. Прозрачность кладём на белый фон."""
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.getchannel("A"))
        return bg
    return img.convert("RGB")


def jpeg_quality(img):
    """Грубая оценка качества JPEG (1..100) по таблице квантования яркости.
    Веб-картинки обычно 70-90, камеры телефонов 90-98."""
    q = getattr(img, "quantization", None)
    if not q or 0 not in q:
        return ""
    scale = 100.0 * sum(q[0]) / sum(STD_LUMA_QTABLE)
    est = (200.0 - scale) / 2.0 if scale <= 100 else 5000.0 / scale
    return int(round(min(100.0, max(1.0, est))))


def _clean(v):
    return str(v).replace("\x00", "").strip()[:60]


def read_exif(img):
    out = {"exif_orientation": 1}
    try:
        ex = img.getexif()
        if not ex:
            return out
        out["exif_orientation"] = int(ex.get(0x0112, 1) or 1)
        for key, tag in (("exif_make", 0x010F), ("exif_model", 0x0110),
                         ("exif_software", 0x0131), ("exif_datetime", 0x0132)):
            if ex.get(tag):
                out[key] = _clean(ex.get(tag))
        try:
            original = ex.get_ifd(0x8769).get(0x9003)   # DateTimeOriginal
            if original:
                out["exif_datetime"] = _clean(original)
        except Exception:
            pass
    except Exception:
        pass
    return out


def shooting_features(rgb):
    """Признаки условий съёмки на картинке с длинной стороной FEAT_SIDE.
    'Край' = полоса 10% по периметру кадра: обычно это фон, а не овощ."""
    w, h = rgb.size
    s = FEAT_SIDE / max(w, h)
    small = rgb.resize((max(1, round(w * s)), max(1, round(h * s))), RESAMPLING.BILINEAR)
    a = np.asarray(small, dtype=np.float32)
    luma = a @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    mx, mn = a.max(axis=2), a.min(axis=2)
    sat = (mx - mn) / np.maximum(mx, 1.0)
    hh, ww = luma.shape
    bh, bw = max(1, hh // 10), max(1, ww // 10)
    edge = np.zeros((hh, ww), dtype=bool)
    edge[:bh, :] = True
    edge[-bh:, :] = True
    edge[:, :bw] = True
    edge[:, -bw:] = True
    edge_px, edge_luma = a[edge], luma[edge]
    if hh >= 3 and ww >= 3:
        lap = (luma[:-2, 1:-1] + luma[2:, 1:-1] + luma[1:-1, :-2] + luma[1:-1, 2:]
               - 4.0 * luma[1:-1, 1:-1])
        lap_var = float(lap.var())        # дисперсия лапласиана: мало = размыто
    else:
        lap_var = float("nan")
    feats = {
        "mean_r": a[..., 0].mean(), "mean_g": a[..., 1].mean(), "mean_b": a[..., 2].mean(),
        "luma_mean": luma.mean(), "luma_std": luma.std(), "sat_mean": sat.mean(),
        "border_luma_mean": edge_luma.mean(), "border_luma_std": edge_luma.std(),
        "border_white_frac": (edge_px.min(axis=1) > 220).mean(),
        "border_dark_frac": (edge_px.max(axis=1) < 40).mean(),
        "lap_var": lap_var,
    }
    return {k: round(float(v), 4) for k, v in feats.items()}


def analyze_file(task):
    base, rel = task
    row = {"relpath": rel, "ok": 0, "error": "", "warning": ""}
    try:
        data = (Path(base) / rel).read_bytes()
    except OSError as e:
        row["error"] = f"read error: {e}"[:200]
        return row
    row["bytes"] = len(data)
    row["md5"] = hashlib.md5(data).hexdigest()
    if not data:
        row["error"] = "empty file"
        return row
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            with Image.open(io.BytesIO(data)) as im:
                row["format"], row["mode"] = im.format, im.mode
                row["width"], row["height"] = im.size
                row["n_frames"] = getattr(im, "n_frames", 1)
                row["has_alpha"] = int(im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info)
                row["progressive"] = int(bool(im.info.get("progressive") or im.info.get("progression")))
                row["jpeg_q_est"] = jpeg_quality(im)
                row.update(read_exif(im))
                if im.format == "JPEG":
                    im.draft("RGB", (2 * FEAT_SIDE, 2 * FEAT_SIDE))  # быстрое декодирование в уменьшенном виде
                im.load()  # здесь падают обрезанные/битые файлы
                rgb = to_rgb(ImageOps.exif_transpose(im))
        except Exception as e:
            msg = re.sub(r"\s*<_io\.BytesIO object at 0x[0-9a-fA-F]+>", "", str(e))
            row["error"] = f"{type(e).__name__}: {msg}"[:200]
            return row
        finally:
            if caught:
                row["warning"] = "; ".join(sorted({str(w.message)[:80] for w in caught}))[:200]
    o = row.get("exif_orientation", 1)
    rotated = o in (5, 6, 7, 8)
    row["oriented_w"] = row["height"] if rotated else row["width"]
    row["oriented_h"] = row["width"] if rotated else row["height"]
    hashes = phash_variants(rgb)
    row["phash"], row["phash_variants"] = hashes[0], " ".join(hashes[1:])
    row.update(shooting_features(rgb))
    row["ok"] = 1
    return row


# --------------------------------------------------------------------------------------
# Дубликаты
# --------------------------------------------------------------------------------------
def bits_matrix(hex_hashes):
    raw = np.frombuffer(bytes.fromhex("".join(hex_hashes)), dtype=np.uint8)
    return np.unpackbits(raw).reshape(len(hex_hashes), 64).astype(np.float32)


def min_hamming(query_bits, target_variants):
    """query_bits (q,64), target_variants (8,t,64) -> (q,t) минимальное расстояние Хэмминга
    по 8 вариантам цели и номер варианта (0 = без преобразования; при равенстве - он)."""
    qs = query_bits.sum(axis=1)[:, None]
    best = arg = None
    for g in range(target_variants.shape[0]):
        tv = target_variants[g]
        d = qs + tv.sum(axis=1)[None, :] - 2.0 * (query_bits @ tv.T)
        if best is None:
            best, arg = d, np.zeros(d.shape, dtype=np.int8)
        else:
            better = d < best
            best = np.where(better, d, best)
            arg[better] = g
    return np.rint(best).astype(np.int16), arg


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def find_near_duplicates(rows, ok_idx, pair_max_dist, near_thresh):
    """Ищет почти-дубликаты внутри каждого корня (Dataset/, final_dataset/ ...) и ближайших
    соседей: в своём корне, в другом сплите того же корня, в другом корне."""
    n = len(ok_idx)
    q_bits = bits_matrix([rows[i]["phash"] for i in ok_idx])
    variants = [q_bits] + [bits_matrix([rows[i]["phash_variants"].split()[g] for i in ok_idx])
                           for g in range(7)]
    all_variants = np.stack(variants)                         # (8, n, 64)
    roots = np.array([rows[i]["root"] for i in ok_idx])
    splits = np.array([rows[i]["split"] for i in ok_idx])
    nn = {key: (np.full(n, -1), np.full(n, -1)) for key in ("same_root", "other_split", "other_root")}
    pa, pb, pd, pg = [], [], [], []

    for r in sorted(set(roots)):
        members = np.nonzero(roots == r)[0]
        others = np.nonzero(roots != r)[0]
        tv_in = all_variants[:, members]
        tv_out = all_variants[:, others] if len(others) else None
        m_splits = splits[members]
        col = np.arange(len(members))
        for st in range(0, len(members), CHUNK):
            q = members[st:st + CHUNK]
            nq = len(q)
            rows_local = np.arange(st, st + nq)
            dist, var = min_hamming(q_bits[q], tv_in)
            # ближайший сосед в своём корне (кроме себя)
            d_self = dist.copy()
            d_self[np.arange(nq), rows_local] = 999
            j = d_self.argmin(axis=1)
            nn["same_root"][0][q] = d_self[np.arange(nq), j]
            nn["same_root"][1][q] = members[j]
            # ближайший сосед в другом сплите того же корня (утечка train/test)
            other_split = m_splits[None, :] != m_splits[st:st + nq][:, None]
            if other_split.any():
                d_os = np.where(other_split, dist, 999)
                j2 = d_os.argmin(axis=1)
                d2 = d_os[np.arange(nq), j2]
                has = d2 < 999
                nn["other_split"][0][q[has]] = d2[has]
                nn["other_split"][1][q[has]] = members[j2[has]]
            # пары (каждая один раз: j > i)
            mask = (dist <= pair_max_dist) & (col[None, :] > rows_local[:, None])
            a, b = np.nonzero(mask)
            pa.append(members[st + a]); pb.append(members[b])
            pd.append(dist[a, b]); pg.append(var[a, b])
            # ближайший сосед в другом корне (например, Dataset/ <-> final_dataset/)
            if tv_out is not None:
                d3, _ = min_hamming(q_bits[q], tv_out)
                j3 = d3.argmin(axis=1)
                nn["other_root"][0][q] = d3[np.arange(nq), j3]
                nn["other_root"][1][q] = others[j3]

    cat = (lambda xs, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(0, dt))
    pa, pb, pd, pg = cat(pa, np.int64), cat(pb, np.int64), cat(pd, np.int16), cat(pg, np.int8)
    order = np.lexsort((pb, pa, pd))
    pa, pb, pd, pg = pa[order], pb[order], pd[order], pg[order]

    uf = UnionFind(n)
    for a, b in zip(pa[pd <= near_thresh], pb[pd <= near_thresh]):
        uf.union(int(a), int(b))
    comp = [uf.find(i) for i in range(n)]
    sizes = Counter(comp)
    ranking = sorted(sizes, key=lambda c: (-sizes[c], c))
    cid = {c: k for k, c in enumerate(ranking)}             # 0 = самый большой кластер
    clusters = [cid[c] for c in comp]
    cluster_sizes = [sizes[c] for c in comp]
    return nn, (pa, pb, pd, pg), clusters, cluster_sizes


# --------------------------------------------------------------------------------------
# Контрольные листы
# --------------------------------------------------------------------------------------
_FONT = None


def _font():
    global _FONT
    if _FONT is None:
        try:
            _FONT = ImageFont.load_default(size=13)
        except TypeError:
            _FONT = ImageFont.load_default()
    return _FONT


def put_label(draw, xy, text, fg=(255, 255, 0), bg=(0, 0, 0)):
    text = str(text).encode("ascii", "replace").decode("ascii")  # шрифт по умолчанию может не знать кириллицу
    f = _font()
    try:
        l, t, r, b = draw.textbbox(xy, text, font=f)
    except AttributeError:
        w, h = draw.textsize(text, font=f)
        l, t, r, b = xy[0], xy[1], xy[0] + w, xy[1] + h
    draw.rectangle((l - 2, t - 1, r + 2, b + 1), fill=bg)
    draw.text(xy, text, fill=fg, font=f)


def load_thumb(base, rel, side):
    """Превью с сохранением пропорций на тёмно-сером поле (так видно и форму кадра, и белый фон)."""
    canvas = Image.new("RGB", (side, side), (45, 45, 45))
    try:
        with Image.open(Path(base) / rel) as im:
            if im.format == "JPEG":
                im.draft("RGB", (side * 2, side * 2))
            im = to_rgb(ImageOps.exif_transpose(im))
            im.thumbnail((side, side), RESAMPLING.BILINEAR)
            canvas.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
    except Exception:
        canvas.paste((160, 0, 0), (0, 0, side, side))
    return canvas


def new_sheet(width, height, title):
    sheet = Image.new("RGB", (width, height + 26), (20, 20, 20))
    put_label(ImageDraw.Draw(sheet), (6, 6), title, fg=(255, 255, 255), bg=(20, 20, 20))
    return sheet


def sheet_of_images(base, rels, title, cols=8, side=160):
    rows_n = max(1, (len(rels) + cols - 1) // cols)
    sheet = new_sheet(cols * (side + 4) + 4, rows_n * (side + 4) + 4, title)
    draw = ImageDraw.Draw(sheet)
    for k, rel in enumerate(rels):
        x, y = 4 + (k % cols) * (side + 4), 30 + (k // cols) * (side + 4)
        sheet.paste(load_thumb(base, rel, side), (x, y))
        put_label(draw, (x + 3, y + 2), k)
    return sheet


def pair_cell(base, rel_a, rel_b, caption, side=128):
    cell = Image.new("RGB", (2 * side + 4, side + 34), (20, 20, 20))
    cell.paste(load_thumb(base, rel_a, side), (0, 0))
    cell.paste(load_thumb(base, rel_b, side), (side + 4, 0))
    draw = ImageDraw.Draw(cell)
    for k, line in enumerate(caption.split("\n")[:2]):
        put_label(draw, (2, side + 2 + 16 * k), line, fg=(255, 255, 255), bg=(20, 20, 20))
    return cell


def sheet_of_pairs(base, items, title, cols=4, side=128):
    """items: [(rel_a, rel_b, caption)]"""
    cw, ch = 2 * side + 4, side + 34
    rows_n = max(1, (len(items) + cols - 1) // cols)
    sheet = new_sheet(cols * (cw + 10) + 6, rows_n * (ch + 8) + 4, title)
    draw = ImageDraw.Draw(sheet)
    for k, (a, b, cap) in enumerate(items):
        x, y = 6 + (k % cols) * (cw + 10), 30 + (k // cols) * (ch + 8)
        sheet.paste(pair_cell(base, a, b, cap, side), (x, y))
        put_label(draw, (x + 3, y + 2), k)
    return sheet


def short(row):
    return f"{row['split']}/{row['cls']}" if row["split"] != "all" else row["cls"]


def make_sheets(base, rows, ok_idx, pairs, clusters, args, out_dir):
    sheets_dir = out_dir / "sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    index_rows = []
    pa, pb, pd, pg = pairs

    def save(img, name):
        img.save(sheets_dir / name, quality=82, optimize=True)

    # 1) случайные уникальные (по MD5) фото каждого класса
    by_cls = defaultdict(dict)
    for i in ok_idx:
        by_cls[rows[i]["cls"]].setdefault(rows[i]["md5"], rows[i]["relpath"])
    for cls in sorted(by_cls):
        rels = sorted(by_cls[cls].values())
        pick = sorted(rng.sample(rels, min(args.sheet_n, len(rels))))
        name = f"class_{re.sub(r'[^A-Za-z0-9_-]+', '_', cls)}.jpg"
        save(sheet_of_images(base, pick, f"{cls}: {len(pick)} random unique images of {len(rels)}"), name)
        index_rows += [{"sheet": name, "cell": k, "relpath_a": r} for k, r in enumerate(pick)]

    if len(pd) == 0:
        return index_rows
    not_same_md5 = np.array([rows[ok_idx[a]]["md5"] != rows[ok_idx[b]]["md5"] for a, b in zip(pa, pb)])

    def pair_items(sel, name, title, caption_fn):
        items = []
        for k, p in enumerate(sel):
            ra, rb = rows[ok_idx[pa[p]]], rows[ok_idx[pb[p]]]
            items.append((ra["relpath"], rb["relpath"], caption_fn(p, ra, rb)))
            index_rows.append({"sheet": name, "cell": k, "relpath_a": ra["relpath"],
                               "relpath_b": rb["relpath"], "dist": int(pd[p]), "transform": TRANSFORMS[pg[p]]})
        if items:
            save(sheet_of_pairs(base, [(a, b, c) for a, b, c in items], title), name)

    cap = lambda p, ra, rb: f"d={pd[p]} {TRANSFORMS[pg[p]]}\n{short(ra)} | {short(rb)}"
    # 2) калибровка порога: по 4 случайные пары на каждом расстоянии (побайтовые копии не берём)
    sel = []
    for d in range(0, args.pair_max_dist + 1):
        cand = np.nonzero((pd == d) & not_same_md5)[0].tolist()
        sel += sorted(rng.sample(cand, min(4, len(cand))))
    pair_items(sel, "pairs_by_distance.jpg",
               f"pHash distance calibration: 4 random pairs per distance 0..{args.pair_max_dist} (byte-identical skipped)",
               cap)
    # 3) пары между сплитами одного корня (возможная утечка train/test)
    split_a = np.array([rows[ok_idx[a]]["split"] for a in pa])
    split_b = np.array([rows[ok_idx[b]]["split"] for b in pb])
    cand = np.nonzero((split_a != split_b) & (pd <= args.near_thresh))[0].tolist()
    sel = sorted(rng.sample(cand, min(32, len(cand))), key=lambda p: (pd[p], p))
    pair_items(sel, "pairs_cross_split.jpg",
               f"Cross-split pairs with d<={args.near_thresh}: {len(sel)} random of {len(cand)}",
               cap)
    # 4) пары из разных классов (шум в разметке или ложные срабатывания хеша)
    cls_a = np.array([rows[ok_idx[a]]["cls"] for a in pa])
    cls_b = np.array([rows[ok_idx[b]]["cls"] for b in pb])
    cand = np.nonzero(cls_a != cls_b)[0].tolist()[:32]          # уже отсортированы по расстоянию
    pair_items(cand, "pairs_cross_class.jpg",
               f"Closest cross-class pairs (d<={args.pair_max_dist}): {len(cand)} shown",
               cap)
    # 5) крупнейшие кластеры почти-дубликатов
    members = defaultdict(list)
    for pos, c in enumerate(clusters):
        members[c].append(pos)
    big, seen = [], set()
    for c in sorted(members, key=lambda c: (-len(members[c]), c)):
        key = frozenset(rows[ok_idx[pos]]["md5"] for pos in members[c])
        if len(members[c]) >= 3 and key not in seen:   # тот же кластер в другом корне не повторяем
            big.append(c)
            seen.add(key)
        if len(big) == 6:
            break
    if big:
        side, cols = 112, 10
        sheet = new_sheet(cols * (side + 4) + 4, len(big) * (side + 22) + 4,
                          f"Largest near-duplicate clusters (d<={args.near_thresh}), up to {cols} members each")
        draw = ImageDraw.Draw(sheet)
        for r_i, c in enumerate(big):
            y = 30 + r_i * (side + 22)
            mem = members[c]
            root = rows[ok_idx[mem[0]]]["root"]
            comp = Counter(short(rows[ok_idx[pos]]) for pos in mem)
            put_label(draw, (6, y), f"cluster {c} ({root}): {len(mem)} images: "
                      + ", ".join(f"{k} x{v}" for k, v in comp.most_common(4)), fg=(255, 255, 255), bg=(20, 20, 20))
            for k, pos in enumerate(mem[:cols]):
                row = rows[ok_idx[pos]]
                x = 4 + k * (side + 4)
                sheet.paste(load_thumb(base, row["relpath"], side), (x, y + 18))
                put_label(draw, (x + 2, y + 20), short(row)[:16])
                index_rows.append({"sheet": "largest_clusters.jpg", "cell": f"{c}:{k}", "relpath_a": row["relpath"]})
        save(sheet, "largest_clusters.jpg")
    return index_rows


# --------------------------------------------------------------------------------------
# Сводка
# --------------------------------------------------------------------------------------
def pct(a, b):
    return f"{100.0 * a / b:.1f}%" if b else "-"


def quantiles(values, qs=(0.0, 0.5, 1.0)):
    v = sorted(x for x in values if x not in ("", None) and x == x)
    if not v:
        return ["-"] * len(qs)
    return [v[min(len(v) - 1, int(round(q * (len(v) - 1))))] for q in qs]


def build_summary(rows, ok_idx, pairs, args, base_name, elapsed):
    L = []
    S = {}
    ok_rows = [rows[i] for i in ok_idx]
    L.append("=== Аудит датасета ===")
    L.append(f"База: {base_name} | порог почти-дубликата pHash <= {args.near_thresh} (из 64 бит), "
             f"пары собраны до {args.pair_max_dist} | время {elapsed:.0f} c")
    n_err = len(rows) - len(ok_rows)
    L.append(f"Файлов: {len(rows)} | открылись как изображения: {len(ok_rows)} | не открылись: {n_err}")
    S["files"], S["ok"], S["errors"] = len(rows), len(ok_rows), n_err

    # --- количество по корню/сплиту/классу
    groups = sorted({(r["root"], r["split"]) for r in ok_rows})
    classes = sorted({r["cls"] for r in ok_rows}, key=lambda c: (c not in CLASSES, c))
    cnt = Counter((r["root"], r["split"], r["cls"]) for r in ok_rows)
    L.append("\n--- Изображений по классам (столбцы: корень/сплит) ---")
    heads = [f"{g[0]}/{g[1]}" for g in groups]
    width = max([10] + [len(h) for h in heads]) + 2
    L.append("class".ljust(12) + "".join(h.rjust(width) for h in heads))
    for c in classes:
        L.append(c[:11].ljust(12) + "".join(str(cnt[(g[0], g[1], c)]).rjust(width) for g in groups))
    L.append("ИТОГО".ljust(12) + "".join(str(sum(cnt[(g[0], g[1], c)] for c in classes)).rjust(width) for g in groups))
    S["counts"] = {f"{k[0]}/{k[1]}/{k[2]}": v for k, v in sorted(cnt.items())}
    unknown_cls = sorted({r["cls"] for r in ok_rows if not r["cls_known"]})
    if unknown_cls:
        L.append(f"! Папки, не совпавшие с 6 классами ТЗ: {unknown_cls[:10]}")

    # --- уникальные изображения (по MD5) для статистик, чтобы копии не считались дважды
    uniq = {}
    for r in ok_rows:
        uniq.setdefault(r["md5"], r)
    U = list(uniq.values())
    L.append(f"\nУникальных по содержимому (MD5) изображений: {len(U)}")
    S["unique_md5"] = len(U)

    L.append("\n--- Форматы / режимы / файлы (уникальные) ---")
    L.append("format: " + ", ".join(f"{k}={v}" for k, v in Counter(r["format"] for r in U).most_common()))
    L.append("mode:   " + ", ".join(f"{k}={v}" for k, v in Counter(r["mode"] for r in U).most_common()))
    L.append("ext:    " + ", ".join(f"{k}={v}" for k, v in Counter(r["ext"] for r in rows).most_common(12)))
    L.append(f"с прозрачностью: {sum(int(r['has_alpha']) for r in U)} | многокадровых: "
             f"{sum(1 for r in U if int(r['n_frames']) > 1)} | progressive JPEG: {sum(int(r['progressive']) for r in U)}")

    L.append("\n--- Размеры по классам (уникальные; ш x в с учётом EXIF-поворота) ---")
    L.append("class        n     мин.сторона мин/мед/макс   макс.сторона мед   <224px   кв.(0.9-1.1)  топ размеры")
    by_cls = defaultdict(list)
    for r in U:
        by_cls[r["cls"]].append(r)
    for c in classes:
        rs = by_cls.get(c, [])
        if not rs:
            continue
        mins = [min(r["oriented_w"], r["oriented_h"]) for r in rs]
        maxs = [max(r["oriented_w"], r["oriented_h"]) for r in rs]
        sq = sum(1 for r in rs if 0.9 <= r["oriented_w"] / r["oriented_h"] <= 1.1)
        small = sum(1 for m in mins if m < 224)
        top = Counter(f"{r['oriented_w']}x{r['oriented_h']}" for r in rs).most_common(3)
        mn = quantiles(mins)
        L.append(f"{c[:11].ljust(12)}{len(rs):<6}{mn[0]:>5}/{mn[1]}/{mn[2]:<14}{quantiles(maxs)[1]:>8}"
                 f"{pct(small, len(rs)):>12}{pct(sq, len(rs)):>12}   " + ", ".join(f"{k}({v})" for k, v in top))
    L.append("Самые частые размеры всего: " + ", ".join(
        f"{k}({v})" for k, v in Counter(f"{r['oriented_w']}x{r['oriented_h']}" for r in U).most_common(10)))

    L.append("\n--- Как получены (уникальные): EXIF, качество JPEG ---")
    for c in classes:
        rs = by_cls.get(c, [])
        if not rs:
            continue
        exif_cam = sum(1 for r in rs if r.get("exif_make") or r.get("exif_model"))
        rot = sum(1 for r in rs if int(r.get("exif_orientation") or 1) != 1)
        q = quantiles([r["jpeg_q_est"] for r in rs if r["jpeg_q_est"] != ""], (0.1, 0.5, 0.9))
        L.append(f"{c[:11].ljust(12)} EXIF камеры: {pct(exif_cam, len(rs)):>6} | EXIF-поворот≠1: {rot:<4} "
                 f"| качество JPEG p10/p50/p90: {q[0]}/{q[1]}/{q[2]}")
    cams = Counter(f"{r.get('exif_make', '')} {r.get('exif_model', '')}".strip() for r in U
                   if r.get("exif_make") or r.get("exif_model"))
    if cams:
        L.append("Камеры: " + ", ".join(f"{k}({v})" for k, v in cams.most_common(8)))

    L.append("\n--- Шаблоны имён файлов (топ-4 на корень/сплит/класс) ---")
    tmpl = defaultdict(Counter)
    example = {}
    for r in rows:
        key = (r["root"], r["split"], r["cls"])
        tmpl[key][r["name_template"]] += 1
        example.setdefault((key, r["name_template"]), r["filename"])
    for key in sorted(tmpl):
        tops = tmpl[key].most_common(4)
        L.append(f"{key[0]}/{key[1]}/{key[2]}: " + " | ".join(
            f"'{t}' x{v} (напр. {example[(key, t)][:40]})" for t, v in tops))

    L.append("\n--- Условия съёмки по классам (медианы, уникальные) ---")
    L.append("class        яркость контраст насыщ. край:ярк. край:белый край:тёмный резкость(lap_var)")
    for c in classes:
        rs = by_cls.get(c, [])
        if not rs:
            continue
        med = lambda k: statistics.median(float(r[k]) for r in rs)
        L.append(f"{c[:11].ljust(12)}{med('luma_mean'):8.0f}{med('luma_std'):9.0f}{med('sat_mean'):7.2f}"
                 f"{med('border_luma_mean'):10.0f}{med('border_white_frac'):11.2f}{med('border_dark_frac'):11.2f}"
                 f"{med('lap_var'):12.0f}")

    # --- точные дубликаты
    L.append("\n--- Точные дубликаты (одинаковый MD5) ---")
    by_md5 = defaultdict(list)
    for r in ok_rows:
        by_md5[r["md5"]].append(r)
    S["exact"] = {}
    for root in sorted({r["root"] for r in ok_rows}):
        grp = defaultdict(list)
        for r in ok_rows:
            if r["root"] == root:
                grp[r["md5"]].append(r)
        multi = [g for g in grp.values() if len(g) > 1]
        extra = sum(len(g) - 1 for g in multi)
        x_cls = sum(1 for g in multi if len({r["cls"] for r in g}) > 1)
        x_split = sum(1 for g in multi if len({r["split"] for r in g}) > 1)
        L.append(f"{root}: групп копий {len(multi)}, лишних файлов {extra}, групп в разных классах {x_cls}, "
                 f"групп в разных сплитах {x_split}")
        S["exact"][root] = {"groups": len(multi), "extra_files": extra, "cross_class_groups": x_cls,
                            "cross_split_groups": x_split}
    roots = sorted({r["root"] for r in ok_rows})
    if len(roots) > 1:
        L.append("Связь корней:")
        for root in roots:
            rs = [r for r in ok_rows if r["root"] == root]
            twin = [r for r in rs if int(r["md5_copies_other_roots"]) > 0]
            mism = sum(1 for r in twin if not any(o["cls"] == r["cls"] for o in by_md5[r["md5"]] if o["root"] != root))
            near = sum(1 for r in rs if r["nn_other_root_dist"] != "" and int(r["nn_other_root_dist"]) <= 2)
            L.append(f"  {root}: {len(rs)} изобр.; побайтовая копия в другом корне: {len(twin)} ({pct(len(twin), len(rs))}), "
                     f"из них без копии того же класса: {mism}; pHash-близнец (d<=2) в другом корне: {near} ({pct(near, len(rs))})")

    # --- почти-дубликаты
    pa, pb, pd, pg = pairs
    L.append(f"\n--- Почти-дубликаты (pHash, отражения/повороты учтены), порог d <= {args.near_thresh} ---")
    S["near"] = {}
    for root in roots:
        sel = [k for k in range(len(pd)) if rows[ok_idx[pa[k]]]["root"] == root and pd[k] <= args.near_thresh]
        cc = sum(1 for k in sel if rows[ok_idx[pa[k]]]["cls"] != rows[ok_idx[pb[k]]]["cls"])
        cs = sum(1 for k in sel if rows[ok_idx[pa[k]]]["split"] != rows[ok_idx[pb[k]]]["split"])
        tr = sum(1 for k in sel if pg[k] != 0)
        rs = [r for r in ok_rows if r["root"] == root]
        in_cl = [r for r in rs if int(r["near_cluster_size"]) > 1]
        ncl = len({r["near_cluster"] for r in in_cl})
        biggest = max([int(r["near_cluster_size"]) for r in rs] or [0])
        L.append(f"{root}: пар {len(sel)} (из разных классов {cc}, из разных сплитов {cs}, через отражение/поворот {tr}); "
                 f"кластеров >=2: {ncl}, в них изображений {len(in_cl)} ({pct(len(in_cl), len(rs))}), крупнейший {biggest}")
        buckets = Counter()
        for r in rs:
            d = int(r["nn_same_root_dist"]) if r["nn_same_root_dist"] != "" else 99
            buckets["0" if d == 0 else "1-4" if d <= 4 else "5-8" if d <= 8 else "9-12" if d <= 12 else ">12"] += 1
        L.append("   расстояние до ближайшего соседа в корне: " + ", ".join(
            f"{b}: {buckets[b]}" for b in ["0", "1-4", "5-8", "9-12", ">12"]))
        S["near"][root] = {"pairs": len(sel), "cross_class": cc, "cross_split": cs, "clusters": ncl,
                           "images_in_clusters": len(in_cl), "largest": biggest}

    # --- утечка между сплитами
    L.append("\n--- Утечка между сплитами (внутри одного корня) ---")
    S["leak"] = {}
    for root in roots:
        sp = sorted({r["split"] for r in ok_rows if r["root"] == root})
        if len(sp) < 2:
            continue
        for s in sp:
            rs = [r for r in ok_rows if r["root"] == root and r["split"] == s]
            exact = sum(1 for r in rs if any(o["root"] == root and o["split"] != s for o in by_md5[r["md5"]]))
            near = sum(1 for r in rs if r["nn_other_split_dist"] != "" and int(r["nn_other_split_dist"]) <= args.near_thresh)
            L.append(f"{root}/{s}: {len(rs)} изобр.; побайтовая копия в другом сплите: {exact} ({pct(exact, len(rs))}); "
                     f"почти-дубликат (d<={args.near_thresh}) в другом сплите: {near} ({pct(near, len(rs))})")
            S["leak"][f"{root}/{s}"] = {"n": len(rs), "exact": exact, "near": near}

    # --- ошибки
    errs = [r for r in rows if not r["ok"]]
    if errs:
        L.append("\n--- Файлы, которые не открылись ---")
        L.append(", ".join(f"{k}: {v}" for k, v in Counter(r["error"].split(":")[0] for r in errs).most_common()))
        for r in errs[:25]:
            L.append(f"  {r['relpath']}  ->  {r['error']}")
    warn = [r for r in rows if r.get("warning")]
    if warn:
        L.append(f"\nПредупреждения Pillow: {len(warn)} файлов, напр.: "
                 + " | ".join(f"{r['relpath']}: {r['warning']}" for r in warn[:5]))
    return "\n".join(L) + "\n", S


# --------------------------------------------------------------------------------------
def descend_wrappers(path):
    """Если в папке ровно одна подпапка (обёртка архива), спускаемся в неё."""
    while True:
        kids = [c for c in path.iterdir() if not c.name.startswith(".") and c.name != "__MACOSX"]
        if (len(kids) == 1 and kids[0].is_dir() and kids[0].name.lower() not in CLASS_ALIASES
                and kids[0].name.lower() not in SPLIT_ALIASES):
            path = kids[0]
        else:
            return path


def parse_args():
    p = argparse.ArgumentParser(description="Аудит датасета изображений (только чтение).")
    p.add_argument("--data", required=True, help="папка с распакованным датасетом")
    p.add_argument("--out", default="audit_out", help="папка для результатов (default: audit_out)")
    p.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1),
                   help="число процессов (1 = без multiprocessing)")
    p.add_argument("--near-thresh", type=int, default=8,
                   help="порог pHash (бит из 64) для кластеров почти-дубликатов, default 8")
    p.add_argument("--pair-max-dist", type=int, default=12,
                   help="до какого расстояния сохранять пары для калибровки, default 12")
    p.add_argument("--max-pairs-csv", type=int, default=50000, help="макс. строк в near_dup_pairs.csv")
    p.add_argument("--sheet-n", type=int, default=48, help="картинок на лист класса")
    p.add_argument("--no-sheets", action="store_true", help="не рисовать контрольные листы")
    return p.parse_args()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    args = parse_args()
    t0 = time.time()
    data_dir = Path(args.data).expanduser().resolve()
    if data_dir.is_file() and data_dir.suffix.lower() in (".zip", ".rar", ".7z", ".tar", ".gz"):
        sys.exit(f"Это архив: {data_dir.name}. Сначала распакуйте его и укажите --data на папку.")
    if not data_dir.is_dir():
        sys.exit(f"Папка не найдена: {data_dir}")
    base = descend_wrappers(data_dir)
    out_dir = Path(args.out).expanduser().resolve()
    if out_dir == base or base in out_dir.parents:
        sys.exit("--out не должен лежать внутри папки датасета")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in base.rglob("*") if p.is_file())
    rels = [p.relative_to(base).as_posix() for p in files]
    print(f"База датасета: {base}\nФайлов: {len(rels)}; процессов: {args.workers}")
    if not rels:
        sys.exit("Файлов не найдено.")

    tasks = [(str(base), r) for r in rels]
    rows = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for k, row in enumerate(ex.map(analyze_file, tasks, chunksize=16), 1):
                rows.append(row)
                if k % 1000 == 0 or k == len(tasks):
                    print(f"  {k}/{len(tasks)} файлов, {time.time() - t0:.0f} c", flush=True)
    else:
        for k, task in enumerate(tasks, 1):
            rows.append(analyze_file(task))
            if k % 1000 == 0 or k == len(tasks):
                print(f"  {k}/{len(tasks)} файлов, {time.time() - t0:.0f} c", flush=True)

    for r in rows:
        r["root"], r["split"], r["cls"], r["cls_known"] = parse_relpath(r["relpath"])
        r["filename"] = PurePosixPath(r["relpath"]).name
        r["ext"] = PurePosixPath(r["relpath"]).suffix.lower()
        r["name_template"] = name_template(r["filename"])

    # MD5: копии внутри своего корня и в других корнях
    by_md5 = defaultdict(list)
    for r in rows:
        if r.get("bytes"):
            by_md5[r["md5"]].append(r)
    for r in rows:
        same = by_md5.get(r.get("md5"), []) if r.get("bytes") else []
        r["md5_dups_same_root"] = sum(1 for o in same if o is not r and o["root"] == r["root"])
        r["md5_copies_other_roots"] = sum(1 for o in same if o["root"] != r["root"])

    ok_idx = [i for i, r in enumerate(rows) if r["ok"]]
    print(f"Открылось изображений: {len(ok_idx)}. Ищу почти-дубликаты...", flush=True)
    pairs = (np.zeros(0, np.int64),) * 2 + (np.zeros(0, np.int16), np.zeros(0, np.int8))
    clusters = []
    if ok_idx:
        nn, pairs, clusters, cluster_sizes = find_near_duplicates(rows, ok_idx, args.pair_max_dist, args.near_thresh)
        for pos, i in enumerate(ok_idx):
            r = rows[i]
            for key in ("same_root", "other_split", "other_root"):
                d, j = nn[key][0][pos], nn[key][1][pos]
                r[f"nn_{key}_dist"] = int(d) if j >= 0 else ""
                r[f"nn_{key}_relpath"] = rows[ok_idx[j]]["relpath"] if j >= 0 else ""
            r["near_cluster"], r["near_cluster_size"] = clusters[pos], cluster_sizes[pos]

    with open(out_dir / "inventory.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLUMNS})

    pa, pb, pd, pg = pairs
    with open(out_dir / "near_dup_pairs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["root", "dist", "transform", "a_relpath", "b_relpath", "a_split", "b_split",
                    "a_cls", "b_cls", "same_md5"])
        for k in range(min(len(pd), args.max_pairs_csv)):
            ra, rb = rows[ok_idx[pa[k]]], rows[ok_idx[pb[k]]]
            w.writerow([ra["root"], int(pd[k]), TRANSFORMS[pg[k]], ra["relpath"], rb["relpath"],
                        ra["split"], rb["split"], ra["cls"], rb["cls"], int(ra["md5"] == rb["md5"])])
    if len(pd) > args.max_pairs_csv:
        print(f"! Пар {len(pd)}, в CSV записаны {args.max_pairs_csv} самых близких")

    if not args.no_sheets and ok_idx:
        print("Рисую контрольные листы...", flush=True)
        index_rows = make_sheets(base, rows, ok_idx, pairs, clusters, args, out_dir)
        with open(out_dir / "sheets_index.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["sheet", "cell", "relpath_a", "relpath_b", "dist", "transform"])
            w.writeheader()
            w.writerows(index_rows)

    text, summary = build_summary(rows, ok_idx, pairs, args, base.name, time.time() - t0)
    summary["args"] = {k: v for k, v in vars(args).items() if k not in ("data", "out")}
    summary["versions"] = {"python": sys.version.split()[0], "numpy": np.__version__,
                           "pillow": getattr(Image, "__version__", "?")}
    (out_dir / "summary.txt").write_text(text, encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    archive = shutil.make_archive(str(out_dir), "zip", root_dir=out_dir.parent, base_dir=out_dir.name)
    print("\n" + text)
    print(f"Готово за {time.time() - t0:.0f} c. Архив для отправки: {archive} "
          f"({os.path.getsize(archive) / 1e6:.1f} МБ)")


if __name__ == "__main__":
    main()