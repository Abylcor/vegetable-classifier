"""Можно ли доверять авторскому разбиению final_dataset/ (train / val / test)?

Скрипт не читает картинки: он сравнивает «отпечатки» файлов из inventory.csv,
который строит audit_dataset.py:
  * md5    - отпечаток байтов: совпадает только у одинаковых файлов;
  * pHash  - 64-битный отпечаток картинки; nn_other_split_dist - расстояние (число
             несовпавших бит, с учётом 8 поворотов и отражений) до ближайшей картинки
             из другой части того же разбиения. До 8 бит - почти-дубликат.
Источник снимка берётся из имени файла: 20251113_210729_frame_N.jpg - кадр видео
20251113_210729, 20251113_210958.jpg - отдельное фото.

Запуск:
    python audit_dataset.py --data путь/к/data_raw --out audit_out
    python check_author_split.py --inventory audit_out/inventory.csv
"""
import argparse
import re
import sys

import pandas as pd

NAME = re.compile(r"^(\d{8}_\d{6})(?:_frame_\d+)?\.\w+$")


def pct(a, b):
    return f"{a} из {b} ({100 * a / b:.0f}%)" if b else f"{a} из 0"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--inventory", default="audit_out/inventory.csv", help="inventory.csv от audit_dataset.py")
    ap.add_argument("--near-thresh", type=int, default=8, help="порог почти-дубликата по pHash, бит из 64")
    args = ap.parse_args()

    inv = pd.read_csv(args.inventory, dtype={"md5": str, "phash": str})
    inv["source"] = inv["filename"].str.extract(NAME, expand=False)
    D = inv[inv["root"] == "Dataset"]
    F = inv[inv["root"] == "final_dataset"]
    if D.empty or F.empty:
        sys.exit("В inventory.csv нужны обе папки: Dataset/ и final_dataset/ (запустите аудит на data_raw).")

    parts = {s: F[F["split"] == s] for s in ("train", "val", "test")}
    md5 = {s: set(p["md5"]) for s, p in parts.items()}
    unique = set(D["md5"])

    print("== Состав")
    print(f"Dataset/: {len(D)} файлов, уникальных изображений {len(unique)}")
    print(f"final_dataset/: {len(F)} файлов: " + ", ".join(f"{s} {len(p)}" for s, p in parts.items()))
    print(f"Файлов final_dataset/, которые есть в Dataset/ байт в байт: {pct(int(F['md5'].isin(unique).sum()), len(F))}")
    print(f"Уникальных изображений в авторском train: {pct(len(unique & md5['train']), len(unique))}")

    t, v = parts["test"], parts["val"]
    print("\n== Test против train и val")
    print(f"Байтовая копия файла из train:        {pct(int(t['md5'].isin(md5['train']).sum()), len(t))}")
    print(f"Байтовая копия из train или val:      {pct(int(t['md5'].isin(md5['train'] | md5['val']).sum()), len(t))}")
    near = t["nn_other_split_dist"].le(args.near_thresh)
    print(f"Почти-дубликат (pHash <= {args.near_thresh}) в train или val: {pct(int(near.sum()), len(t))}")

    print("\n== Val против train")
    print(f"Байтовая копия файла из train:        {pct(int(v['md5'].isin(md5['train']).sum()), len(v))}")

    print("\n== Источники (видео и отдельные фото)")
    src_train = set(parts["train"]["source"].dropna())
    for s in ("test", "val"):
        src = set(parts[s]["source"].dropna())
        print(f"{s}: источников {len(src)}, из них есть и в train: {pct(len(src & src_train), len(src))}")


if __name__ == "__main__":
    main()
