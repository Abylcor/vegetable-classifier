"""Сверяет ответы predict.py с папками, по которым вы разложили свои фото.

Разложите фото по папкам с именами классов: Capsicum, Carrot, Cucumber, Potato, Raddish, Squash.
Фото в любой другой папке (например, other/) считаются «не из шести классов»: для них скрипт
просто покажет, что ответила модель и насколько она уверена.

    python predict.py --input my_photos --output my_preds.csv
    python check_my_photos.py my_preds.csv
"""
import csv
import sys
from collections import Counter

CLASSES = ["Capsicum", "Carrot", "Cucumber", "Potato", "Raddish", "Squash"]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "my_preds.csv"
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    known, other = [], []
    for r in rows:
        folder = r["filename"].split("/")[0] if "/" in r["filename"] else ""
        (known if folder in CLASSES else other).append((folder, r))

    if known:
        right = sum(r["label"] == true for true, r in known)
        print(f"Фото из шести классов: {len(known)}, верно: {right} ({100 * right / len(known):.0f}%)\n")

        print("По классам (верно / всего):")
        for c in CLASSES:
            items = [r for true, r in known if true == c]
            if items:
                print(f"  {c:9s} {sum(r['label'] == c for r in items)} / {len(items)}")

        answers = CLASSES + ["unknown"]
        conf = Counter((true, r["label"]) for true, r in known)
        print("\nМатрица ошибок: строка — папка, столбец — ответ модели")
        print(" " * 10 + "".join(f"{a[:8]:>9s}" for a in answers))
        for c in CLASSES:
            if any(true == c for true, _ in known):
                print(f"{c:10s}" + "".join(f"{conf[(c, a)]:9d}" for a in answers))

        errors = sorted(((true, r) for true, r in known if r["label"] != true),
                        key=lambda x: -float(x[1]["confidence"]))
        if errors:
            print("\nОшибки, от самых уверенных:")
            for true, r in errors:
                print(f"  {r['filename']}: модель сказала {r['label']} ({float(r['confidence']):.2f})")

    if other:
        print(f"\nФото не из шести классов: {len(other)}. Что ответила модель:")
        for _, r in other:
            print(f"  {r['filename']}: {r['label']} ({float(r['confidence']):.2f})")

    if not rows:
        print("CSV пустой: в папке не нашлось фото.")


if __name__ == "__main__":
    main()
