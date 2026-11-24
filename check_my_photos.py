"""Сверяет ответы predict.py с папками классов, по которым разложены свои фото.

python check_my_photos.py my_preds.csv
"""
import csv
import sys
from collections import Counter

CLASSES = ["Capsicum", "Carrot", "Cucumber", "Potato", "Raddish", "Squash"]


def pct(a, b):
    return f"{a} из {b} ({100 * a / b:.0f}%)" if b else "0"


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    path = sys.argv[1] if len(sys.argv) > 1 else "my_preds.csv"
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("CSV пустой: в папке не нашлось фото.")
        return

    known, other = [], []
    for r in rows:
        folder = r["filename"].split("/")[0] if "/" in r["filename"] else ""
        (known if folder in CLASSES else other).append((folder, r))

    if known:
        right = [r for true, r in known if r["label"] == true]
        unknown = [(true, r) for true, r in known if r["label"] == "unknown"]
        wrong = [(true, r) for true, r in known if r["label"] not in (true, "unknown")]
        print(f"Фото из шести классов: {len(known)}. Верно: {len(right)}, ошибок: {len(wrong)}, "
              f"unknown: {len(unknown)}")
        print(f"Верно среди всех фото: {pct(len(right), len(known))}")
        print(f"Верно среди фото, на которые модель ответила: {pct(len(right), len(right) + len(wrong))}\n")

        print("По классам (верно / ошибок / unknown / всего):")
        for c in CLASSES:
            items = [r for true, r in known if true == c]
            if items:
                ok = sum(r["label"] == c for r in items)
                unk = sum(r["label"] == "unknown" for r in items)
                print(f"  {c:9s} {ok} / {len(items) - ok - unk} / {unk} / {len(items)}")

        answers = CLASSES + ["unknown"]
        conf = Counter((true, r["label"]) for true, r in known)
        print("\nМатрица ошибок: строка - папка, столбец - ответ модели")
        print(" " * 10 + "".join(f"{a[:8]:>9s}" for a in answers))
        for c in CLASSES:
            if any(true == c for true, _ in known):
                print(f"{c:10s}" + "".join(f"{conf[(c, a)]:9d}" for a in answers))

        if wrong:
            print("\nОшибки, от самых уверенных:")
            for true, r in sorted(wrong, key=lambda x: -float(x[1]["confidence"])):
                print(f"  {r['filename']}: модель сказала {r['label']} ({float(r['confidence']):.2f})")
        if unknown:
            print("\nunknown (уверенность ниже порога или файл не прочитан):")
            for true, r in sorted(unknown, key=lambda x: -float(x[1]["confidence"])):
                print(f"  {r['filename']}: уверенность {float(r['confidence']):.2f}")

    if other:
        caught = sum(r["label"] == "unknown" for _, r in other)
        print(f"\nФото не из шести классов: {len(other)}, unknown у {pct(caught, len(other))}. Ответы модели:")
        for _, r in other:
            print(f"  {r['filename']}: {r['label']} ({float(r['confidence']):.2f})")


if __name__ == "__main__":
    main()
