"""Подбор порога unknown на out-of-fold ответах: минимум цены error_cost * ошибки + unknown.

python choose_threshold.py [--error-cost 5]
"""
import argparse
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description="Подбор порога unknown по out-of-fold ответам.")
    p.add_argument("--oof", default=str(HERE / "results" / "oof_preds.csv"),
                   help="out-of-fold ответы (колонки label, pred, confidence)")
    p.add_argument("--splits", default=str(HERE / "splits.csv"), help="для проверки по фолдам (колонка fold)")
    p.add_argument("--error-cost", type=float, default=5.0, help="цена ошибки относительно unknown")
    p.add_argument("--step", type=float, default=0.05, help="шаг сетки порогов")
    p.add_argument("--table", default=str(HERE / "results" / "threshold_table.csv"))
    return p.parse_args()


def stats(rows, t, error_cost):
    n = len(rows)
    answered = [r for r in rows if r["confidence"] >= t]
    errors = sum(r["label"] != r["pred"] for r in answered)
    unknown = n - len(answered)
    wrong_total = sum(r["label"] != r["pred"] for r in rows)
    return {
        "threshold": round(t, 2),
        "answered_share": len(answered) / n,
        "accuracy_answered": (len(answered) - errors) / len(answered) if answered else 1.0,
        "errors_left": errors,
        "errors_caught": wrong_total - errors,
        "correct_lost": unknown - (wrong_total - errors),
        "unknown": unknown,
        "cost": error_cost * errors + unknown,
    }


def best_threshold(rows, error_cost, grid):
    return min(grid, key=lambda t: (stats(rows, t, error_cost)["cost"], t))


def auroc(rows):
    """0,5 - уверенность не отделяет ошибки от верных ответов, 1 - отделяет идеально."""
    ranked = sorted(rows, key=lambda r: r["confidence"])
    pos = sum(r["label"] == r["pred"] for r in rows)
    neg = len(rows) - pos
    rank_sum, i = 0.0, 0
    while i < len(ranked):  # средние ранги для одинаковых значений
        j = i
        while j < len(ranked) and ranked[j]["confidence"] == ranked[i]["confidence"]:
            j += 1
        rank_sum += sum(r["label"] == r["pred"] for r in ranked[i:j]) * (i + j + 1) / 2
        i = j
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def main():
    args = parse_args()
    with open(args.oof, encoding="utf-8") as f:
        rows = [dict(r, confidence=float(r["confidence"])) for r in csv.DictReader(f)]
    grid = [round(i * args.step, 4) for i in range(int(round(0.95 / args.step)) + 1)]
    table = [stats(rows, t, args.error_cost) for t in grid]
    best = best_threshold(rows, args.error_cost, grid)

    n, wrong = len(rows), sum(r["label"] != r["pred"] for r in rows)
    print(f"Out-of-fold ответов: {n}, ошибок без порога: {wrong} (accuracy {1 - wrong / n:.3f})")
    print(f"Насколько уверенность отделяет верные ответы от ошибок (AUROC): {auroc(rows):.3f}\n")
    print(f"{'порог':>6} {'отвечает':>9} {'верно среди ответов':>20} {'ошибок осталось':>16} "
          f"{'верных стало unknown':>21} {'цена':>7}")
    for s in table:
        if 0.25 <= s["threshold"] <= 0.75:
            mark = "  <- выбран" if s["threshold"] == best else ""
            print(f"{s['threshold']:>6.2f} {s['answered_share']:>9.1%} {s['accuracy_answered']:>20.1%} "
                  f"{s['errors_left']:>16} {s['correct_lost']:>21} {s['cost']:>7.0f}{mark}")

    s = stats(rows, best, args.error_cost)
    print(f"\nПорог {best:.2f} (ошибка дороже unknown в {args.error_cost:g} раз): модель отвечает "
          f"на {s['answered_share']:.1%} фото и права в {s['accuracy_answered']:.1%} из них; "
          f"ловит {s['errors_caught']} из {wrong} ошибок, теряет {s['correct_lost']} верных ответов.")

    print("\nЕсли ошибка дороже unknown в другое число раз:")
    for k in (2, 3, 5, 10):
        print(f"  в {k:>2} раз -> порог {best_threshold(rows, k, grid):.2f}")

    if Path(args.splits).is_file():  # тот же выбор отдельно на каждом фолде
        with open(args.splits, encoding="utf-8") as f:
            fold = {r["path"]: r["fold"] for r in csv.DictReader(f)}
        for k in sorted({fold.get(r["path"]) for r in rows} - {None}):
            part = [r for r in rows if fold.get(r["path"]) == k]
            print(f"Только фолд {k} ({len(part)} фото): порог {best_threshold(part, args.error_cost, grid):.2f}")

    Path(args.table).parent.mkdir(parents=True, exist_ok=True)
    with open(args.table, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        for t in table:
            w.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in t.items()})
    print(f"\nВся таблица: {args.table}")


if __name__ == "__main__":
    main()
