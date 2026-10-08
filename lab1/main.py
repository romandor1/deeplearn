"""Точка входу ЛР1.

  uv run python main.py          # коректна реалізація (за замовчуванням)
  uv run python main.py --bug    # дослід 4: навмисна помилка (без ділення на N)
  uv run python main.py --all    # обидва режими + запис results/*.md
"""
from __future__ import annotations

import argparse
import io
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from gradient_checks import EPS, TOL_NUM, TOL_TORCH, compare_with_torch, eps_scan, numeric_check
from data import load_split_standardize
from mlp_numpy import PARAM_NAMES, cross_entropy, forward, backward, init_params

np.set_printoptions(precision=6, linewidth=120)
RESULTS = Path(__file__).parent / "results"
ok_s = lambda ok: "так" if ok else "НІ"


def run(buggy: bool) -> None:
    X, y, X_test, y_test = load_split_standardize()
    params = init_params()
    assert all(v.dtype == np.float64 for v in params.values())
    print(f"Режим: {'НАВМИСНА ПОМИЛКА (dZ2 = P - Y, без /N)' if buggy else 'коректна реалізація (dZ2 = (P - Y)/N)'}")
    print(f"Навчальна вибірка: {X.shape}, класи {np.bincount(y).tolist()}; "
          f"тест: {X_test.shape}, класи {np.bincount(y_test).tolist()}")
    print("Розмірності параметрів:", {k: v.shape for k, v in params.items()})

    loss, cache = forward(params, X, y)
    print("Розмірності проміжних значень:",
          {k: v.shape for k, v in cache.items()})
    grads = backward(params, cache, y, buggy=buggy)
    print("Розмірності градієнтів:", {k: v.shape for k, v in grads.items()})

    # --- sanity checks
    print(f"\nSanity checks: кількість параметрів = {sum(v.size for v in params.values())}; "
          f"початкова втрата = {loss:.6f} (ln 3 = {np.log(3):.6f}); "
          f"точність на train з початковими вагами = {(cache['Z2'].argmax(1) == y).mean():.3f}")
    print(f"Стандартизація: max|mean(train)| = {np.abs(X.mean(axis=0)).max():.2e}, "
          f"std(train) = {X.std(axis=0, ddof=0)}, max|mean(test)| = {np.abs(X_test.mean(axis=0)).max():.3f}")

    # --- стабільність функції втрат: стабільна проти наївної CE
    print("\n## Стабільність log-softmax\n")
    print("| Логіти | Клас | Стабільна CE | Наївна CE -log(exp(z_c) / sum(exp(z))) |")
    print("|---|---|---|---|")
    for z, c in (([1000.0, 0.0, -1000.0], 0), ([1000.0, 0.0, -1000.0], 2), ([0.0, -1000.0, 0.0], 1)):
        Zb = np.array([z])
        stable = cross_entropy(Zb, np.array([c]))[0] + 0.0
        with np.errstate(all="ignore"):
            naive = -np.log(np.exp(Zb[0, c]) / np.exp(Zb).sum())
        print(f"| {z} | {c} | {stable:.4f} | {naive} |")

    # --- п.2: звірка з PyTorch
    loss_t, grads_t, rows = compare_with_torch(params, X, y, loss, grads)
    print(f"\n## Звірка з PyTorch (критерій |a-b| <= {TOL_TORCH:g}, усі значення скінченні)\n")
    print(f"Втрата NumPy  = {loss:.15f}")
    print(f"Втрата PyTorch = {loss_t:.15f}\n")
    print("| Величина | Макс. абсолютна різниця NumPy / PyTorch | Перевірку пройдено |")
    print("|---|---|---|")
    for name, d, ok in rows:
        print(f"| {name} | {d:.3e} | {ok_s(ok)} |")

    # --- п.3: чисельне диференціювання
    nrows = numeric_check(params, X, y, grads)
    print(f"\n## Чисельна перевірка (eps = {EPS:g}, критерій |g_num - g_manual| <= {TOL_NUM:g})\n")
    print("| Параметр | Градієнт backward() | Чисельна похідна | Абсолютна різниця | Перевірку пройдено |")
    print("|---|---|---|---|---|")
    for label, g_man, g_num, d, ok, _ in nrows:
        print(f"| {label} | {g_man:.12e} | {g_num:.12e} | {d:.3e} | {ok_s(ok)} |")
    print(f"\nМаски ReLU при θ-ε, θ, θ+ε збігаються (збурення не перетинає злам): "
          f"{ {r[0]: ok_s(r[5]) for r in nrows} }")

    if not buggy:
        print("\n## Похибка центральної різниці залежно від eps (параметр W1[0, 0])\n")
        print("| eps | |g_num - g_manual| float64 | |g_num - g_manual| float32 |")
        print("|---|---|---|")
        r64 = eps_scan(params, X, y, grads)
        r32 = eps_scan(params, X, y, grads, dtype=np.float32)
        fmt = lambda d: "крок = 0 після округлення" if d is None else f"{d:.3e}"
        for (e, d64), (_, d32) in zip(r64, r32):
            print(f"| {e:.0e} | {fmt(d64)} | {fmt(d32)} |")
        best = lambda rs: min((r for r in rs if r[1] is not None), key=lambda r: r[1])
        print(f"\nМінімум float64: {best(r64)[1]:.3e} при eps = {best(r64)[0]:.0e}; "
              f"мінімум float32: {best(r32)[1]:.3e} при eps = {best(r32)[0]:.0e}")

    # --- додатково: співвідношення manual / torch (корисно для досліду 4)
    print("\n## Співвідношення manual / PyTorch для ненульових елементів градієнта\n")
    print("| Градієнт | min | max | к-сть нульових елементів (manual / torch) |")
    print("|---|---|---|---|")
    for name in PARAM_NAMES:
        a, b = grads[name], grads_t[name]
        nz = np.abs(b) > 1e-14
        r = a[nz] / b[nz]
        print(f"| {name} | {r.min():.9f} | {r.max():.9f} | {int((~(np.abs(a) > 1e-14)).sum())} / {int((~nz).sum())} |")

    all_ok = all(r[2] for r in rows) and all(r[4] for r in nrows)
    if not buggy:
        print(f"\nПІДСУМОК: усі перевірки пройдено = {all_ok}")
    else:
        print(f"\nПІДСУМОК: помилку виявлено = {not all_ok} (очікуваний результат досліду)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bug", action="store_true", help="дослід із навмисною помилкою")
    ap.add_argument("--all", action="store_true", help="обидва режими, запис у results/")
    args = ap.parse_args()
    if args.all:
        RESULTS.mkdir(exist_ok=True)
        for buggy, fname in ((False, "correct.md"), (True, "bug.md")):
            buf = io.StringIO()
            with redirect_stdout(buf):
                run(buggy)
            text = buf.getvalue()
            (RESULTS / fname).write_text(text, encoding="utf-8")
            print(text)
            print("=" * 100)
    else:
        run(args.bug)


if __name__ == "__main__":
    main()
