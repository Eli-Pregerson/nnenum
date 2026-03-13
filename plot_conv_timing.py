"""
Plot conv timing data from conv_timing_data.csv.

Produces a scatter plot:
  - X axis: total verification time (seconds)
  - Y axis: percentage of time spent in conv layer propagation (%)
  - Red dots   : runs with BATCHING
  - Blue dots  : runs with DENSE (no batching)
  - Green dots : runs with SPARSE
  - Thin black lines : connect all dots for the same instance

Usage:
  python3 plot_conv_timing.py [--input conv_timing_data.csv] [--output conv_timing.png]
"""

import csv
import os
import argparse
import sys


def load_data(csv_path: str):
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # ── batching columns (always present) ────────────────────────────
            try:
                bat_total = float(row['batching_total_secs'])
                bat_conv  = float(row['batching_conv_secs'])
            except (ValueError, KeyError):
                continue
            if bat_total < 0:
                continue
            bat_pct = 100.0 * bat_conv / max(bat_total, 1e-9)

            # ── dense columns (new schema: 'dense_*'; old schema: 'nobatch_*') ─
            den_total = den_conv = None
            for total_key, conv_key in [('dense_total_secs', 'dense_conv_secs'),
                                        ('nobatch_total_secs', 'nobatch_conv_secs')]:
                try:
                    v_total = float(row[total_key])
                    v_conv  = float(row[conv_key])
                    if v_total >= 0:
                        den_total, den_conv = v_total, v_conv
                        break
                except (ValueError, KeyError):
                    continue
            if den_total is None:
                continue
            den_pct = 100.0 * den_conv / max(den_total, 1e-9)

            # ── sparse columns (optional — absent in old CSV) ─────────────────
            spa_total = spa_conv = None
            try:
                v_total = float(row['sparse_total_secs'])
                v_conv  = float(row['sparse_conv_secs'])
                if v_total >= 0:
                    spa_total, spa_conv = v_total, v_conv
            except (ValueError, KeyError):
                pass
            spa_pct = (100.0 * spa_conv / max(spa_total, 1e-9)) if spa_total is not None else None

            label = (
                os.path.basename(row.get('onnx_path', ''))
                + '\n' + os.path.basename(row.get('vnnlib_path', ''))
            )

            # result strings — tolerate missing keys
            den_res_key = 'dense_result' if 'dense_result' in row else 'nobatch_result'

            rows.append({
                'label':      label,
                'category':   row.get('category', ''),
                'bat_total':  bat_total,
                'bat_pct':    bat_pct,
                'bat_result': row.get('batching_result', ''),
                'den_total':  den_total,
                'den_pct':    den_pct,
                'den_result': row.get(den_res_key, ''),
                'spa_total':  spa_total,
                'spa_pct':    spa_pct,
                'spa_result': row.get('sparse_result', ''),
            })
    return rows


def plot(rows, output_path: str):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — printing table instead.")
        print_table(rows)
        return

    has_sparse = any(r['spa_total'] is not None for r in rows)

    fig, ax = plt.subplots(figsize=(12, 7))

    # ── Connecting lines (draw first so dots appear on top) ───────────────────
    for r in rows:
        pts_x = [r['bat_total'], r['den_total']]
        pts_y = [r['bat_pct'],   r['den_pct']]
        if r['spa_total'] is not None:
            pts_x.append(r['spa_total'])
            pts_y.append(r['spa_pct'])
        ax.plot(pts_x, pts_y, color='black', linewidth=0.6, alpha=0.4, zorder=1)

    # ── Dense (blue, bottom layer) ────────────────────────────────────────────
    den_x = [r['den_total'] for r in rows]
    den_y = [r['den_pct']   for r in rows]
    ax.scatter(den_x, den_y, color='steelblue', s=60, zorder=3,
               label='Dense (no batching)', edgecolors='navy', linewidths=0.5)

    # ── Sparse (green, middle layer) ─────────────────────────────────────────
    if has_sparse:
        spa_rows = [r for r in rows if r['spa_total'] is not None]
        spa_x = [r['spa_total'] for r in spa_rows]
        spa_y = [r['spa_pct']   for r in spa_rows]
        ax.scatter(spa_x, spa_y, color='mediumseagreen', s=60, zorder=4,
                   label='Sparse (Toeplitz W)', edgecolors='darkgreen', linewidths=0.5)

    # ── Batching (red, top layer) ─────────────────────────────────────────────
    bat_x = [r['bat_total'] for r in rows]
    bat_y = [r['bat_pct']   for r in rows]
    ax.scatter(bat_x, bat_y, color='tomato', s=60, zorder=5,
               label='Batching', edgecolors='darkred', linewidths=0.5)

    ax.set_xlabel('Total verification time (s)', fontsize=12)
    ax.set_ylabel('Conv layer time as % of total (%)', fontsize=12)
    method_str = 'red=batching, blue=dense, green=sparse' if has_sparse else 'red=batching, blue=dense'
    ax.set_title('Conv Layer Time vs Total Verification Time\n'
                 f'({method_str}; lines connect same instance)', fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Annotate with category (anchored to batching dot)
    for r in rows:
        cat_short = r['category'].replace('_2023', '').replace('_2022', '')
        ax.annotate(
            cat_short,
            xy=(r['bat_total'], r['bat_pct']),
            xytext=(4, 4), textcoords='offset points',
            fontsize=6, color='darkred', alpha=0.7,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")
    plt.close()


def print_table(rows):
    """Fallback: print a readable table when matplotlib is unavailable."""
    has_sparse = any(r['spa_total'] is not None for r in rows)
    if has_sparse:
        header = (
            f"{'category':<22} {'bat_tot':>8} {'bat_%':>7} "
            f"{'den_tot':>8} {'den_%':>7} "
            f"{'spa_tot':>8} {'spa_%':>7}  "
            f"{'bat_res':<7} {'den_res':<7} {'spa_res':<7}"
        )
    else:
        header = (
            f"{'category':<22} {'bat_tot':>8} {'bat_%':>7} "
            f"{'den_tot':>8} {'den_%':>7}  "
            f"{'bat_res':<7} {'den_res':<7}"
        )
    print(header)
    print('-' * len(header))
    for r in sorted(rows, key=lambda x: -x['bat_total']):
        if has_sparse:
            spa_tot = f"{r['spa_total']:>8.2f}" if r['spa_total'] is not None else f"{'N/A':>8}"
            spa_pct = f"{r['spa_pct']:>6.1f}%" if r['spa_pct'] is not None else f"{'N/A':>7}"
            print(
                f"{r['category']:<22} {r['bat_total']:>8.2f} {r['bat_pct']:>6.1f}% "
                f"{r['den_total']:>8.2f} {r['den_pct']:>6.1f}% "
                f"{spa_tot} {spa_pct}  "
                f"{r['bat_result']:<7} {r['den_result']:<7} {r['spa_result']:<7}"
            )
        else:
            print(
                f"{r['category']:<22} {r['bat_total']:>8.2f} {r['bat_pct']:>6.1f}% "
                f"{r['den_total']:>8.2f} {r['den_pct']:>6.1f}%  "
                f"{r['bat_result']:<7} {r['den_result']:<7}"
            )


def main():
    parser = argparse.ArgumentParser(description='Plot conv timing results')
    parser.add_argument('--input',  default='conv_timing_data.csv',
                        help='Input CSV produced by run_conv_timing.py')
    parser.add_argument('--output', default='conv_timing.png',
                        help='Output PNG file')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        sys.exit(1)

    rows = load_data(args.input)
    if not rows:
        print("No valid rows found in input file.")
        sys.exit(1)

    print(f"Loaded {len(rows)} valid rows from {args.input}.")
    print_table(rows)
    print()
    plot(rows, args.output)


if __name__ == '__main__':
    main()
