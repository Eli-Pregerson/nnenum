"""
Plot conv timing data from conv_timing_data.csv.

Produces a scatter plot:
  - X axis: total verification time (seconds)
  - Y axis: percentage of time spent in conv layer propagation (%)
  - Red dots   : runs WITH batching
  - Blue dots  : runs WITHOUT batching
  - Thin black lines : connect the two dots for the same instance

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
            try:
                bat_total = float(row['batching_total_secs'])
                bat_conv  = float(row['batching_conv_secs'])
                nob_total = float(row['nobatch_total_secs'])
                nob_conv  = float(row['nobatch_conv_secs'])
            except (ValueError, KeyError):
                continue

            # Skip rows that errored out (negative sentinels)
            if bat_total < 0 or nob_total < 0:
                continue

            bat_pct = 100.0 * bat_conv / max(bat_total, 1e-9)
            nob_pct = 100.0 * nob_conv / max(nob_total, 1e-9)

            label = (
                os.path.basename(row.get('onnx_path', ''))
                + '\n' + os.path.basename(row.get('vnnlib_path', ''))
            )

            rows.append({
                'label':     label,
                'category':  row.get('category', ''),
                'bat_total': bat_total,
                'bat_pct':   bat_pct,
                'nob_total': nob_total,
                'nob_pct':   nob_pct,
                'bat_result': row.get('batching_result', ''),
                'nob_result': row.get('nobatch_result', ''),
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

    fig, ax = plt.subplots(figsize=(11, 7))

    for r in rows:
        # Connecting line
        ax.plot(
            [r['bat_total'], r['nob_total']],
            [r['bat_pct'],   r['nob_pct']],
            color='black', linewidth=0.6, alpha=0.5, zorder=1,
        )

    # Plot no-batching first (blue, underneath)
    nob_x = [r['nob_total'] for r in rows]
    nob_y = [r['nob_pct']   for r in rows]
    ax.scatter(nob_x, nob_y, color='steelblue', s=60, zorder=3,
               label='No batching', edgecolors='navy', linewidths=0.5)

    # Plot batching (red, on top)
    bat_x = [r['bat_total'] for r in rows]
    bat_y = [r['bat_pct']   for r in rows]
    ax.scatter(bat_x, bat_y, color='tomato', s=60, zorder=4,
               label='With batching', edgecolors='darkred', linewidths=0.5)

    ax.set_xlabel('Total verification time (s)', fontsize=12)
    ax.set_ylabel('Conv layer time as % of total (%)', fontsize=12)
    ax.set_title('Conv Layer Time vs Total Verification Time\n'
                 '(red = batching enabled, blue = batching disabled, '
                 'lines connect same instance)', fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Annotate with category name (small, offset slightly)
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
    header = (
        f"{'category':<22} {'bat_total':>10} {'bat_conv%':>10} "
        f"{'nob_total':>10} {'nob_conv%':>10}  {'bat_res':<8} {'nob_res':<8}"
    )
    print(header)
    print('-' * len(header))
    for r in sorted(rows, key=lambda x: -x['bat_total']):
        print(
            f"{r['category']:<22} {r['bat_total']:>10.2f} {r['bat_pct']:>9.1f}% "
            f"{r['nob_total']:>10.2f} {r['nob_pct']:>9.1f}%  "
            f"{r['bat_result']:<8} {r['nob_result']:<8}"
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

    print(f"Loaded {len(rows)} valid instance pairs from {args.input}.")
    print_table(rows)
    print()
    plot(rows, args.output)


if __name__ == '__main__':
    main()
