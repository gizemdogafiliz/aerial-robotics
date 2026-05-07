"""
Assignment 04: State Estimation Plots
Parse pom.log files and generate state + variance comparison plots.

Usage:
  python3 plot_04.py                          # pairwise: baseline vs each config
  python3 plot_04.py all                      # all configs overlaid (one figure)
  python3 plot_04.py baseline                 # single config
  python3 plot_04.py baseline gps_mag_x10     # compare specific configs
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import math

LOG_BASE = os.path.expanduser('~/tk3lab-ws/logs/04-state-estimation')

ALL_CONFIGS = ['baseline', 'gps_mag_x10', 'gps_mag_x100', 'imu_only_x10', 'imu_only_x100']

COMPARE_CONFIGS = ['gps_mag_x10', 'gps_mag_x100', 'imu_only_x10', 'imu_only_x100']

LINESTYLES = {
    'baseline':       '-',
    'gps_mag_x10':    '--',
    'gps_mag_x100':   ':',
    'imu_only_x10':   '-.',
    'imu_only_x100':  (0, (3, 1, 1, 1)),
}

COLORS = {'x': 'red', 'y': 'green', 'z': 'blue'}
RAD2DEG = 180.0 / math.pi


def parse_pom_log(filepath):
    lines = []
    header = None
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                continue
            if header is None:
                header = line
                continue
            lines.append(line)

    columns = header.split()
    data = []
    for line in lines:
        parts = line.split()
        if len(parts) == len(columns):
            try:
                data.append([float(x) for x in parts])
            except ValueError:
                continue

    return pd.DataFrame(data, columns=columns)


def load_configs(config_names):
    loaded = {}
    for name in config_names:
        logfile = os.path.join(LOG_BASE, name, 'pom.log')
        if os.path.exists(logfile):
            df = parse_pom_log(logfile)
            df['t'] = df['ts'] - df['ts'].iloc[0]
            loaded[name] = df
            print(f"  {name}: {len(df)} samples, {df['t'].max():.1f}s")
        else:
            print(f"  {name}: no log found, skipping")
    return loaded


def add_dual_legend(fig, ax_empty, configs_data):
    ax_empty.set_visible(True)
    ax_empty.axis('off')

    style_handles = []
    for cfg_name in configs_data:
        ls = LINESTYLES.get(cfg_name, '-')
        h, = ax_empty.plot([], [], color='gray', ls=ls, lw=1.5, label=cfg_name)
        style_handles.append(h)

    color_handles = []
    for axis_label, color in [('x / roll', 'red'), ('y / pitch', 'green'), ('z / yaw', 'blue')]:
        h, = ax_empty.plot([], [], color=color, ls='-', lw=2.5, label=axis_label)
        color_handles.append(h)

    leg1 = ax_empty.legend(handles=style_handles, title='Config',
                           loc='center left', fontsize=8, title_fontsize=9,
                           bbox_to_anchor=(0.0, 0.5))
    ax_empty.add_artist(leg1)
    ax_empty.legend(handles=color_handles, title='Axis',
                    loc='center right', fontsize=8, title_fontsize=9,
                    bbox_to_anchor=(1.0, 0.5))


def plot_states(configs_data, save_dir=None, title_suffix=''):
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(f'State Estimation — States{title_suffix}', fontsize=13)

    state_groups = [
        (0, 0, 'Position [m]',         ['x', 'y', 'z'],    1.0),
        (0, 1, 'Attitude [deg]',       ['roll', 'pitch', 'yaw'], RAD2DEG),
        (1, 0, 'Linear Velocity [m/s]', ['vx', 'vy', 'vz'], 1.0),
        (1, 1, 'Angular Velocity [deg/s]', ['wx', 'wy', 'wz'], RAD2DEG),
        (2, 0, 'Linear Acceleration [m/s²]', ['ax', 'ay', 'az'], 1.0),
    ]

    for row, col, title, cols, scale in state_groups:
        ax = axes[row, col]
        ax.set_title(title)
        for cfg_name, df in configs_data.items():
            ls = LINESTYLES.get(cfg_name, '-')
            lw = 1.2 if cfg_name == 'baseline' else 0.8
            for c, axis_label in zip(cols, ['x', 'y', 'z']):
                color = COLORS[axis_label]
                ax.plot(df['t'], df[c] * scale, color=color, ls=ls,
                        lw=lw, alpha=0.85)
        ax.set_xlabel('t [s]')
        ax.grid(True, alpha=0.3)

    add_dual_legend(fig, axes[2, 1], configs_data)
    fig.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        suffix = '_'.join(configs_data.keys())
        fig.savefig(os.path.join(save_dir, f'states_{suffix}.png'),
                    dpi=150, bbox_inches='tight')
        print(f"Saved: {save_dir}/states_{suffix}.png")

    return fig


def decimate(series, n=50):
    return series.rolling(n, center=True, min_periods=1).mean()


def plot_variances(configs_data, save_dir=None, title_suffix=''):
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(f'State Estimation — Variances (diag P){title_suffix}', fontsize=13)

    var_groups = [
        (0, 0, 'Position Variance [m²]',
         ['sx2', 'sy2', 'sz2'], 1.0, False),
        (0, 1, 'Attitude Variance [deg²]',
         ['sr2', 'sp2', 'sh2'], RAD2DEG**2, False),
        (1, 0, 'Velocity Variance [m²/s²]',
         ['svx2', 'svy2', 'svz2'], 1.0, False),
        (1, 1, 'Angular Vel. Variance [deg²/s²]',
         ['swx2', 'swy2', 'swz2'], RAD2DEG**2, False),
        (2, 0, 'Acceleration Variance [m²/s⁴]',
         ['sax2', 'say2', 'saz2'], 1.0, True),
    ]

    for row, col, title, cols, scale, smooth in var_groups:
        ax = axes[row, col]
        ax.set_title(title)
        for cfg_name, df in configs_data.items():
            ls = LINESTYLES.get(cfg_name, '-')
            lw = 1.2 if cfg_name == 'baseline' else 0.8
            for c, axis_label in zip(cols, ['x', 'y', 'z']):
                color = COLORS[axis_label]
                y = df[c] * scale
                if smooth:
                    y = decimate(y)
                ax.plot(df['t'], y, color=color, ls=ls,
                        lw=lw, alpha=0.85)
        ax.set_xlabel('t [s]')
        ax.grid(True, alpha=0.3)

    add_dual_legend(fig, axes[2, 1], configs_data)
    fig.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        suffix = '_'.join(configs_data.keys())
        fig.savefig(os.path.join(save_dir, f'variances_{suffix}.png'),
                    dpi=150, bbox_inches='tight')
        print(f"Saved: {save_dir}/variances_{suffix}.png")

    return fig


def main():
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots_04')

    if len(sys.argv) > 1 and sys.argv[1] == 'all':
        print("Loading all configs (overlay mode)...")
        configs_data = load_configs(ALL_CONFIGS)
        if not configs_data:
            print("No logs found."); sys.exit(1)
        plot_states(configs_data, save_dir)
        plot_variances(configs_data, save_dir)

    elif len(sys.argv) > 1:
        config_names = sys.argv[1:]
        for c in config_names:
            if c not in ALL_CONFIGS:
                print(f"Unknown config: {c}")
                print(f"Valid: {', '.join(ALL_CONFIGS)}, all")
                sys.exit(1)
        print("Loading configs...")
        configs_data = load_configs(config_names)
        if not configs_data:
            print("No logs found."); sys.exit(1)
        suffix = f' ({" vs ".join(configs_data.keys())})'
        plot_states(configs_data, save_dir, suffix)
        plot_variances(configs_data, save_dir, suffix)

    else:
        print("Loading logs (pairwise: baseline vs each)...")
        all_data = load_configs(ALL_CONFIGS)
        if 'baseline' not in all_data:
            print("baseline log missing."); sys.exit(1)
        for cfg_name in COMPARE_CONFIGS:
            if cfg_name not in all_data:
                continue
            pair = {'baseline': all_data['baseline'], cfg_name: all_data[cfg_name]}
            suffix = f' (baseline vs {cfg_name})'
            print(f"\n--- {suffix} ---")
            plot_states(pair, save_dir, suffix)
            plot_variances(pair, save_dir, suffix)

    plt.show()


if __name__ == '__main__':
    main()
