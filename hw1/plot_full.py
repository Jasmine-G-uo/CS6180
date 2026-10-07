"""
Plots and summary for the full default config runs (config/train_shakespeare_char.py, 5000 iters, GPU).
Usage: python plot_full.py
Reads runs_full/<model>_lr<lr>/loss_log.json, writes figures_full/*.pdf, *.png and summary.md
"""
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODELS = ['baseline', 'rmsnorm', 'swiglu', 'nope', 'rope', 'gqa']
NAME = dict(baseline='Baseline', rmsnorm='RMSNorm', swiglu='SwiGLU', nope='NoPE', rope='RoPE', gqa='GQA')
COLOR = dict(baseline='#2a78d6', rmsnorm='#eb6834', swiglu='#eb6834', nope='#eb6834', rope='#1baf7a', gqa='#eb6834')
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
OUT = 'figures_full'
LRS = ['5e-4', '1e-3', '2e-3']
MAIN_LR = '1e-3'

os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False,
    'axes.spines.right': False, 'legend.frameon': False, 'lines.linewidth': 1.8,
})

def load(m, lr=MAIN_LR):
    path = os.path.join('runs_full', f'{m}_lr{lr}', 'loss_log.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

logs = {m: load(m) for m in MODELS}
all_logs = {(m, lr): load(m, lr) for m in MODELS for lr in LRS}

def style(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')

def save(fig, stem):
    fig.tight_layout()
    fig.savefig(f'{OUT}/{stem}.pdf')
    fig.savefig(f'{OUT}/{stem}.png', dpi=200)
    plt.close(fig)

d = logs['baseline']
if d is not None:
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot([r['iter'] for r in d['train_iter']], [r['loss'] for r in d['train_iter']], color='#86b6ef',
            linewidth=0.8, label='Train loss (one mini-batch, every 10 iters)')
    ev = d['eval']
    ax.plot([e['iter'] for e in ev], [e['train'] for e in ev], color='#2a78d6', marker='o', markersize=4,
            label='Train loss (mean of 200 batches)')
    ax.plot([e['iter'] for e in ev], [e['val'] for e in ev], color='#eb6834', marker='o', markersize=4,
            label='Validation loss (mean of 200 batches)')
    b = min(ev, key=lambda e: e['val'])
    ax.annotate(f"best val {b['val']:.4f} (iter {b['iter']})", xy=(b['iter'], b['val']), xytext=(10, 45),
                textcoords='offset points', fontsize=9, color=INK,
                arrowprops=dict(arrowstyle='-', color=MUTED, linewidth=0.8))
    ax.set_ylim(0.5, 4.4)
    style(ax)
    ax.legend(fontsize=8.5, loc='upper right')
    save(fig, 'baseline_loss')

def compare(models, stem, xmin=500):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for m in models:
        d = logs[m]
        if d is None:
            continue
        ev = [e for e in d['eval'] if e['iter'] >= xmin]
        x = [e['iter'] for e in ev]
        ax.plot(x, [e['val'] for e in ev], color=COLOR[m], marker='o', markersize=3.5, label=f'{NAME[m]} val')
        ax.plot(x, [e['train'] for e in ev], color=COLOR[m], linestyle='--', linewidth=1.4, label=f'{NAME[m]} train')
    ax.set_ylim(0.5, 2.0)
    style(ax)
    ax.legend(fontsize=8.5, ncol=len(models), loc='lower left')
    save(fig, stem)

compare(['baseline', 'rmsnorm'], 'cmp_rmsnorm')
compare(['baseline', 'swiglu'], 'cmp_swiglu')
compare(['baseline', 'nope', 'rope'], 'cmp_posenc')
compare(['baseline', 'gqa'], 'cmp_gqa')

# learning rate sweep
fig, ax = plt.subplots(figsize=(6.4, 3.8))
SWEEP_COLOR = dict(baseline='#2a78d6', rmsnorm='#eb6834', swiglu='#1baf7a', nope='#eda100', rope='#e87ba4', gqa='#4a3aa7')
xs = [float(lr) for lr in LRS]
for m in MODELS:
    ys = [min(e['val'] for e in all_logs[(m, lr)]['eval']) if all_logs[(m, lr)] else float('nan') for lr in LRS]
    ax.plot(xs, ys, color=SWEEP_COLOR[m], marker='o', markersize=4, label=NAME[m])
ax.set_xscale('log'); ax.set_xticks(xs); ax.set_xticklabels(LRS); ax.minorticks_off()
ax.grid(True, color=GRID, linewidth=0.8)
ax.set_xlabel('Peak learning rate'); ax.set_ylabel('Best validation loss')
ax.legend(fontsize=8.5, ncol=3, loc='upper center')
save(fig, 'lr_sweep')

def best(d):
    return min(d['eval'], key=lambda e: e['val'])

rows = ['| model | params | ' + ' | '.join(f'best val lr={lr} (iter)' for lr in LRS) + ' | best lr |', '|' + '---|' * (len(LRS) + 3)]
for m in MODELS:
    cells, vals = [], {}
    for lr in LRS:
        d = all_logs[(m, lr)]
        if d is None:
            cells.append('n/a'); continue
        b = best(d); vals[lr] = b['val']; cells.append(f"{b['val']:.4f} ({b['iter']})")
    blr = min(vals, key=vals.get) if vals else 'n/a'
    rows.append(f"| {NAME[m]} | {logs[m]['n_params']:,} | " + ' | '.join(cells) + f' | {blr} |')
rows.append('')
rows += ['| model (lr=1e-3) | params | best val | iter of best | train at best | val at 5000 | train at 5000 |', '|---|---|---|---|---|---|---|']
for m in MODELS:
    d = logs[m]
    if d is None or not d['eval']:
        rows.append(f'| {NAME[m]} | not finished |')
        continue
    ev = d['eval']; b = min(ev, key=lambda e: e['val']); last = ev[-1]
    rows.append(f"| {NAME[m]} | {d['n_params']:,} | {b['val']:.4f} | {b['iter']} | {b['train']:.4f} | {last['val']:.4f} | {last['train']:.4f} |")
with open(f'{OUT}/summary.md', 'w') as f:
    f.write('\n'.join(rows) + '\n')
print('\n'.join(rows))
