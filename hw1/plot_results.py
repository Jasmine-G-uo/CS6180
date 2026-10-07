"""
Plots and summary table for the mini-LLM exercise.
Usage: python plot_results.py
Reads runs/<model>_lr<lr>/loss_log.json, writes figures/*.pdf, figures/*.png and figures/summary.md
"""
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODELS = ['baseline', 'rmsnorm', 'swiglu', 'nope', 'rope', 'gqa']
NAME = dict(baseline='Baseline', rmsnorm='RMSNorm', swiglu='SwiGLU', nope='NoPE', rope='RoPE', gqa='GQA')
LRS = ['5e-4', '1e-3', '2e-3', '4e-3', '8e-3']
MAIN_LR = '1e-3'
COLOR = dict(baseline='#2a78d6', rmsnorm='#eb6834', swiglu='#eb6834', nope='#eb6834', rope='#1baf7a', gqa='#eb6834')
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'

os.makedirs('figures', exist_ok=True)
plt.rcParams.update({
    'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False,
    'axes.spines.right': False, 'legend.frameon': False, 'lines.linewidth': 1.8,
})

def load(model, lr):
    path = os.path.join('runs', f'{model}_lr{lr}', 'loss_log.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

logs = {(m, lr): load(m, lr) for m in MODELS for lr in LRS}

def style(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')

def save(fig, stem):
    fig.tight_layout()
    fig.savefig(f'figures/{stem}.pdf')
    fig.savefig(f'figures/{stem}.png', dpi=200)
    plt.close(fig)

# 1) baseline training and validation loss
d = logs[('baseline', MAIN_LR)]
if d is not None:
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    it = [r['iter'] for r in d['train_iter']]
    ax.plot(it, [r['loss'] for r in d['train_iter']], color='#86b6ef', linewidth=0.8,
            label='Train loss (one mini-batch, every 10 iters)')
    ev = d['eval']
    ax.plot([e['iter'] for e in ev], [e['train'] for e in ev], color='#2a78d6', marker='o', markersize=4,
            label='Train loss (mean of 200 batches)')
    ax.plot([e['iter'] for e in ev], [e['val'] for e in ev], color='#eb6834', marker='o', markersize=4,
            label='Validation loss (mean of 200 batches)')
    ax.set_ylim(1.5, 4.4)
    style(ax)
    ax.legend(fontsize=8.5, loc='upper right')
    save(fig, 'baseline_loss')

# 2) baseline vs variants at the same lr, solid = validation, dashed = training
def compare(models, stem, xmin=250):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    lo, hi = 9, 0
    for m in models:
        d = logs[(m, MAIN_LR)]
        if d is None:
            continue
        ev = [e for e in d['eval'] if e['iter'] >= xmin]
        x = [e['iter'] for e in ev]
        ax.plot(x, [e['val'] for e in ev], color=COLOR[m], marker='o', markersize=3.5, label=f'{NAME[m]} val')
        ax.plot(x, [e['train'] for e in ev], color=COLOR[m], linestyle='--', linewidth=1.4, label=f'{NAME[m]} train')
        lo = min(lo, min(e['train'] for e in ev)); hi = max(hi, max(e['val'] for e in ev))
    ax.set_ylim(lo - 0.05, min(hi + 0.05, 3.0))
    style(ax)
    ax.legend(fontsize=8.5, ncol=len(models), loc='upper right')
    save(fig, stem)

compare(['baseline', 'rmsnorm'], 'cmp_rmsnorm')
compare(['baseline', 'swiglu'], 'cmp_swiglu')
compare(['baseline', 'nope', 'rope'], 'cmp_posenc')
compare(['baseline', 'gqa'], 'cmp_gqa')

# 3) learning rate sweep: best validation loss vs learning rate
fig, ax = plt.subplots(figsize=(6.4, 3.8))
SWEEP_COLOR = dict(baseline='#2a78d6', rmsnorm='#eb6834', swiglu='#1baf7a', nope='#eda100', rope='#e87ba4', gqa='#4a3aa7')
lr_val = [float(lr) for lr in LRS]
for m in MODELS:
    ys = [min(e['val'] for e in logs[(m, lr)]['eval']) if logs[(m, lr)] else float('nan') for lr in LRS]
    ax.plot(lr_val, ys, color=SWEEP_COLOR[m], marker='o', markersize=4, label=NAME[m])
ax.set_xscale('log')
ax.set_xticks(lr_val)
ax.set_xticklabels(LRS)
ax.minorticks_off()
ax.grid(True, color=GRID, linewidth=0.8)
ax.set_xlabel('Peak learning rate')
ax.set_ylabel('Best validation loss')
ax.legend(fontsize=8.5, ncol=3, loc='upper center')
save(fig, 'lr_sweep')

# 4) summary table (markdown)
def best(d):
    return min(d['eval'], key=lambda e: e['val'])

head = '| model | params | ' + ' | '.join(f'best val lr={lr}' for lr in LRS) + ' | best lr |'
rows = [head, '|' + '---|' * (len(LRS) + 3)]
for m in MODELS:
    d = logs[(m, MAIN_LR)]
    vals = {lr: (best(logs[(m, lr)])['val'] if logs[(m, lr)] else float('nan')) for lr in LRS}
    blr = min(LRS, key=lambda lr: vals[lr])
    rows.append(f"| {NAME[m]} | {d['n_params']:,} | " + ' | '.join(f'{vals[lr]:.4f}' for lr in LRS) + f' | {blr} |')
with open('figures/summary.md', 'w') as f:
    f.write('\n'.join(rows) + '\n')
print('\n'.join(rows))
