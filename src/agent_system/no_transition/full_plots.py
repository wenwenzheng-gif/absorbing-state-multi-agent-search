"""Full experiment figures. Historical extinction and task success stay separate."""
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from ..analysis.summarize import write_csv
from .plots import STYLES


def plot_full(rows, episodes, mechanisms, historical, output: Path):
    folder = output / 'figures'
    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.dpi': 220, 'pdf.fonttype': 42})
    def save(fig, name):
        for ext in ['png', 'pdf']:
            fig.savefig(folder / f'{name}.{ext}', bbox_inches='tight')
        plt.close(fig)
    def success(ax, sub, color, marker, label, style='-'):
        sub = sorted(sub, key=lambda r: r['degree'])
        if not sub:
            return
        ax.errorbar([r['degree'] for r in sub], [r['p_success'] for r in sub],
                    yerr=[[max(0, r['p_success']-r['ci_low']) for r in sub],
                          [max(0, r['ci_high']-r['p_success']) for r in sub]],
                    color=color, marker=marker, linestyle=style, capsize=3,
                    markerfacecolor='white', label=label)
        ax.set_xticks([r['degree'] for r in sub])
        ax.set_ylim(-.07, 1.07)
        ax.set_xlabel('Communication degree d (N=20)')
        ax.grid(axis='y', alpha=.2)
    for number, task, conditions in [(1, 'Needle', ['Needle-Low', 'Needle-High']),
            (2, 'Chain', ['Chain-Short-Horizon', 'Chain-Sufficient-Horizon'])]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, layout='constrained')
        for ax, condition in zip(axes, conditions):
            for kind, (color, marker, label) in STYLES.items():
                sub = [r for r in rows if r['condition']==condition and r['agent_type']==kind]
                if sub:
                    success(ax, sub, color, marker, f'{label} (n={sub[0]["episodes"]}/degree)',
                            '-' if kind=='deterministic' else '--')
            ax.set_title(condition.replace('-Horizon', ''))
        axes[0].set_ylabel('Task success probability\nWilson 95% interval')
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside lower center', ncol=2)
        fig.suptitle(f'{task} Search · full v2', fontweight='bold')
        save(fig, f'figure_{number}_{task.lower()}')
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
    for j, condition in enumerate(['Needle-Low', 'Needle-High']):
        for i, metric in enumerate(['mean_coverage_fraction', 'mean_duplicate_fraction']):
            ax = axes[i, j]
            for kind, (color, marker, label) in STYLES.items():
                sub = sorted([r for r in mechanisms if r['condition']==condition and r['agent_type']==kind],
                             key=lambda r:r['degree'])
                ax.plot([r['degree'] for r in sub], [r[metric] for r in sub], color=color,
                        marker=marker, markerfacecolor='white', linestyle='-' if kind=='deterministic' else '--', label=label)
            ax.set_title(condition)
            ax.set_xticks([0,4,8,12,16,19])
            ax.set_xlabel('Communication degree d (N=20)')
            ax.set_ylabel('Unique tested / S' if i==0 else 'Duplicate tests / valid tests')
            ax.grid(alpha=.2)
            if i == 0:
                ax.set_ylim((0, .05) if j==0 else (0, 1.05))
            else:
                ax.set_ylim(-.03, 1.03)
        axes[0,j].legend(fontsize=9)
    fig.suptitle('Needle coverage and duplication · full v2\nTwo High passes imply 50% duplicates even under the reference policy', fontsize=12)
    save(fig, 'figure_3_needle_mechanisms')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout='constrained')
    degrees = [0,4,8,12,16,19]
    colors = plt.get_cmap('viridis')(np.linspace(.05,.9,6))
    depth_rows = []
    for ax, condition in zip(axes, ['Chain-Short-Horizon','Chain-Sufficient-Horizon']):
        groups = defaultdict(list)
        for e in episodes:
            if e['condition']==condition:
                groups[e['agent_type'], e['degree']].append(e)
        for (kind, degree), es in sorted(groups.items()):
            horizon = es[0]['horizon']
            values = [[0]+e['depth_after_each_round']+[e['maximum_depth']]*(horizon-len(e['depth_after_each_round'])) for e in es]
            avg = np.mean(values, axis=0)
            ax.plot(range(horizon+1), avg, color=colors[degrees.index(degree)],
                    linestyle='-' if kind=='deterministic' else '--',
                    marker='o' if kind=='deterministic' else 'x', markersize=3, alpha=.8)
            depth_rows.extend({'condition':condition,'agent_type':kind,'degree':degree,'round':r,
                               'mean_depth':float(v),'episodes':len(es)} for r,v in enumerate(avg))
        ax.set_title(condition.replace('-Horizon',''))
        ax.set_xlabel('Global round (completed depth is absorbing)')
        ax.set_ylabel('Mean maximum completed depth')
        ax.set_ylim(-.25,8.5)
        ax.set_yticks(range(9))
        ax.grid(alpha=.2)
    handles = [Line2D([0],[0],color=c,label=f'd={d}') for c,d in zip(colors,degrees)]
    handles += [Line2D([0],[0],color='black',marker='o',label='Deterministic'),
                Line2D([0],[0],color='black',marker='x',linestyle='--',label='GPT-5.4 Nano')]
    fig.legend(handles=handles, loc='outside lower center', ncol=4, fontsize=9)
    fig.suptitle('Chain sequential depth · full v2\nCoincident trajectories are drawn at their measured values', fontsize=12)
    save(fig, 'figure_4_chain_depth')
    write_csv(output/'depth_by_round.csv', depth_rows)
    fig, axes = plt.subplots(1,3,figsize=(14,4.7),layout='constrained')
    ax=axes[0]
    ax.errorbar([r['degree'] for r in historical], [r['p_extinction'] for r in historical],
        yerr=[[max(0,r['p_extinction']-r['ci_low']) for r in historical],
              [max(0,r['ci_high']-r['p_extinction']) for r in historical]],color='#7461A8',marker='o',capsize=3)
    ax.set_title('A  Historical Hierarchical (N=40)\nWrong-lineage extinction; not task success',fontsize=11)
    ax.set_ylabel('P(wrong lineage extinct by T=8)')
    ax.set_xlabel('Communication degree d (N=40)')
    ax.set_xticks([0,1,2,3,4,5,6,8])
    ax.set_ylim(-.07,1.07)
    ax.grid(axis='y',alpha=.2)
    ax.text(.02,.96,'Deterministic n=3/degree\nVerified success unavailable',transform=ax.transAxes,va='top',fontsize=9)
    for ax, letter, task, conditions in [(axes[1],'B','Needle',['Needle-Low','Needle-High']),
                                       (axes[2],'C','Chain',['Chain-Short-Horizon','Chain-Sufficient-Horizon'])]:
        for condition,color,marker in zip(conditions,['#17629C','#C45118'],['o','s']):
            sub=[r for r in rows if r['condition']==condition and r['agent_type']=='llm']
            success(ax,sub,color,marker,condition.replace('-Horizon',''))
        ax.set_title(f'{letter}  {task} full v2 (N=20)\nGPT-5.4 Nano, n=12/degree',fontsize=11)
        ax.set_ylabel('P(verified task success)')
        ax.legend(loc='center',fontsize=9)
    fig.suptitle('Historical mechanism and new task outcomes · separate degree axes\nPoints with Wilson 95% intervals; connecting lines are descriptive',fontsize=12)
    save(fig,'figure_5_historical_comparison')
