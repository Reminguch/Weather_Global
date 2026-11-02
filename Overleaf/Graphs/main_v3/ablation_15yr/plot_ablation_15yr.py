"""Rebuild the appendix (a,b,c) figure from the frozen, verified source values.

Run with Python, NumPy and Matplotlib. No weather data or checkpoints are needed.
The two original figures remain alongside this generator for comparison.
"""
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from paper_style import apply_style, LEGEND_SIZE

DATA = HERE / 'ablation_15yr_data.json'


def linestyle(value):
    return (value[0], tuple(value[1])) if isinstance(value, list) else value


def style(ax, ylim, xmax, xticks):
    ax.set_axisbelow(True)
    ax.grid(color='#e4e6e8', lw=.55)
    ax.axhline(0, color='#454545', lw=.85)
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    for side in ['left', 'bottom']:
        ax.spines[side].set_color('#a3a3a3')
        ax.spines[side].set_linewidth(.6)
    ax.set_ylim(*ylim)
    ax.set_xlim(0, xmax)
    ax.set_xticks(xticks)
    ax.set_xlabel(r'Optimizer updates ($\times 10^3$)', labelpad=3)
    ax.tick_params(length=3, width=.6)
    ax.set_ylabel('Approx. loss reduction\nat 240 h (%)', labelpad=5)


def line(ax, s, **kwargs):
    ax.plot(np.asarray(s['steps'])/1000, s['values'], color=s['color'], lw=1.25,
            linestyle=linestyle(s.get('linestyle', 'solid')), **kwargs)


def main():
    data = json.loads(DATA.read_text())
    groups = data['groups']
    apply_style()
    plt.rcParams['legend.title_fontsize'] = LEGEND_SIZE
    fig = plt.figure(figsize=(6.5, 6.3))
    axa, axb, axc = [fig.add_axes([.135, bottom, .555, .210])
                      for bottom in [.735, .415, .095]]
    style(axa, (-1,29), 15, [0,3,6,9,12,15])
    style(axb, (-34,29), 27, [0,5,10,15,20,25])
    # Only the displayed range changes; the frozen full-length series is retained.
    style(axc, (-1,29), 10, [0,2,4,6,8,10])
    for ax, title in zip([axa,axb,axc],
                        ['(a) Spatial depth', '(b) Memory and training feedback',
                         '(c) Mamba dimensions']):
        ax.set_title(title, loc='left', fontweight='bold', pad=8)
    legend_style = dict(loc='upper left', frameon=False, borderaxespad=0,
                        handlelength=1.8, handletextpad=.6, labelspacing=.4)

    spatial_handles = []
    for s in groups['spatial']:
        line(axa,s)
        spatial_handles.append(Line2D([],[],color=s['color'],lw=1.4,
                    linestyle=linestyle(s['linestyle']),label=s['label']))
        for p in s['swa']:
            filled=p['window']==[6000,11000]
            axa.plot(p['x']/1000,p['value'],marker='*',ms=9,ls='none',
                     mfc=s['color'] if filled else 'white',
                     mec='#303030' if filled else s['color'],mew=.8,zorder=6)
    for p in data['spatial_additional_seeds']:
        axa.plot(p['x']/1000,p['value'],marker='*',ms=5,color='#626262',ls='none',zorder=7)
    depth_legend = axa.legend(handles=spatial_handles,bbox_to_anchor=(1.065,1.0),
                             **legend_style)
    axa.add_artist(depth_legend)
    swa_handles = [
        Line2D([],[],marker='*',ms=8,color='#777777',ls='none',mec='#303030',label='6–11'),
        Line2D([],[],marker='*',ms=8,color='#777777',ls='none',mfc='white',label='2–8'),
        Line2D([],[],marker='*',ms=5,color='#626262',ls='none',label='Seeds 23/24'),
    ]
    axa.legend(handles=swa_handles,bbox_to_anchor=(1.065,.53),
               title=r'SWA window ($\times10^3$)',**legend_style)

    memory_handles=[]
    for s in groups['memory']:
        line(axb,s)
        if s['lr']=='1e-4':
            memory_handles.append(Line2D([],[],color=s['color'],lw=1.4,label=s['label']))
        p=s['swa'][0]
        marker='X' if s['lr']=='3e-4' else '*'
        axb.plot(p['x']/1000,p['value'],ls='none',marker=marker,
                 ms=6 if marker=='X' else 9,
                 mfc='white' if s['lr']=='3e-5' else s['color'],
                 mec=s['color'] if s['lr']=='3e-5' else '#303030',mew=.7,zorder=6)
    memory_legend = axb.legend(handles=memory_handles,bbox_to_anchor=(1.065,1.0),
                               **legend_style)
    axb.add_artist(memory_legend)

    aliases={'S16':'16 / 16','S128':'128 / 16 (1)','I256':'256 / 16 (1)',
             'H256i128':'128 / 16 (2)','H256':'256 / 16 (2)','N64':'128 / 64'}
    size_handles=[]
    for s in groups['size']:
        line(axc,s)
        p=s['swa'][0]
        axc.plot(p['x']/1000,p['value'],marker='*',ms=9,color=s['color'],
                 ls='none',mec='#303030',mew=.7,zorder=6)
        size_handles.append(Line2D([],[],color=s['color'],lw=1.4,
                              label=aliases[s['id']]))
    axc.legend(handles=size_handles,bbox_to_anchor=(1.065,1.0),
               title=r'$d_{\rm inner}$ / $d_{\rm state}$',**legend_style)

    lr_handles=[
        Line2D([],[],color='#555555',ls=(0,(1,1.4)),marker='*',mfc='white',ms=7,
               label=r'$3\times10^{-5}$'),
        Line2D([],[],color='#555555',ls='-',marker='*',ms=7,label=r'$10^{-4}$'),
        Line2D([],[],color='#555555',ls=(0,(5,2)),marker='X',ms=5,label=r'$3\times10^{-4}$'),
    ]
    axb.legend(handles=lr_handles,bbox_to_anchor=(1.065,.40),
               title='Learning rate',**legend_style)
    for extension in ['pdf','png']:
        fig.savefig(HERE/f'additional_ablations_15yr.{extension}',dpi=300)
    plt.close(fig)


if __name__=='__main__':
    main()
