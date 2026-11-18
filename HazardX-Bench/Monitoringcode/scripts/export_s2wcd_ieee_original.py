#!/usr/bin/env python3
"""Export a native two-column IEEE figure, not a downscaled slide-sized plot."""
import argparse
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter
from plot_s2wcd_efficiency import load_rows, callouts, COLORS, FAMILIES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',default='output/efficiency/s2wcd_256_fp32_all29')
    p.add_argument('--output-dir')
    args=p.parse_args();root=Path(args.results)
    out=Path(args.output_dir) if args.output_dir else root/'ieee_final'
    out.mkdir(parents=True,exist_ok=True)
    rows,versions,contended=load_rows(root,allow_contended=False,allow_mixed=True)
    assert len(rows)==29 and not contended
    plt.rcParams.update({'font.family':'Liberation Serif','font.size':9,'axes.labelsize':9,
                         'xtick.labelsize':8.5,'ytick.labelsize':8.5,'pdf.fonttype':42,
                         'ps.fonttype':42,'savefig.facecolor':'white','axes.linewidth':.6})
    width,height=7.16,7.05
    fig=plt.figure(figsize=(width,height),dpi=200)
    axes=[fig.add_axes([.085,.56,.385,.385]),fig.add_axes([.585,.56,.385,.385])]
    anns=[]
    bubble_scale=.016
    for ax,key,title,xlab in zip(axes,['total_params_m','gops_fma1_per_pair'],
                ['(a) Parameters versus throughput','(b) Operations versus throughput'],
                ['Total parameters (M)','Forward operations (G; FMA=1)']):
        for r in rows:
            x,y=r[key],r['throughput_pairs_s_mean'];c=COLORS[r['_family']]
            modern=r['torch']!=versions[0]
            ax.errorbar(x,y,yerr=r['throughput_pairs_s_std'],fmt='none',ecolor=c,
                        capsize=1.8,linewidth=.65,zorder=2)
            # All memory bubbles are circles; center markers encode software.
            ax.scatter(x,y,s=r['peak_allocated_mib']*bubble_scale,facecolor=c,
                       edgecolor=c,alpha=.28,linewidth=.65,marker='o',zorder=3)
            ax.scatter(x,y,s=9 if modern else 5,c=c,marker='^' if modern else 'o',zorder=4)
        ax.set_xscale('log');xs=[r[key] for r in rows]
        ax.set_xlim(min(xs)/1.6,max(xs)*1.75);ax.set_ylim(-6,202)
        ax.xaxis.set_major_locator(FixedLocator([5,10,50,100,500] if key=='total_params_m' else [1,10,100,1000]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:g}'))
        ax.set_yticks([0,50,100,150,200]);ax.tick_params(which='major',length=3,width=.6,pad=3)
        ax.tick_params(which='minor',length=1.5,width=.4)
        ax.set_xlabel(xlab,labelpad=4);ax.set_ylabel('Throughput (pairs/s)',labelpad=4)
        ax.set_title(title,fontsize=9,pad=9)
        ax.spines[['top','right']].set_visible(False)
        ax.spines[['left','bottom']].set_color('#555555')
        ax.grid(axis='y',color='#DADADA',linewidth=.45)
        ax.grid(axis='x',which='major',color='#EEEEEE',linewidth=.4)
        ax.set_axisbelow(True)
        labels=callouts(ax,[(r[key],r['throughput_pairs_s_mean'],str(r['_id'])) for r in rows],fontsize=8.5)
        anns.append((ax,labels))
    keyax=fig.add_axes([.055,.232,.92,.225]);keyax.axis('off')
    keyax.text(0,1.12,'Model index',fontsize=9,fontweight='bold',transform=keyax.transAxes)
    ncol=3;percol=math.ceil(len(rows)/ncol)
    keytexts=[]
    for i,r in enumerate(rows):
        col,line=divmod(i,percol);x=col/3;y=.99-line*.101
        keytexts.append(keyax.text(x,y,f'{r["_id"]:02d}',fontsize=8.5,color=COLORS[r['_family']],fontweight='bold',
                                  va='center',transform=keyax.transAxes))
        keyax.plot(x+.043,y,marker='^' if r['torch']!=versions[0] else 'o',markersize=3.2,
                   color=COLORS[r['_family']],transform=keyax.transAxes,clip_on=False)
        keytexts.append(keyax.text(x+.063,y,r['_name'],fontsize=8.5,va='center',transform=keyax.transAxes))
    family_handles=[Line2D([],[],marker='o',linestyle='',markersize=4,color=COLORS[f],label=f) for f in FAMILIES]
    fig.legend(handles=family_handles,loc='center',bbox_to_anchor=(.5,.178),ncol=3,
               frameon=False,fontsize=8.5,columnspacing=1.2,handletextpad=.45,labelspacing=.5)
    memory_handles=[axes[0].scatter([],[],s=m*bubble_scale,facecolor='#A7B4BE',edgecolor='#697883',
                                  linewidth=.6,alpha=.5,label=f'{m:,}') for m in [500,2000,10000]]
    fig.legend(handles=memory_handles,loc='center',bbox_to_anchor=(.5,.105),ncol=3,frameon=False,
               fontsize=8.5,title='Bubble area: peak allocated GPU memory (MiB)',title_fontsize=9,
               columnspacing=2.2,labelspacing=.6)
    software=[Line2D([],[],marker=m,color='#333333',linestyle='',markersize=4,label=t)
              for m,t in [('o','PyTorch 1.13.1 (27 models)'),('^','PyTorch 2.7.1 (2 models)')]]
    fig.legend(handles=software,loc='center',bbox_to_anchor=(.5,.035),ncol=2,frameon=False,
               fontsize=8.5,title='Software groups; throughput is not framework-normalized',title_fontsize=8.5,
               columnspacing=1.4,handletextpad=.5,labelspacing=.4)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    overlaps=0
    for ax,labels in anns:
        boxes=[a.get_bbox_patch().get_window_extent(renderer) for a in labels]
        overlaps+=sum(a.overlaps(b) for i,a in enumerate(boxes) for b in boxes[i+1:])
    keyboxes=[t.get_window_extent(renderer) for t in keytexts]
    key_overlap=sum(a.overlaps(b) for i,a in enumerate(keyboxes) for b in keyboxes[i+1:])
    if overlaps or key_overlap:raise RuntimeError(f'Label overlaps: plots={overlaps}, index={key_overlap}')
    # Fixed dimensions: no bbox_inches=tight, which would change the physical width.
    stem=out/'fig_efficiency_ieee'
    fig.savefig(stem.with_suffix('.pdf'),bbox_inches=None)
    fig.savefig(stem.with_suffix('.png'),dpi=600,bbox_inches=None)
    layout={'width_inches':width,'height_inches':height,'png_dpi':600,
            'plot_label_overlaps':int(overlaps),'index_label_overlaps':int(key_overlap),
            'font':'Liberation Serif (embedded TrueType, Times-compatible)',
            'plot_and_index_font_pt':8.5,'axis_and_panel_font_pt':9,
            'models':len(rows),'software_groups':versions,'strict_same_framework_comparison':False}
    (out/'figure_validation.json').write_text(json.dumps(layout,indent=2))
    print(stem.with_suffix('.pdf'));print(json.dumps(layout,indent=2))

if __name__=='__main__':main()
