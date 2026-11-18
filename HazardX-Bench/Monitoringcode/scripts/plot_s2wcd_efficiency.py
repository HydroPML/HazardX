#!/usr/bin/env python3
"""Readable numbered overview and family detail plots with collision-free callouts."""
import argparse
import json
import math
from pathlib import Path

FAMILIES = ['Task-specific CNN','Task-specific Transformer','Change detection','General vision FM','Earth observation FM']
COLORS = dict(zip(FAMILIES,['#0072B2','#7C5AA6','#CC79A7','#009E73','#D48B00']))
NAMES = {
 'cd_unet':'U-Net', 'cd_bit_r50':'BIT / ResNet-50','cd_siamcrnn_r50':'SiamCRNN / ResNet-50',
 'cd_dsifn':'DSIFN','cd_changeos_r50':'ChangeOS / ResNet-50','cd_changemamba':'ChangeMamba',
 'cd_anysat_base':'AnySat Base','cd_clay_v15_large':'Clay v1.5 Large',
 'cd_dinov3_dpt_vitb_lvd':'DINOv3 ViT-B/16','cd_dinov3_dpt_vitl_lvd':'DINOv3 ViT-L/16',
 'cd_dofav2_base':'DOFA v2 Base','cd_galileo_nano':'Galileo Nano','cd_prithvieo2':'Prithvi-EO 2.0',
 'cd_satmae_b':'SatMAE Base','cd_skysensepp':'SkySense++','cd_spectralgpt':'SpectralGPT',
 'cd_terramind':'TerraMind Base','cd_olmoearth_nano':'OLMoEarth Nano','cd_satlas_tiny':'SatlasPretrain Tiny',
 'cd_sam2_fpn_bplus':'SAM 2.1 Base+','cd_sam2_fpn_l':'SAM 2.1 Large',
 'cd_sam2_fpn_s':'SAM 2.1 Small','cd_sam2_fpn_t':'SAM 2.1 Tiny',
 **{f'cd_segformer_b{i}':f'SegFormer B{i}' for i in range(6)},
}


def family(row):
    name=row['model']
    if name=='unet':return FAMILIES[0]
    if name=='segformer':return FAMILIES[1]
    if name in ('bit','siamcrnn','dsifn','changemamba','changeos'):return FAMILIES[2]
    if name in ('dinov3dpt','sam2fpn','sam2'):return FAMILIES[3]
    return FAMILIES[4]


def load_rows(root, allow_contended=False, allow_mixed=False):
    report_path=root/'completeness_report.json'
    report=json.loads(report_path.read_text()) if report_path.exists() else {}
    contended=report.get('timing_requires_idle_gpu_rerun',False)
    if contended and not allow_contended:raise SystemExit('GPU contention recorded; use --allow-contended only for diagnostic plots.')
    rows=[json.loads(p.read_text()) for p in sorted(root.glob('cd_*.json'))]
    rows=[r for r in rows if r.get('status')=='ok' and r.get('device','').startswith('cuda')]
    if not rows:raise SystemExit('No successful GPU measurements')
    # Every factor except software must match even when mixed frameworks are explicitly enabled.
    protocols={(r['patch_size'],r['batch_size'],r['precision'],r['gpu_name'],r['operator_policy'],
                tuple(r['sample_ids']),r['settings']['samples'],r['settings']['warmup'],
                r['settings']['iterations'],r['settings']['repeats']) for r in rows}
    if len(protocols)!=1:raise SystemExit('Mixed hardware, inputs, precision, or timing protocols; cannot combine.')
    versions=sorted({r['torch'] for r in rows})
    if len(versions)>1 and not allow_mixed:raise SystemExit('Mixed PyTorch environments; explicitly pass --allow-mixed-frameworks for a descriptive comparison.')
    rows.sort(key=lambda r:(FAMILIES.index(family(r)),NAMES.get(r['config_stem'],r['config_stem'])))
    for i,r in enumerate(rows,1):r['_id']=i;r['_name']=NAMES.get(r['config_stem'],r['config_stem']);r['_family']=family(r)
    return rows,versions,contended


def callouts(ax, items, fontsize=9):
    """Greedy pixel-space placement, converted to points to preserve layout in PDF/300dpi PNG."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.transforms import Bbox
    fig=ax.figure;fig.canvas.draw();renderer=fig.canvas.get_renderer()
    bounds=ax.get_window_extent(renderer).shrunk(.98,.96)
    font=FontProperties(size=fontsize,weight='medium')
    used=[];annotations=[]
    dots=[ax.transData.transform((x,y)) for x,y,_ in items]
    # Short and deterministic nearest-position search; avoid every prior label and every point.
    for (x,y,label),(px,py) in zip(items,dots):
        width,_,_=renderer.get_text_width_height_descent(label,font,ismath=False)
        width+=8; height=fontsize*fig.dpi/72*1.55+4
        candidates=[]
        for radius in [16,25,37,52,72,96,128,170,220,280,360,460]:
            for angle in [30,150,-30,-150,90,-90,0,180,60,120,-60,-120]:
                a=math.radians(angle);cx=px+radius*math.cos(a);cy=py+radius*math.sin(a)
                box=Bbox.from_bounds(cx-width/2,cy-height/2,width,height)
                if box.x0<bounds.x0 or box.x1>bounds.x1 or box.y0<bounds.y0 or box.y1>bounds.y1:continue
                overlap=sum(max(0,min(box.x1,b.x1)-max(box.x0,b.x0))*max(0,min(box.y1,b.y1)-max(box.y0,b.y0)) for b in used)
                covered=sum(box.contains(*d) for d in dots)
                score=overlap*1e6+covered*1e4+radius+abs(cy-py)*.03
                candidates.append((score,cx,cy,box))
        if not candidates:raise RuntimeError(f'No label position for {label}')
        _,cx,cy,box=min(candidates,key=lambda c:c[0]);used.append(box)
        ann=ax.annotate(label,(x,y),xytext=((cx-px)*72/fig.dpi,(cy-py)*72/fig.dpi),
            textcoords='offset points',ha='center',va='center',fontsize=fontsize,fontweight='medium',
            color='#18232D',bbox=dict(facecolor='white',edgecolor='none',alpha=.92,pad=.6),
            arrowprops=dict(arrowstyle='-',color='#8B969E',linewidth=.55,shrinkA=2,shrinkB=3),zorder=5)
        annotations.append(ann)
    return annotations


def style(ax,xlabel):
    from matplotlib.ticker import FuncFormatter,LogLocator
    ax.set_xscale('log');ax.set_xlabel(xlabel,labelpad=8)
    ax.set_ylabel('Throughput (temporal pairs/s)',labelpad=8)
    ax.xaxis.set_major_locator(LogLocator(base=10,subs=[1,2,5]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}'))
    ax.grid(axis='y',color='#DFE4E8',linewidth=.7);ax.grid(axis='x',color='#EDF0F2',linewidth=.5)
    ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color('#8F9AA4');ax.tick_params(colors='#34434F',labelsize=9)


def points(ax,rows,key,versions):
    for r in rows:
        x=r.get(key);mem=r.get('peak_allocated_mib')
        if x is None or x<=0 or mem is None:continue
        y=r['throughput_pairs_s_mean'];c=COLORS[r['_family']]
        marker='^' if len(versions)>1 and r['torch']!=versions[0] else 'o'
        ax.errorbar(x,y,yerr=r.get('throughput_pairs_s_std',0),fmt='none',ecolor=c,capsize=2,linewidth=.8,zorder=2)
        ax.scatter(x,y,s=mem*.045,c=c,edgecolor=c,alpha=.38,linewidth=1,marker=marker,zorder=3)
        ax.scatter(x,y,s=8,c=c,marker=marker,zorder=4)
    valid=[r for r in rows if r.get(key) is not None and r[key]>0]
    if valid:
        xs=[r[key] for r in valid];ys=[r['throughput_pairs_s_mean'] for r in valid]
        ax.set_xlim(min(xs)/1.48,max(xs)*1.65)
        ymax=max(y+r.get('throughput_pairs_s_std',0) for y,r in zip(ys,valid))
        ax.set_ylim(-max(2,ymax*.035),ymax*1.16)
    return valid


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',default='output/efficiency/s2wcd_256_fp32_seed43_idle')
    p.add_argument('--output',help='Output stem; also creates _families_parameters and _families_operations')
    p.add_argument('--allow-contended',action='store_true');p.add_argument('--allow-mixed-frameworks',action='store_true')
    args=p.parse_args();root=Path(args.results)
    rows,versions,contended=load_rows(root,args.allow_contended,args.allow_mixed_frameworks)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10,
                         'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
    stem=Path(args.output) if args.output else root/'efficiency'
    stem.parent.mkdir(parents=True,exist_ok=True)
    framework_note=('Software differs: '+ '; '.join(f'{"circle" if i==0 else "triangle"}: PyTorch {v}' for i,v in enumerate(versions))+
                    '. Throughput comparison is descriptive.') if len(versions)>1 else f'PyTorch {versions[0]}'
    prefix='DIAGNOSTIC • contended GPU timings | ' if contended else ''
    subtitle=f'{prefix}S2WCD | {rows[0]["gpu_name"]} | 256×256 per date | 13 bands/date | batch 1 | FP32'
    handles=[Line2D([],[],marker='o',linestyle='',color=COLORS[f],label=f,markersize=6) for f in FAMILIES if any(r['_family']==f for r in rows)]
    overlap_report={}
    def export(fig,path,anns):
        fig.canvas.draw();renderer=fig.canvas.get_renderer();overlaps=0
        for ax,labels in anns:
            boxes=[a.get_bbox_patch().get_window_extent(renderer) for a in labels]
            overlaps+=sum(a.overlaps(b) for i,a in enumerate(boxes) for b in boxes[i+1:])
        overlap_report[path.name]={'label_overlaps':int(overlaps)}
        if overlaps:raise RuntimeError(f'{path.name}: {overlaps} label collisions; refusing export')
        for ext in ['png','pdf','svg']:fig.savefig(path.with_suffix('.'+ext),dpi=300,bbox_inches='tight',pad_inches=.16)
        plt.close(fig)
    # Overview: short IDs on plots, full names in a spacious three-column key.
    fig=plt.figure(figsize=(15.5,10.5),dpi=120)
    gs=fig.add_gridspec(2,2,height_ratios=[1.55,1],left=.065,right=.975,top=.87,bottom=.06,hspace=.35,wspace=.22)
    axs=[fig.add_subplot(gs[0,i]) for i in range(2)];keyax=fig.add_subplot(gs[1,:]);keyax.axis('off')
    fig.suptitle('Model efficiency on S2WCD',x=.065,y=.98,ha='left',fontsize=19,fontweight='bold',color='#18232D')
    fig.text(.065,.944,subtitle,fontsize=10,color='#4C5A65')
    fig.text(.065,.918,framework_note,fontsize=9,color='#6A4A24' if len(versions)>1 else '#4C5A65')
    anns=[]
    for ax,key,title,xlab in zip(axs,['total_params_m','gops_fma1_per_pair'],['(a) Model size and speed','(b) Computation and speed'],
                                ['Total parameters (M)','Operations per pair (G; FMA=1)']):
        valid=points(ax,rows,key,versions);style(ax,xlab);ax.set_title(title,loc='left',fontweight='bold',pad=14)
        anns.append((ax,callouts(ax,[(r[key],r['throughput_pairs_s_mean'],str(r['_id'])) for r in valid],9)))
    keyax.text(0,1.1,'Model index',fontsize=12,fontweight='bold',transform=keyax.transAxes)
    percol=math.ceil(len(rows)/3)
    for i,r in enumerate(rows):
        col,line=divmod(i,percol);x=col/3;y=.96-line*.078
        marker='▲' if len(versions)>1 and r['torch']!=versions[0] else '●'
        keyax.text(x,y,f'{r["_id"]:02d}  {marker}',color=COLORS[r['_family']],fontweight='bold',fontsize=10,transform=keyax.transAxes)
        keyax.text(x+.055,y,r['_name'],fontsize=10,transform=keyax.transAxes,color='#22323E')
    keyax.legend(handles=handles,loc='lower left',bbox_to_anchor=(-.005,-.08),ncol=3,frameon=False,fontsize=9)
    # Memory scale in reserved whitespace, outside the data panels.
    mem_handles=[axs[0].scatter([],[],s=v*.045,facecolor='#A7B4BE',edgecolor='#596C79',alpha=.5,label=f'{v:g} MiB') for v in [500,2000,10000]]
    fig.legend(handles=mem_handles,loc='lower right',bbox_to_anchor=(.975,.005),ncol=3,frameon=False,fontsize=8,title='Bubble area: peak GPU memory',title_fontsize=9)
    fig.text(.065,.015,'Error bars: SD over 3 repeats. Numbered callouts map to the model index. FMA=1; analytical operator-count policy.',fontsize=8,color='#5E6B75')
    export(fig,stem,anns)
    # Detail figures: full names with leader lines in independent family panels.
    for key,suffix,xlab in [('total_params_m','families_parameters','Total parameters (M)'),('gops_fma1_per_pair','families_operations','Operations per pair (G; FMA=1)')]:
        fig,axes=plt.subplots(2,3,figsize=(18,11),dpi=120)
        fig.subplots_adjust(left=.055,right=.97,top=.86,bottom=.085,wspace=.25,hspace=.36)
        fig.suptitle('Efficiency by model family',x=.055,y=.977,ha='left',fontsize=19,fontweight='bold',color='#18232D')
        fig.text(.055,.941,subtitle,fontsize=10,color='#4C5A65');fig.text(.055,.915,framework_note,fontsize=9,color='#6A4A24')
        anns=[]
        for ax,f in zip(axes.flat,FAMILIES):
            sub=[r for r in rows if r['_family']==f];valid=points(ax,sub,key,versions);style(ax,xlab)
            ax.set_title(f,loc='left',color=COLORS[f],fontweight='bold',fontsize=12,pad=14)
            anns.append((ax,callouts(ax,[(r[key],r['throughput_pairs_s_mean'],r['_name']) for r in valid],8.5)))
        info=axes.flat[-1];info.axis('off')
        info.text(.02,.92,'Reading the figure',fontweight='bold',fontsize=13,transform=info.transAxes)
        info.text(.02,.81,'Higher = faster inference\nFurther left = lower cost\nLarger bubble = more peak GPU memory\nWhiskers = repeat standard deviation\n\nAxes use independent ranges per family.\nFMA=1; includes analytical estimates.',fontsize=11,linespacing=1.8,transform=info.transAxes,va='top',color='#455660')
        info.legend(handles=mem_handles,loc='lower left',frameon=False,fontsize=9)
        fig.text(.055,.025,'Full names use automatically placed callouts. Circle/triangle symbols identify software groups when applicable.',fontsize=9,color='#5E6B75')
        export(fig,stem.with_name(stem.name+'_'+suffix),anns)
    (stem.parent/(stem.name+'_layout_check.json')).write_text(json.dumps(overlap_report,indent=2))
    index=[{'id':r['_id'],'name':r['_name'],'config':r['config_stem'],'family':r['_family'],'torch':r['torch']} for r in rows]
    (stem.parent/(stem.name+'_model_index.json')).write_text(json.dumps(index,indent=2))
    print(stem.with_suffix('.pdf'))
    print('Verified label overlaps:',overlap_report)

if __name__=='__main__':main()
