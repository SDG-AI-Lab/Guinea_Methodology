"""Always retain the original local implementations in improvement reports."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio

from .comparison import DEFAULT_DATA, evaluate, read_raster, write_csv


def read_mask(path, reference):
    with rasterio.open(path) as src:
        if any(src.profile[key] != reference[key]
               for key in ('crs', 'transform', 'height', 'width')):
            raise ValueError('Evaluation mask grid differs from the land-cover grid')
        return src.read(1).astype(bool)


def compare_with_original(run_directory, data_dir=DEFAULT_DATA):
    directory, data = Path(run_directory), Path(data_dir)
    report = json.loads((directory / 'report.json').read_text())
    source, profile = read_raster(data/'Guinea_LandCover_2015.tif',categorical=True)
    observed, _ = read_raster(data/'Guinea_LandCover_2020.tif',profile,categorical=True)
    mask=read_mask(directory/'evaluation_mask.tif', profile)
    target_hash=hashlib.sha256((data/'Guinea_LandCover_2020.tif').read_bytes()).hexdigest()
    recorded_target=report.get('evaluation_target_sha256')
    for entry in report.get('inputs', []):
        if Path(entry['path']).name == 'Guinea_LandCover_2020.tif':
            recorded_target=entry['sha256']
    if recorded_target != target_hash:
        raise ValueError('New report must record the same target map as the comparison')
    output_root=data.parent/'outputs'
    # Fresh checkouts do not contain generated artifacts. Build the old local
    # baselines only after the new runner has frozen/saved all its forecasts.
    legacy_root=output_root/'local_comparison_2010_2015_2020'
    if not (legacy_root/'report.json').exists():
        from .comparison import run_comparison
        run_comparison(data_dir=data,output_dir=legacy_root,sampling='legacy_balanced')
    controlled_root=output_root/'controlled_2010_2015_2020'
    if not (controlled_root/'report.json').exists():
        from .controlled import run_controlled_comparison
        run_controlled_comparison(data_dir=data,output_dir=controlled_root)
    definitions=[
        ('Original-style transition RF (local reproduction)', 'local_comparison_2010_2015_2020',
         'transition_rf_allocation.tif', 'Balanced samples per transition; no shared spatial split'),
        ('Original-style binary RF, shared samples/blocks', 'controlled_2010_2015_2020',
         'temporal_holdout/binary_rf_allocation.tif', '67,938 shared pixels in 95 original training blocks'),
        ('Initial multiclass RF, shared samples/blocks', 'controlled_2010_2015_2020',
         'temporal_holdout/multiclass_rf_allocation.tif', '67,938 shared pixels in 95 original training blocks')]
    baselines=[]
    for label,folder,filename,training in definitions:
        root=output_root/folder
        old=json.loads((root/'report.json').read_text())
        recorded=next(i['sha256'] for i in old['inputs'] if Path(i['path']).name=='Guinea_LandCover_2020.tif')
        if recorded!=target_hash:
            raise ValueError('Original and new reports must use the same target map')
        source_hash=next(i['sha256'] for i in old['inputs'] if Path(i['path']).name=='Guinea_LandCover_2015.tif')
        if source_hash!=hashlib.sha256((data/'Guinea_LandCover_2015.tif').read_bytes()).hexdigest():
            raise ValueError('Original and new starting maps differ')
        old_mask_path=root/Path(filename).parent/'evaluation_mask.tif'
        if not np.array_equal(read_mask(old_mask_path, profile),mask):
            raise ValueError('Original and new evaluation masks differ')
        pred,_=read_raster(root/filename,profile,categorical=True)
        metrics=evaluate(source,observed,pred,mask)['summary']
        saved=old.get('results',old.get('temporal_holdout',{}).get('results',{}))[Path(filename).stem]['summary']
        if not np.isclose(metrics['figure_of_merit_pct'],saved['figure_of_merit_pct']):
            raise ValueError('Original baseline map no longer reproduces its saved FoM')
        baselines.append(dict(name=label,role='baseline',training_design=training,
            allocation='Historical transition quantities',run_directory=str(root),metrics=metrics))
    old_fom=baselines[0]['metrics']['figure_of_merit_pct']
    controlled_fom=baselines[1]['metrics']['figure_of_merit_pct']
    selected=report.get('selected_model',report.get('selection',{}).get('selected_model'))
    rows=[]
    candidates=baselines+[dict(name=n,role='selected' if n==selected else 'candidate',
        training_design='See candidate manifest: expanded/pooled/weighted variants are labeled',
        allocation='Common fixed demand' if n.endswith('_fixed_demand') else 'Validation-selected decision policy',
        run_directory=str(directory),metrics=r['summary']) for n,r in report['results'].items() if n!='no_change']
    for item in candidates:
        m=item.pop('metrics')
        if m['pixels']!=int(mask.sum()):
            raise ValueError('Model metrics do not cover the common evaluation mask')
        rows.append(dict(**item,forecast_fom=m['figure_of_merit_pct'],
            delta_fom_vs_original=m['figure_of_merit_pct']-old_fom,
            delta_fom_vs_shared_binary=m['figure_of_merit_pct']-controlled_fom,
            overall_accuracy_pct=100*m['overall_accuracy'],
            predicted_change_pct=100*m['predicted_change_pixels']/m['pixels'],
            pixels=m['pixels']))
    result=dict(selected_model=selected,evaluation_period=[2015,2020],rows=rows,
        notes=['Original-style baseline is a local reproduction using bundled terrain/map features, not an untouched cloud-backend run.',
               'The controlled binary baseline incorporates the agreed shared samples and 100x100 spatial blocks.',
               'Feature, training-size, weighting, validation and allocation changes are not algorithm-only comparisons.',
               'Every FoM difference is in percentage points on the identical full evaluation mask.'])
    (directory/'comparison_with_original.json').write_text(json.dumps(result,indent=2))
    write_csv(directory/'comparison_with_original.csv',rows)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',nargs='+',required=True)
    args=parser.parse_args()
    for directory in args.runs:
        result=compare_with_original(directory)
        print(directory, 'compared with both original-style baselines')


if __name__=='__main__':
    main()
