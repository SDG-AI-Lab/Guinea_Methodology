"""Five consecutive years of vegetation history, ending at the forecast start."""

import numpy as np

from .modis_phenology import TARGET_DAYS
from .phenology_models import HistoricalPhenology, trajectory_statistics
from .vegetation_context import VegetationContext


class BoundedPhenology(HistoricalPhenology):
    """Keep at most two full seasonal rasters in memory while reading histories."""
    def read_rows(self, year, indices):
        values = super().read_rows(year, indices)
        if year in self.cache:
            self.cache[year] = self.cache.pop(year)
        while len(self.cache) > 2:
            del self.cache[next(iter(self.cache))]
        return values


def history_statistics(values):
    """Values are ordered oldest to newest. Missing years never become zeros."""
    valid = np.isfinite(values)
    count = valid.sum(axis=0)
    clean = np.where(valid, values, 0)
    mean = clean.sum(axis=0)/np.maximum(count, 1)
    std = np.sqrt(np.maximum(0, (clean**2).sum(axis=0)/np.maximum(count, 1)-mean**2))
    t = np.arange(1-len(values), 1, dtype=np.float32)[:, None]
    tmean = (t*valid).sum(axis=0)/np.maximum(count, 1)
    denominator = (t*t*valid).sum(axis=0)-count*tmean*tmean
    slope = np.divide((t*clean).sum(axis=0)-count*tmean*mean, denominator,
        out=np.zeros_like(mean), where=denominator>1e-6)
    slopes = np.stack([np.where(valid[i]&valid[j], (values[j]-values[i])/(j-i), np.nan)
                      for i in range(len(values)) for j in range(i+1, len(values))])
    slopes[:, ~np.isfinite(slopes).any(axis=0)] = 0
    robust = np.nanmedian(slopes, axis=0)
    paired = valid[1:] & valid[:-1]
    changes = np.where(paired, values[1:]-values[:-1], 0)
    recent_available = valid[-1] & valid[-2]
    recent = np.where(recent_available, values[-1]-values[-2], 0)
    previous_count = valid[:-1].sum(axis=0)
    previous_mean = clean[:-1].sum(axis=0)/np.maximum(previous_count, 1)
    anomaly_available = valid[-1] & (previous_count >= 2)
    anomaly = np.where(anomaly_available, values[-1]-previous_mean, 0)
    return dict(mean=mean, interannual_std=std, linear_slope=slope, robust_slope=robust,
        recent_change=recent, recent_change_available=recent_available,
        current_anomaly=anomaly, standardized_anomaly=np.clip(anomaly/np.maximum(std,.02),-5,5),
        anomaly_available=anomaly_available, valid_years=count,
        decreasing_steps=np.sum((changes<0)&paired,axis=0),
        increasing_steps=np.sum((changes>0)&paired,axis=0))


def annual_features(year, indices, read_rows):
    rows, names, histories = [], [], {}
    phase = 2*np.pi*np.asarray(TARGET_DAYS)/365.25
    for lag in range(5):
        cube = read_rows(year-lag, indices)
        for b, variable in enumerate(('ndvi','evi')):
            values = cube[b::2]
            stats = trajectory_statistics(values)
            valid = np.isfinite(values)
            count = valid.sum(axis=0)
            # Recover the fitted harmonic intercept from the existing ridge fit.
            intercept = (np.where(valid,values,0).sum(axis=0)
                -stats['harmonic_cos']*(valid*np.cos(phase)[:,None]).sum(axis=0)
                -stats['harmonic_sin']*(valid*np.sin(phase)[:,None]).sum(axis=0))/(count+.001)
            stats['harmonic_mean'] = np.where(count>=6,intercept,0)
            stats['harmonic_mean_available'] = count>=6
            stats['wet_season_available'] = valid[4:10].sum(axis=0)>=2
            stats['dry_season_available'] = valid[[0,1,2,3,10,11]].sum(axis=0)>=2
            for key, values in stats.items():
                rows.append(values)
                names.append(f'annual_lag_{lag}_{variable}_{key}')
            for key in ('harmonic_mean','wet_season','dry_season','harmonic_amplitude','std'):
                available = count>=6
                if key in ('wet_season','dry_season'):
                    available &= stats[key+'_available']
                histories.setdefault(f'{variable}_{key}',[]).append(np.where(available,stats[key],np.nan))
    for key, values in histories.items():
        for statistic, row in history_statistics(np.stack(values[::-1])).items():
            rows.append(row)
            names.append(f'annual_history_{key}_{statistic}')
    return np.column_stack(rows).astype(np.float32), names


class AnnualVegetation:
    def __init__(self, root, reference_path, spectra=None, years=tuple(range(2001,2016))):
        self.phenology = BoundedPhenology(root, reference_path, years)
        self.context = VegetationContext(self.phenology, spectra)
        self.manifests = dict(self.context.manifests, annual_history_years=5,
            annual_features='Five consecutive annual summaries, robust trends, variability and anomalies',
            excluded_endpoint_features='Previous-five-year spectra and endpoint trends removed; consistent five-year vegetation histories replace them')

    def __call__(self, year, indices):
        if any(y not in self.phenology.years for y in range(year-4, year+1)):
            raise ValueError('A complete five-year observation window is required')
        base, names = self.context(year, indices)
        # Do not let wholly absent year-2000 imagery mark the initial fit period.
        keep = [i for i,n in enumerate(names) if not n.startswith(('modis_previous_','landsat_previous_'))
                and 'annual_trend' not in n]
        base, names = base[:,keep], [names[i] for i in keep]
        extra, extra_names = annual_features(year, indices, self.phenology.read_rows)
        return np.column_stack([base,extra]).astype(np.float32), names+extra_names
