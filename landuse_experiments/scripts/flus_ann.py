"""FLUS's suitability network (GeoSOS-FLUS 2.3, Liu et al. 2017), reproduced in Python (notebook 09).

FLUS trains one neural network to tell, from the predictors, which class each pixel of the start
map has; its outputs, one probability per class summing to 1, are the suitability maps. From the
FLUS manual (Liu, Li & Liang 2019) and its log files (FilesGenerate/logFileTrain.log,
ANNoutput.log): the network comes from the Shark 3.1 library; one hidden layer of 12 nodes;
predictors scaled to 0-1; trained on 2% of the valid pixels drawn class by class in proportion to
each class's area ("Sampling in Proportion", rate 20 per thousand). Not documented, chosen in
notebook 09 as the closest to FLUS's own maps: logistic hidden nodes, softmax outputs,
cross-entropy, trained by Shark's default full-batch iRprop+ (Igel & Hüsken 2003) for 2,000
iterations.
"""
import numpy as np

import esri_methods as E
import nb_helpers as H

BENCH = H.OUTPUTS / "benchmark"
N_CLASSES = 7                                    # the tool codes 1..7 of benchmark_export


def _logistic(z):
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))


def _softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


class OccurrenceNetwork:
    """One hidden layer of logistic nodes, full-batch iRprop+. FLUS's set-up is the default:
    softmax outputs with cross-entropy; logistic outputs or squared error are the alternatives
    compared in notebook 09 (outputs then rescaled to sum to 1, as FLUS's do)."""

    def __init__(self, hidden=12, iterations=2000, seed=E.SEED, output="softmax", loss="cross-entropy"):
        self.hidden, self.iterations, self.seed, self.output, self.loss = hidden, iterations, seed, output, loss

    def _forward(self, X):
        h = _logistic(X @ self.w1 + self.b1)
        z = h @ self.w2 + self.b2
        return h, _softmax(z) if self.output == "softmax" else _logistic(z)

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        Y = np.eye(N_CLASSES)[y]
        self.w1, self.b1 = rng.uniform(-.1, .1, (X.shape[1], self.hidden)), np.zeros(self.hidden)
        self.w2, self.b2 = rng.uniform(-.1, .1, (self.hidden, N_CLASSES)), np.zeros(N_CLASSES)
        params = [self.w1, self.b1, self.w2, self.b2]
        steps = [np.full_like(p, 0.01) for p in params]
        last = [np.zeros_like(p) for p in params]
        last_error = np.inf
        for _ in range(self.iterations):
            h, out = self._forward(X)
            d_out = (out - Y) / len(X)
            if self.loss == "cross-entropy":           # with softmax outputs, the gradient is out - Y
                error = -np.mean(np.log(np.clip(out[Y == 1], 1e-12, None)))
            else:
                error = np.mean((out - Y) ** 2)
                d_out = d_out * out * (1 - out) if self.output == "logistic" else out * (d_out - (d_out * out).sum(1, keepdims=True))
            d_h = (d_out @ self.w2.T) * h * (1 - h)
            grads = [X.T @ d_h, d_h.sum(0), h.T @ d_out, d_out.sum(0)]
            for p, g, step, g_last in zip(params, grads, steps, last):
                same = g * g_last
                step[same > 0] = np.minimum(step[same > 0] * 1.2, 50)
                step[same < 0] = np.maximum(step[same < 0] * 0.5, 1e-6)
                if error > last_error:                     # iRprop+: undo the steps that overshot
                    p[same < 0] += np.sign(g_last[same < 0]) * step[same < 0]
                g = np.where(same < 0, 0, g)
                p -= np.sign(g) * step
                g_last[:] = g
            last_error = error
        return self

    def predict_proba(self, X, chunk=500_000, rescale=True):
        out = np.vstack([self._forward(X[i:i + chunk])[1] for i in range(0, len(X), chunk)])
        return out / out.sum(1, keepdims=True) if rescale and self.output != "softmax" else out

    def rmse(self, X, y):
        """Root mean squared error against the one-hot classes, as FLUS's log reports it."""
        return float(np.sqrt(np.mean((self.predict_proba(X, rescale=False) - np.eye(N_CLASSES)[y]) ** 2)))


def proportional_sample(y, share=0.02, seed=E.SEED):
    """FLUS's 'Sampling in Proportion': the same share of every class's pixels."""
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.choice(np.flatnonzero(y == k), max(1, round(share * (y == k).sum())), replace=False)
                           for k in np.unique(y)])


def inputs(year, drop=(), extra=()):
    """FLUS's inputs for a start year (benchmark_export): the 0-1 predictors of every valid pixel,
    its class (0..6 = tool codes 1..7) and the valid-pixel mask. The factor files are in the order
    of E.PREDICTORS; drop leaves predictors out by name, extra adds layers (2-D arrays on the same
    grid, scaled 0-1), for notebook 12."""
    import rasterio
    files = sorted((BENCH / "inputs" / f"factors_{year}_flus").glob("*.tif"))
    keep = [f for f, name in zip(files, E.PREDICTORS) if name not in drop]
    factors = np.stack([rasterio.open(f).read(1) for f in keep] + [np.asarray(a, np.float32) for a in extra])
    land = H.read_tif(BENCH / "inputs" / f"landcover_{year}.tif")[0]
    valid = land > 0
    return factors[:, valid].T, land[valid].astype(int) - 1, valid


def probability_map(year, out_path, seed=E.SEED, share=0.02, drop=(), extra=(), log=print):
    """Train on `year`'s map and write the probability of each class as a 7-band float32 GeoTIFF,
    band k = tool code k, like FLUS's Probability-of-occurrence.tif. Returns the network."""
    import rasterio
    X, y, valid = inputs(year, drop, extra)
    sample = proportional_sample(y, share, seed)
    net = OccurrenceNetwork(seed=seed).fit(X[sample], y[sample])
    log(f"[{year}, seed {seed}] trained on {sample.size:,} pixels; RMSE {net.rmse(X[sample], y[sample]):.3f}")
    P = net.predict_proba(X)
    with rasterio.open(BENCH / "inputs" / f"landcover_{year}.tif") as src:
        profile = {**src.profile, "count": N_CLASSES, "dtype": "float32", "nodata": None, **E.COMPRESS, "predictor": 3}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        for k in range(N_CLASSES):
            band = np.zeros(valid.shape, np.float32)
            band[valid] = P[:, k]
            dst.write(band, k + 1)
    return net
