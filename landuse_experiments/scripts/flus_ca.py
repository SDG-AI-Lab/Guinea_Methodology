"""FLUS's cellular automaton, the "self-adaptive inertia and competition mechanism" (Liu et al. 2017),
in code (grid_100m notebook 07).

The allocation is pyflus-ca's port of the CA module of FLUS_console, the C++ program behind
GeoSOS-FLUS (J. Herrera, https://github.com/jorge-herrera-b/pyflus-ca, commit fcdf563, MIT licence,
notice below). Copied unchanged, except that the DAREA option (off in FLUS_console) and restricted
areas (none in our runs) are left out. Per iteration, every cell takes a class drawn by roulette
from suitability x neighbourhood share x cost, with an inertia bonus for its own class that adapts
to the gap between allocated area and demand; a draw converts the cell only while both classes
still need to move towards their demand. FLUS's own settings for our runs (its config_mp.log):
neighbourhood 3, weights 1, every conversion allowed, acceleration 0.1, at most 300 iterations.

Its random numbers differ from FLUS's C++ rand(), so maps match FLUS's statistically, not cell
for cell. Notebook 07 compares it with FLUS's program at 250 m.

pyflus-ca licence:
    MIT License. Copyright (c) 2026 Jorge Herrera. Permission is hereby granted, free of charge, to
    any person obtaining a copy of this software and associated documentation files (the
    "Software"), to deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
    Software, and to permit persons to whom the Software is furnished to do so, subject to the
    following conditions: The above copyright notice and this permission notice shall be included
    in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS",
    WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
    ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
    OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import numpy as np
from numba import njit

SETTINGS = {"size": 3, "acceleration": 0.1, "max_iter": 300}     # FLUS's config_mp.log for our runs


@njit(cache=True)
def _count_classes(land, n_types):
    counts = np.zeros(n_types, dtype=np.int64)
    rows, cols = land.shape
    for i in range(rows):
        for j in range(cols):
            v = land[i, j]
            if 1 <= v <= n_types:
                counts[v - 1] += 1
    return counts


@njit(cache=True)
def _run_ca(land0, prob, future_pixels, cost_matrix, weights, max_iter, neighborhood_size, acceleration,
            seed, stop_tolerance_fraction, stable_iterations):
    np.random.seed(seed)
    rows, cols = land0.shape
    n_types = future_pixels.shape[0]
    land = land0.copy()
    temp = land0.copy()
    counts = _count_classes(land, n_types)
    pixel_sum = 0
    for c in range(n_types):
        pixel_sum += counts[c]
    if neighborhood_size != 1:
        num_windows = neighborhood_size * neighborhood_size - 1
    else:
        num_windows = 1
    half = neighborhood_size // 2

    initial_dist = np.zeros(n_types, dtype=np.float64)
    dyna_dist = np.zeros(n_types, dtype=np.float64)
    best_dist = np.zeros(n_types, dtype=np.float64)
    adjustment_effect = np.ones(n_types, dtype=np.float64)
    opposite2reverse = np.zeros(n_types, dtype=np.int64)
    dist_to_goal = np.zeros(n_types, dtype=np.int64)
    history = np.zeros((max_iter, n_types), dtype=np.int64)
    sumdiff_history = np.zeros(max_iter, dtype=np.int64)
    history_dis = 0
    statistic_history_dis = 0
    last_iter = 0
    val = np.zeros(n_types, dtype=np.int64)
    probability = np.zeros(n_types, dtype=np.float64)
    initial_prob = np.zeros(n_types, dtype=np.float64)
    roulette = np.zeros(n_types + 1, dtype=np.float64)
    neigh_prob = np.zeros(n_types, dtype=np.float64)

    for k in range(max_iter):
        for c in range(n_types):
            dist_to_goal[c] = future_pixels[c] - counts[c]
            if k == 0:
                initial_dist[c] = dist_to_goal[c]
                dyna_dist[c] = initial_dist[c] * 1.01
                best_dist[c] = initial_dist[c]
            if initial_dist[c] == 0:
                adjustment_effect[c] = 1.0
                continue
            if abs(best_dist[c]) > abs(dist_to_goal[c]):
                best_dist[c] = dist_to_goal[c]
            else:
                if abs(initial_dist[c]) > 0:
                    if (abs(dist_to_goal[c]) - abs(best_dist[c])) / abs(initial_dist[c]) > 0.05:
                        opposite2reverse[c] = 1
            if dyna_dist[c] != 0:
                adjustment = dist_to_goal[c] / dyna_dist[c]
            else:
                adjustment = 0.0
            if adjustment < 1.0 and adjustment > 0.0:
                dyna_dist[c] = dist_to_goal[c]
                if initial_dist[c] > 0 and adjustment > (1.0 - acceleration):
                    adjustment_effect[c] = adjustment_effect[c] * (adjustment + acceleration)
                if initial_dist[c] < 0 and adjustment > (1.0 - acceleration):
                    adjustment_effect[c] = adjustment_effect[c] * (1.0 / (adjustment + acceleration))
            if initial_dist[c] > 0 and adjustment > 1.0:
                adjustment_effect[c] = adjustment_effect[c] * adjustment * adjustment
            if initial_dist[c] < 0 and adjustment > 1.0:
                adjustment_effect[c] = adjustment_effect[c] * (1.0 / adjustment) * (1.0 / adjustment)

        for i in range(rows):
            for j in range(cols):
                old_v = land[i, j]
                if old_v < 1 or old_v > n_types:
                    temp[i, j] = old_v
                    continue
                old_type = old_v - 1
                for c in range(n_types):
                    val[c] = 0
                    probability[c] = 0.0
                    initial_prob[c] = 0.0
                    neigh_prob[c] = 0.0
                if neighborhood_size != 1:
                    for dx in range(-half, half + 1):
                        for dy in range(-half, half + 1):
                            if dx == 0 and dy == 0:
                                continue
                            x = i + dx
                            y = j + dy
                            if x < 0 or y < 0 or x >= rows or y >= cols:
                                continue
                            nb = land[x, y]
                            if 1 <= nb <= n_types:
                                val[nb - 1] += 1
                else:
                    for c in range(n_types):
                        val[c] = 1
                inheritance = 10.0 * n_types
                for c in range(n_types):
                    suit = prob[c, i, j]
                    initial_prob[c] = suit
                    neigh = (val[c] / num_windows) * (weights[c] + 0.000000001)
                    p = suit * neigh
                    if old_type == c:
                        p = p * adjustment_effect[c] * inheritance
                    p = p * cost_matrix[old_type, c]
                    probability[c] = p
                total = 0.0
                for c in range(n_types):
                    total += probability[c]
                if total != 0.0:
                    for c in range(n_types):
                        neigh_prob[c] = probability[c] / total
                roulette[0] = 0.0
                for c in range(n_types):
                    roulette[c + 1] = roulette[c] + neigh_prob[c]
                r = np.random.random()
                assigned = False
                for new_type in range(n_types):
                    if r <= roulette[new_type + 1] and r > roulette[new_type]:
                        assigned = True
                        is_convert = False
                        if old_type != new_type and cost_matrix[old_type, new_type] != 0:
                            is_convert = True
                        dis_from = dist_to_goal[old_type]
                        dis_to = dist_to_goal[new_type]
                        if initial_dist[new_type] >= 0 and dis_to == 0:
                            adjustment_effect[new_type] = 1.0
                            is_convert = False
                        if initial_dist[old_type] <= 0 and dis_from == 0:
                            adjustment_effect[old_type] = 1.0
                            is_convert = False
                        if initial_dist[old_type] >= 0 and opposite2reverse[old_type] == 1:
                            is_convert = False
                        if initial_dist[new_type] <= 0 and opposite2reverse[new_type] == 1:
                            is_convert = False
                        if is_convert:
                            r3 = np.random.random()
                            if (r3 + (1.0 / n_types)) / (k + 1.0) < initial_prob[new_type]:
                                is_convert = True
                            else:
                                is_convert = False
                        if is_convert:
                            temp[i, j] = new_type + 1
                            counts[new_type] += 1
                            counts[old_type] -= 1
                            dist_to_goal[new_type] = future_pixels[new_type] - counts[new_type]
                            dist_to_goal[old_type] = future_pixels[old_type] - counts[old_type]
                        else:
                            temp[i, j] = old_v
                        break
                if not assigned:
                    temp[i, j] = old_v

        for c in range(n_types):
            counts[c] = 0
        for i in range(rows):
            for j in range(cols):
                land[i, j] = temp[i, j]
                v = land[i, j]
                if 1 <= v <= n_types:
                    counts[v - 1] += 1
        sum_dis = 0
        for c in range(n_types):
            history[k, c] = counts[c]
            diff = future_pixels[c] - counts[c]
            if diff < 0:
                diff = -diff
            sum_dis += diff
        sumdiff_history[k] = sum_dis
        last_iter = k + 1
        if sum_dis == 0:
            break
        if sum_dis == history_dis:
            statistic_history_dis += 1
        if statistic_history_dis > stable_iterations and sum_dis < pixel_sum * stop_tolerance_fraction:
            break
        history_dis = sum_dis

    return land, counts, history[:last_iter], sumdiff_history[:last_iter]


def simulate(land, prob, demand, seed=0, size=SETTINGS["size"], acceleration=SETTINGS["acceleration"],
             max_iter=SETTINGS["max_iter"], weights=None, cost=None):
    """land: (H, W) classes 1..k, 0 = no data; prob: (k, H, W) probability of each class (negative =
    no data, read as 0); demand: pixels wanted per class, summing to the valid pixels. Returns the
    simulated map and the class counts after each iteration (as FLUS's output.log)."""
    k = len(demand)
    land = np.ascontiguousarray(land, dtype=np.uint8)
    prob = np.ascontiguousarray(np.clip(np.nan_to_num(prob, nan=0.0), 0, None), dtype=np.float32)
    demand = np.asarray(demand, dtype=np.int64)
    if int(demand.sum()) != int(((land >= 1) & (land <= k)).sum()):
        raise ValueError("the demand must add up to the valid pixels of the start map")
    weights = np.ones(k) if weights is None else np.asarray(weights, float)
    cost = np.ones((k, k)) if cost is None else np.asarray(cost, float)
    out, _, history, _ = _run_ca(land, prob, demand, cost, weights, int(max_iter), int(size), float(acceleration),
                                 int(seed), 0.0001, 5)
    return out, history
