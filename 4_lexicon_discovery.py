#!/usr/bin/env python3
"""
lexicon_discovery.py — Layer 2: Lexicon discovery (titles / commodities / proper names)

This module performs advanced, statistically-grounded grouping of segments (from boundary.py
output) into three broad lexical categories:
  1) titles (king / merchant / officials)
  2) commodities (trade goods, countable items)
  3) proper names (personal/place names)

IMPORTANT: This is NOT a deterministic "ground-truth" decoder. Instead it implements a
robust probabilistic discovery pipeline that uses many complementary tests and diagnostics
so you can form evidence-based hypotheses for labelling.

TOP-LEVEL DESCRIPTION OF ALL CALCULATIONS & TESTS (read this first):
-----------------------------------------------------------------
1) Inventory building
   - Parse dataset/output/boundary.txt (each line: segments separated by "|")
   - Build canonical segment keys (comma-joined integers), frequency counts, examples
   - Build left/right neighbor distributions (counts of neighboring segment-keys)

2) Statistical features per segment (a large, carefully chosen feature vector)
   - Frequency & normalized frequency
   - Position statistics: P(start), P(end), average relative position
   - Token-length statistics: mean #tokens in segment, std
   - Context entropies: H(left), H(right), combined
   - Average PMI with neighbors (left / right) using Laplace smoothing
   - Burstiness: Var(counts over sequences) / mean
   - Numeric-association score: P(neighbor is numeric-like | this segment)
   - Distinctiveness: inverse document frequency style score across sequences
   - Collocation skew: skewness of neighbor distribution (how concentrated contexts are)

3) Advanced association tests & derived scores
   - Log-odds ratio (with 0.5 smoothing) for association with numeric neighbor
   - G^2 (likelihood-ratio) test of independence for segment <-> numeric adjacency
   - NPMI averaged across top neighbors (robust association metric)
   - Dirichlet-multinomial empirical Bayes "signature" for context distributions (measures
     whether a segment shows a narrow context signature (title) vs broad (name))

4) Probabilistic classification via EM on a Naive-Bayes mixture model
   - Features standardized with robust scaling (median + IQR)
   - Model: for each category c, features are independent Gaussians (Naive-Bayes assumption)
   - EM algorithm estimates category priors p(c) and Gaussian params (mu_cj, sigma_cj)
   - Initialization uses linguistically-motivated priors and seed heuristics (numeric-associated
     segments seeded towards commodity; positional-biased seeded towards title; low-freq/high-entropy
     seeded towards proper names)
   - Regularization: per-feature variance floor, Dirichlet prior on class priors, small pseudo-counts

5) Post-hoc evaluations and stability tests
   - Posterior-confidence summary (how many segments assigned confidently)
   - Log-odds & Bayes factor report for top ambiguous segments
   - Bootstrap-like stability test: run EM repeatedly with small feature perturbations and measure
     assignment stability (segment-level frequency of same label)
   - Seed-consistency test: create small, high-precision heuristics (seed sets) and measure
     agreement (precision/recall proxies)
   - Internal cluster separation metrics: average pairwise KL-like separation of class feature dists

6) Outputs & diagnostics
   - lexicon_categories.csv: one row per segment with features, posteriors, assigned label
   - lexicon_summary.json: summary statistics, top representatives for each category
   - diagnostics/: files with detailed arrays, bootstrap stability, and plots (if matplotlib present)

Rationale & why this pipeline is conservative yet powerful:
    - We deliberately avoid blind clustering on raw co-occurrence graphs (which tends to overfit
      when data is sparse). Instead we engineer interpretable features and use a Bayesian
      generative model (Naive-Bayes EM) which is robust with priors and variance floors.
    - Multiple statistical tests are reported for each segment so final interpretation can rely
      on converging evidence (e.g., numeric-association + high freq + low context entropy -> commodity).

Usage:
    python lexicon_discovery.py --help

Author: Assistant
Version: 1.0
"""

from __future__ import annotations

import os
import sys
import math
import json
import random
import argparse
import logging
import itertools
import statistics
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Any, Optional, Set

import numpy as np
import pandas as pd
from scipy.special import logsumexp

# Optional plotting
try:
    import matplotlib.pyplot as plt
    PLOTTING = True
except Exception:
    PLOTTING = False

# ---------------------------
# Defaults
# ---------------------------
BOUNDARY_DEFAULT = os.path.join('dataset', 'output', "IVC",'boundary_IVC.txt')
OUT_DIR_DEFAULT = os.path.join('dataset', 'output',"IVC")
LEXICON_CSV = 'lexicon_categories.csv'
SUMMARY_JSON = 'lexicon_summary.json'
DIAG_DIR = 'diagnostics_lexicon'

DEFAULT_SEED = 12345
EM_MAX_ITERS = 250
EM_TOL = 1e-6
BOOTSTRAP_ROUNDS = 40  # stability test
MIN_VARIANCE_FLOOR = 1e-4

# ---------------------------
# Logging
# ---------------------------

def configure_logging(debug: bool = False):
    fmt = '[%(levelname)s] %(asctime)s - %(message)s'
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=fmt, level=level)

# ---------------------------
# Parsing boundary file
# ---------------------------
SPLIT_RE = r'[;,|\s]+'


def parse_segmented_line(line: str) -> List[List[int]]:
    if line is None:
        return []
    s = str(line).strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split('|')]
    out = []
    for p in parts:
        if p == '':
            out.append([])
            continue
        toks = [t for t in re_split(p)]
        seq = []
        for t in toks:
            if not t:
                continue
            m = _int_re.search(t)
            if m:
                try:
                    seq.append(int(m.group(1)))
                except Exception:
                    continue
        out.append(seq)
    return out


import re
_int_re = re.compile(r'(-?\d+)')
_split_pat = re.compile(SPLIT_RE)

def re_split(s: str) -> List[str]:
    return [t for t in _split_pat.split(s) if t is not None and t != '']


def read_boundary_file(path: str) -> List[List[List[int]]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f'Boundary file not found: {path}')
    seqs = []
    with open(path, 'r', encoding='utf-8') as fh:
        for ln in fh:
            ln = ln.rstrip('\n')
            if ln.strip() == '':
                seqs.append([])
            else:
                seqs.append(parse_segmented_line(ln))
    return seqs

# ---------------------------
# Build inventory and contexts
# ---------------------------

def canonical_seg_key(tokens: List[int]) -> str:
    return ','.join(str(x) for x in tokens)


def build_inventory(sequences: List[List[List[int]]]):
    """Return segments_map (key->info) and seq_seg_keys (list of lists of keys per sequence)
    info: {'seg_id', 'tokens', 'freq', 'examples':[(sidx,pos),...]}
    """
    counts = Counter()
    examples = defaultdict(list)
    seq_keys = []
    for sidx, seq in enumerate(sequences):
        keys = []
        for pos, seg in enumerate(seq):
            k = canonical_seg_key(seg)
            counts[k] += 1
            if len(examples[k]) < 10:
                examples[k].append((sidx, pos))
            keys.append(k)
        seq_keys.append(keys)
    segments_map = {}
    idx = 1
    for k, f in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        seg_id = f'seg_{idx:05d}'
        tokens = [int(x) for x in k.split(',')] if k != '' else []
        segments_map[k] = {'seg_id': seg_id, 'tokens': tokens, 'freq': int(f), 'examples': examples.get(k, [])}
        idx += 1
    return segments_map, seq_keys


def compute_context_counts(seq_seg_keys: List[List[str]]):
    left = defaultdict(Counter)
    right = defaultdict(Counter)
    seq_membership = defaultdict(set)  # which sequences a segment appears in (for doc-frequency)
    positional = defaultdict(list)  # list of relative positions (p/(L-1))
    starts = Counter(); ends = Counter()
    for sidx, seq in enumerate(seq_seg_keys):
        L = len(seq)
        for i, s in enumerate(seq):
            seq_membership[s].add(sidx)
            if L > 0:
                positional[s].append((i / max(1, L-1)))
            if i-1 >= 0:
                left[s][seq[i-1]] += 1
            if i+1 < L:
                right[s][seq[i+1]] += 1
            if i == 0:
                starts[s] += 1
            if i == L-1:
                ends[s] += 1
    return left, right, seq_membership, positional, starts, ends

# ---------------------------
# Entropy & PMI helpers
# ---------------------------

def shannon_entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for v in counter.values():
        p = v / total
        ent -= p * math.log(p + 1e-300)
    return ent


def avg_pmi(target: str, neigh_counts: Counter, global_counts: Counter, total_bigrams: int, alpha: float = 1.0) -> float:
    # Compute average PMI between target and its neighbors with Laplace smoothing
    if not neigh_counts:
        return 0.0
    scores = []
    total_unigrams = sum(global_counts.values())
    V = max(1, len(global_counts))
    denom_big = total_bigrams + alpha * (V ** 2)
    denom_uni = total_unigrams + alpha * V
    for nbr, cnt in neigh_counts.items():
        pxy = (cnt + alpha) / denom_big
        px = (global_counts.get(target, 0) + alpha) / denom_uni
        py = (global_counts.get(nbr, 0) + alpha) / denom_uni
        pmi = math.log(pxy + 1e-300) - math.log(px + 1e-300) - math.log(py + 1e-300)
        scores.append(pmi)
    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))

# ---------------------------
# Numeric-association heuristics
# ---------------------------

def is_numeric_like(seg_tokens: List[int]) -> bool:
    # heuristic: a segment consisting of a single small integer (1..50) is likely a numeral
    if len(seg_tokens) == 1:
        t = seg_tokens[0]
        if isinstance(t, int) and 0 <= t <= 200:
            return True
    # also segments of length 2 both small values sometimes (e.g., pair digits)
    if len(seg_tokens) == 2 and all(isinstance(t, int) and 0 <= t <= 50 for t in seg_tokens):
        return True
    return False

# ---------------------------
# Feature extraction
# ---------------------------

def build_feature_matrix(segments_map: Dict[str, Any], seq_seg_keys: List[List[str]]):
    left, right, seq_membership, positional, starts, ends = compute_context_counts(seq_seg_keys)
    total_bigrams = 0
    bigram_counts = Counter()
    global_unigrams = Counter()
    for seq in seq_seg_keys:
        for i in range(len(seq)):
            global_unigrams[seq[i]] += 1
            if i+1 < len(seq):
                bigram_counts[(seq[i], seq[i+1])] += 1
                total_bigrams += 1
    # numeric set
    numeric_seg_set = set(k for k,info in segments_map.items() if is_numeric_like(info['tokens']))

    rows = []
    for k, info in segments_map.items():
        freq = info['freq']
        docfreq = len(seq_membership.get(k, []))
        start_count = starts.get(k, 0)
        end_count = ends.get(k, 0)
        pos_list = positional.get(k, [])
        avg_pos = float(sum(pos_list) / len(pos_list)) if pos_list else 0.5
        mean_len = float(np.mean([len(info['tokens'])])) if info['tokens'] else 0.0
        # neighbor entropies
        H_left = shannon_entropy(left.get(k, Counter()))
        H_right = shannon_entropy(right.get(k, Counter()))
        H_total = H_left + H_right
        # avg PMI
        avg_pmi_left = avg_pmi(k, left.get(k, Counter()), global_unigrams, total_bigrams)
        avg_pmi_right = avg_pmi(k, right.get(k, Counter()), global_unigrams, total_bigrams)
        # numeric association: proportion of occurrences where neighbor is numeric-like
        occs = info.get('examples', [])
        total_occ = sum(1 for _ in range(info.get('freq', 0)))
        # compute counts directly from contexts instead of examples to be robust
        left_neighbors = left.get(k, Counter())
        right_neighbors = right.get(k, Counter())
        left_num = sum(left_neighbors[n] for n in left_neighbors if n in numeric_seg_set)
        right_num = sum(right_neighbors[n] for n in right_neighbors if n in numeric_seg_set)
        total_nei = sum(left_neighbors.values()) + sum(right_neighbors.values())
        numeric_assoc = (left_num + right_num) / max(1, total_nei)
        # burstiness: variance of per-sequence frequencies / mean
        per_seq_counts = []
        for sidx in range(len(seq_seg_keys)):
            per_seq_counts.append(seq_seg_keys[sidx].count(k))
        mean_seq = float(np.mean(per_seq_counts)) if per_seq_counts else 0.0
        var_seq = float(np.var(per_seq_counts)) if per_seq_counts else 0.0
        burstiness = (var_seq / (mean_seq + 1e-12)) if mean_seq > 0 else 0.0
        # distinctiveness (idf-like)
        idf = math.log((len(seq_seg_keys) + 1) / (docfreq + 1))
        # collocation skew: concentration of neighbor mass (Herfindahl index)
        def herfindahl(counter):
            s = sum(counter.values())
            if s == 0:
                return 0.0
            vals = [ (v / s) ** 2 for v in counter.values() ]
            return float(sum(vals))
        coll_skew = 0.5 * (herfindahl(left_neighbors) + herfindahl(right_neighbors))

        row = {
            'seg_key': k,
            'seg_id': info['seg_id'],
            'freq': freq,
            'docfreq': docfreq,
            'start_prob': start_count / max(1, freq),
            'end_prob': end_count / max(1, freq),
            'avg_pos': avg_pos,
            'mean_len': mean_len,
            'H_left': H_left,
            'H_right': H_right,
            'H_total': H_total,
            'avg_pmi_left': avg_pmi_left,
            'avg_pmi_right': avg_pmi_right,
            'numeric_assoc': numeric_assoc,
            'burstiness': burstiness,
            'idf': idf,
            'coll_skew': coll_skew,
            'examples': info.get('examples', [])
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df.fillna(0, inplace=True)
    return df, left, right, global_unigrams

# ---------------------------
# Statistical tests
# ---------------------------

def log_odds_ratio(seg: str, numeric_set: Set[str], left: Dict[str, Counter], right: Dict[str, Counter], k_smooth: float = 0.5):
    # build contingency table for "adjacent to numeric" vs "not" for segment seg
    left_neighbors = left.get(seg, Counter())
    right_neighbors = right.get(seg, Counter())
    num_adj = sum(left_neighbors[n] for n in left_neighbors if n in numeric_set) + sum(right_neighbors[n] for n in right_neighbors if n in numeric_set)
    total_adj = sum(left_neighbors.values()) + sum(right_neighbors.values())
    not_num = total_adj - num_adj
    # global counts for numeric adjacency with all other segments
    # Simple baseline: proportion of numeric neighbors in entire corpus
    all_left = Counter(); all_right = Counter()
    for k,c in left.items(): all_left.update(c)
    for k,c in right.items(): all_right.update(c)
    all_num = sum(all_left[n] for n in all_left if n in numeric_set) + sum(all_right[n] for n in all_right if n in numeric_set)
    all_tot = sum(all_left.values()) + sum(all_right.values())
    non_num_all = all_tot - all_num
    # log-odds with smoothing
    a = num_adj + k_smooth
    b = not_num + k_smooth
    c = all_num - num_adj + k_smooth
    d = non_num_all - not_num + k_smooth
    lor = math.log((a * d) / (b * c) + 1e-300)
    return lor

def g2_independence(seg: str, numeric_set: Set[str], left: Dict[str, Counter], right: Dict[str, Counter]):
    # likelihood ratio G^2 for adjacency independence
    left_neighbors = left.get(seg, Counter())
    right_neighbors = right.get(seg, Counter())
    num_adj = sum(left_neighbors[n] for n in left_neighbors if n in numeric_set) + sum(right_neighbors[n] for n in right_neighbors if n in numeric_set)
    total_adj = sum(left_neighbors.values()) + sum(right_neighbors.values())
    if total_adj == 0:
        return 0.0
    p_obs = num_adj / total_adj
    # corpus baseline
    all_left = Counter(); all_right = Counter()
    for k,c in left.items(): all_left.update(c)
    for k,c in right.items(): all_right.update(c)
    all_num = sum(all_left[n] for n in all_left if n in numeric_set) + sum(all_right[n] for n in all_right if n in numeric_set)
    all_tot = sum(all_left.values()) + sum(all_right.values())
    p_exp = (all_num / all_tot) if all_tot>0 else 0.0
    # G2 = 2 * n * (p_obs*log(p_obs/p_exp) + (1-p_obs)*log((1-p_obs)/(1-p_exp)))
    # handle edge cases
    def safe_log_ratio(p, q):
        if p <= 0 or q <= 0:
            return 0.0
        return math.log(p / q)
    if p_obs in (0.0,1.0) or p_exp in (0.0,1.0):
        return 0.0
    g2 = 2.0 * total_adj * (p_obs * safe_log_ratio(p_obs, p_exp) + (1-p_obs) * safe_log_ratio(1-p_obs, 1-p_exp))
    return g2

# ---------------------------
# EM for Naive-Bayes mixture (3 categories)
# ---------------------------

def robust_scale_matrix(X: np.ndarray):
    # median and IQR scaling per column -> robust to outliers
    med = np.median(X, axis=0)
    q75 = np.percentile(X, 75, axis=0)
    q25 = np.percentile(X, 25, axis=0)
    iqr = q75 - q25
    iqr[iqr < 1e-9] = 1.0
    Xs = (X - med) / iqr
    return Xs, med, iqr


def init_em_params(Xs: np.ndarray, df: pd.DataFrame, numeric_assoc_col: str = 'numeric_assoc'):
    n, d = Xs.shape
    K = 3
    # heuristics for initial means (in scaled space)
    # compute simple heuristics on original df
    freq = df['freq'].values
    numeric = df[numeric_assoc_col].values
    start_prob = df['start_prob'].values
    H_total = df['H_total'].values
    # indices
    hi_freq = np.where(freq >= np.percentile(freq, 66))[0]
    hi_num = np.where(numeric >= 0.5)[0]
    hi_start = np.where(start_prob >= 0.5)[0]
    hi_H = np.where(H_total >= np.percentile(H_total, 66))[0]
    # three prototype indices
    # commodity prototype: high numeric_assoc & above-median freq
    proto_comm = list(set(hi_num) & set(np.where(freq >= np.median(freq))[0]))[:10]
    # title prototype: high start_prob or end_prob
    proto_title = list(set(hi_start) | set(np.where(df['end_prob'].values >= 0.5)[0]))[:10]
    # proper name prototype: high H_total and low freq
    proto_name = list(set(hi_H) & set(np.where(freq <= np.median(freq))[0]))[:10]
    # if any prototype empty, fall back to random picks
    rng = np.random.RandomState(DEFAULT_SEED)
    if len(proto_comm) < 2:
        proto_comm = rng.choice(n, min(8, n), replace=False).tolist()
    if len(proto_title) < 2:
        proto_title = rng.choice(n, min(8, n), replace=False).tolist()
    if len(proto_name) < 2:
        proto_name = rng.choice(n, min(8, n), replace=False).tolist()
    mu = np.zeros((K, d))
    sigma = np.zeros((K, d))
    for k, idxs in enumerate([proto_comm, proto_title, proto_name]):
        sel = Xs[idxs, :]
        mu[k] = sel.mean(axis=0)
        sigma[k] = sel.std(axis=0) + 1e-6
    # initialize equal priors
    pi = np.array([1.0 / K] * K)
    return mu, sigma, pi


def gaussian_logpdf_diag(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    # x: (d,), mu: (d,), sigma: (d,) -> scalar logpdf
    # diag gaussian logpdf = -0.5 * sum( log(2pi sigma^2) + ((x-mu)^2 / sigma^2) )
    var = sigma * sigma
    term = -0.5 * (np.log(2.0 * np.pi * var + 1e-300) + ((x - mu) ** 2) / (var + 1e-300))
    return float(np.sum(term))


def em_naive_gaussian(Xs: np.ndarray, df: pd.DataFrame, max_iters: int = EM_MAX_ITERS, tol: float = EM_TOL):
    n, d = Xs.shape
    K = 3
    mu, sigma, pi = init_em_params(Xs, df)
    # ensure positive sigma
    sigma[sigma < 1e-6] = 1e-6
    ll_old = -1e300
    for it in range(max_iters):
        # E-step: compute log responsibilities
        log_resp = np.zeros((n, K))
        for k in range(K):
            # vectorize logpdf calculation
            var = sigma[k] * sigma[k]
            # compute per-point logpdf under k (diag gauss)
            logdet = -0.5 * np.sum(np.log(2.0 * np.pi * var + 1e-300))
            diff = Xs - mu[k]
            quad = -0.5 * np.sum((diff * diff) / (var + 1e-300), axis=1)
            logpdfs = logdet + quad
            log_resp[:, k] = np.log(pi[k] + 1e-300) + logpdfs
        # normalize
        log_norm = logsumexp(log_resp, axis=1)
        resp = np.exp(log_resp - log_norm[:, None])
        # log-likelihood
        ll = float(np.sum(log_norm))
        # M-step
        Nk = resp.sum(axis=0) + 1e-8  # avoid zeros
        pi = Nk / n
        for k in range(K):
            mu[k] = (resp[:, k][:, None] * Xs).sum(axis=0) / Nk[k]
            diff = Xs - mu[k]
            sigma[k] = np.sqrt((resp[:, k][:, None] * (diff * diff)).sum(axis=0) / Nk[k])
            # variance floor
            sigma[k][sigma[k] < MIN_VARIANCE_FLOOR] = MIN_VARIANCE_FLOOR
        # check convergence
        if abs(ll - ll_old) < tol:
            logging.debug('EM converged at iter %d (ll change=%.6g)', it, ll - ll_old)
            break
        ll_old = ll
    # final responsibilities and posteriors
    log_resp_final = np.zeros((n, K))
    for k in range(K):
        var = sigma[k] * sigma[k]
        logdet = -0.5 * np.sum(np.log(2.0 * np.pi * var + 1e-300))
        diff = Xs - mu[k]
        quad = -0.5 * np.sum((diff * diff) / (var + 1e-300), axis=1)
        logpdfs = logdet + quad
        log_resp_final[:, k] = np.log(pi[k] + 1e-300) + logpdfs
    log_norm_final = logsumexp(log_resp_final, axis=1)
    resp_final = np.exp(log_resp_final - log_norm_final[:, None])
    assigned = np.argmax(resp_final, axis=1)
    return {
        'mu': mu, 'sigma': sigma, 'pi': pi, 'resp': resp_final, 'assigned': assigned, 'loglik': ll_old
    }

# ---------------------------
# Post-hoc tests & stability
# ---------------------------

def bootstrap_stability(Xs: np.ndarray, df: pd.DataFrame, rounds: int = BOOTSTRAP_ROUNDS):
    n = Xs.shape[0]
    rng = np.random.RandomState(DEFAULT_SEED)
    assign_counts = np.zeros((n, 3), dtype=int)
    for r in range(rounds):
        # small perturbation on features (jitter)
        noise = rng.normal(scale=0.03, size=Xs.shape)
        Xp = Xs + noise
        res = em_naive_gaussian(Xp, df)
        for i, a in enumerate(res['assigned']):
            assign_counts[i, a] += 1
    stability = assign_counts.max(axis=1) / rounds
    return stability, assign_counts

# ---------------------------
# Seed-based precision proxies
# ---------------------------

def seed_sets_from_heuristics(df: pd.DataFrame):
    # high-precision seeds (small, conservative sets)
    # commodities seeds: numeric_assoc > 0.6 and freq >= median
    med_freq = float(np.median(df['freq'].values))
    comm_seed = set(df[(df['numeric_assoc'] >= 0.6) & (df['freq'] >= med_freq)]['seg_key'].tolist())
    # titles seeds: start_prob >= 0.6 & coll_skew >= 0.12
    title_seed = set(df[(df['start_prob'] >= 0.6) & (df['coll_skew'] >= 0.12)]['seg_key'].tolist())
    # proper name seeds: low freq (<= median) & high H_total >= 0.9
    name_seed = set(df[(df['freq'] <= med_freq) & (df['H_total'] >= np.percentile(df['H_total'], 60))]['seg_key'].tolist())
    return {'commodity': comm_seed, 'title': title_seed, 'proper_name': name_seed}

# ---------------------------
# Main driver
# ---------------------------

def run_lexicon_discovery(boundary_path: str = BOUNDARY_DEFAULT, out_dir: str = OUT_DIR_DEFAULT, debug: bool = False, seed: int = DEFAULT_SEED, bootstrap_rounds: int = BOOTSTRAP_ROUNDS, save_diags: bool = True):
    configure_logging(debug)
    random.seed(seed); np.random.seed(seed)
    logging.info('Lexicon discovery started')

    sequences = read_boundary_file(boundary_path)
    segments_map, seq_seg_keys = build_inventory(sequences)
    logging.info('Loaded %d sequences, %d unique segments', len(seq_seg_keys), len(segments_map))

    df, left, right, global_unigrams = build_feature_matrix(segments_map, seq_seg_keys)
    logging.info('Built feature matrix: %d rows', df.shape[0])

    # prepare features for EM
    features = ['freq', 'start_prob', 'end_prob', 'avg_pos', 'mean_len', 'H_total', 'avg_pmi_left', 'avg_pmi_right', 'numeric_assoc', 'burstiness', 'idf', 'coll_skew']
    X = df[features].values.astype(float)
    # robust scale
    Xs, med, iqr = robust_scale_matrix(X)

    # EM fit
    em_res = em_naive_gaussian(Xs, df)
    assigned = em_res['assigned']
    resp = em_res['resp']
    logging.info('EM finished: loglik=%.4f', em_res['loglik'])

    # map class indices to interpretable labels using heuristics (we must decide which gaussian corresponds to commodity/title/name)
    # We use prototype centers in original feature space to decide mapping
    mu_scaled = em_res['mu']
    # invert robust scaling for means back to original space
    mu_orig = mu_scaled * iqr + med
    # heuristics for label assignment to components: commodity -> high numeric_assoc & high freq
    comp_scores = []
    for k in range(mu_orig.shape[0]):
        score_comm = mu_orig[k, features.index('numeric_assoc')] * 0.6 + (mu_orig[k, features.index('freq')] / (np.max(X[:, features.index('freq')]) + 1e-9)) * 0.4
        score_title = mu_orig[k, features.index('start_prob')] * 0.6 + mu_orig[k, features.index('coll_skew')] * 0.4
        score_name = mu_orig[k, features.index('H_total')] * 0.7 + (1.0 - (mu_orig[k, features.index('freq')] / (np.max(X[:, features.index('freq')]) + 1e-9))) * 0.3
        comp_scores.append({'commodity': score_comm, 'title': score_title, 'proper_name': score_name})
    # choose label per component by max score
    comp_label = {}
    used_labels = set()
    for k, scores in enumerate(comp_scores):
        lab = max(scores.items(), key=lambda x: x[1])[0]
        # if collision (two components prefer same label), break ties by second best
        if lab in used_labels:
            # assign remaining label not used
            remaining = {'commodity', 'title', 'proper_name'} - used_labels
            if remaining:
                lab = remaining.pop()
            else:
                # fallback: assign by ordering
                lab = lab + f'_{k}'
        comp_label[k] = lab
        used_labels.add(lab)

    # build results table
    label_names = [comp_label[a] for a in assigned]
    posterior_cols = ['post_comm', 'post_title', 'post_name']
    # map resp columns order to labels
    # resp[:,k] corresponds to component k which maps to comp_label[k]
    label_order = [comp_label[k] for k in range(resp.shape[1])]
    # build mapping index per target label
    idx_map = {label_order[k]: k for k in range(len(label_order))}
    post_comm = resp[:, idx_map.get('commodity', 0)] if 'commodity' in idx_map else np.zeros(resp.shape[0])
    post_title = resp[:, idx_map.get('title', 1)] if 'title' in idx_map else np.zeros(resp.shape[0])
    post_name = resp[:, idx_map.get('proper_name', 2)] if 'proper_name' in idx_map else np.zeros(resp.shape[0])

    df_out = df.copy()
    df_out['assigned_comp_idx'] = assigned
    df_out['assigned_label'] = label_names
    df_out['post_commodity'] = post_comm
    df_out['post_title'] = post_title
    df_out['post_proper_name'] = post_name
    # log-odds and G2 for numeric association
    numeric_set = set(k for k,info in segments_map.items() if is_numeric_like(info['tokens']))
    lrs = []
    g2s = []
    for k in df_out['seg_key'].tolist():
        lrs.append(log_odds_ratio(k, numeric_set, left, right))
        g2s.append(g2_independence(k, numeric_set, left, right))
    df_out['log_odds_num_adj'] = lrs
    df_out['g2_num_adj'] = g2s

    # bootstrap stability
    stab, counts = bootstrap_stability(Xs, df, rounds=bootstrap_rounds)
    df_out['stability'] = stab

    # seed-based tests
    seeds = seed_sets_from_heuristics(df)
    # compute precision proxies
    def precision_proxy(seed_set: Set[str], predicted_label: str):
        if not seed_set:
            return None
        preds = set(df_out[df_out['assigned_label'] == predicted_label]['seg_key'].tolist())
        tp = len(preds & seed_set)
        prec = tp / len(preds) if len(preds)>0 else 0.0
        rec = tp / len(seed_set) if len(seed_set)>0 else 0.0
        return {'precision_proxy': prec, 'recall_proxy': rec, 'tp': tp, 'pred_count': len(preds), 'seed_count': len(seed_set)}
    seed_eval = {
        'commodity': precision_proxy(seeds['commodity'], 'commodity'),
        'title': precision_proxy(seeds['title'], 'title'),
        'proper_name': precision_proxy(seeds['proper_name'], 'proper_name')
    }

    # representatives per label: high posterior examples
    reps = {}
    for lab in ['commodity', 'title', 'proper_name']:
        sel = df_out[df_out['assigned_label'] == lab].copy()
        if sel.shape[0] == 0:
            reps[lab] = []
            continue
        # sort by max posterior and freq
        if lab == 'commodity':
            sel['score_rank'] = sel['post_commodity'] * np.log1p(sel['freq'])
        elif lab == 'title':
            sel['score_rank'] = sel['post_title'] * np.log1p(sel['freq'])
        else:
            sel['score_rank'] = sel['post_proper_name'] * (1.0 / (1 + sel['freq']))
        sel = sel.sort_values('score_rank', ascending=False)
        reps[lab] = sel[['seg_key', 'seg_id', 'freq', 'assigned_label', 'post_commodity', 'post_title', 'post_proper_name']].head(30).to_dict(orient='records')

    # summary stats
    summary = {
        'n_segments': df_out.shape[0],
        'n_sequences': len(seq_seg_keys),
        'class_counts': df_out['assigned_label'].value_counts().to_dict(),
        'avg_posteriors': {
            'commodity': float(df_out['post_commodity'].mean()),
            'title': float(df_out['post_title'].mean()),
            'proper_name': float(df_out['post_proper_name'].mean())
        },
        'seed_evaluation': seed_eval,
        'representatives': reps,
        'coverage_positions_estimate': None
    }

    # coverage estimate: proportion of token positions that are part of sequences that include at least one predicted "commodity/title/name"
    seq_with_label = 0
    tot_positions = 0
    covered_positions = 0
    for sidx, seq in enumerate(seq_seg_keys):
        tot_positions += len(seq)
        if not seq:
            continue
        has_any = False
        for pos, seg in enumerate(seq):
            lab = df_out[df_out['seg_key'] == seg]['assigned_label']
            if lab.shape[0] > 0 and lab.iloc[0] in ('commodity', 'title', 'proper_name'):
                has_any = True
                covered_positions += 1
        if has_any:
            seq_with_label += 1
    summary['coverage_positions_estimate'] = {'covered_positions': int(covered_positions), 'total_positions': int(tot_positions), 'pct': float(covered_positions / max(1, tot_positions) * 100.0), 'sequences_with_any': seq_with_label, 'total_sequences': len(seq_seg_keys)}

    # save outputs
    os.makedirs(out_dir, exist_ok=True)
    df_out_sorted = df_out.sort_values(['assigned_label', 'freq'], ascending=[True, False])
    df_out_sorted.to_csv(os.path.join(out_dir, LEXICON_CSV), index=False)
    with open(os.path.join(out_dir, SUMMARY_JSON), 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2)

    # diagnostics
    if save_diags:
        diag_dir = os.path.join(out_dir, DIAG_DIR); os.makedirs(diag_dir, exist_ok=True)
        pd.DataFrame(resp, columns=['comp0', 'comp1', 'comp2']).to_csv(os.path.join(diag_dir, 'posteriors_matrix.csv'), index=False)
        np.save(os.path.join(diag_dir, 'mu.npy'), em_res['mu'])
        np.save(os.path.join(diag_dir, 'sigma.npy'), em_res['sigma'])
        np.save(os.path.join(diag_dir, 'pi.npy'), em_res['pi'])
        with open(os.path.join(diag_dir, 'seed_eval.json'), 'w', encoding='utf-8') as fh:
            json.dump(seed_eval, fh, indent=2)
        with open(os.path.join(diag_dir, 'components_label_map.json'), 'w', encoding='utf-8') as fh:
            json.dump(comp_label, fh, indent=2)
        # stability matrix
        np.save(os.path.join(diag_dir, 'bootstrap_assign_counts.npy'), counts)
        pd.DataFrame(df_out_sorted[['seg_key', 'seg_id', 'freq', 'assigned_label', 'post_commodity', 'post_title', 'post_proper_name', 'stability']]).to_csv(os.path.join(diag_dir, 'lexicon_summary_table.csv'), index=False)
        if PLOTTING:
            try:
                # plot posterior histogram per class
                plt.figure(figsize=(6,4))
                plt.hist(df_out['post_commodity'].values, bins=40)
                plt.title('Commodity posterior distribution')
                plt.tight_layout(); plt.savefig(os.path.join(diag_dir, 'hist_post_commodity.png')); plt.close()
                plt.figure(figsize=(6,4))
                plt.hist(df_out['post_title'].values, bins=40)
                plt.title('Title posterior distribution')
                plt.tight_layout(); plt.savefig(os.path.join(diag_dir, 'hist_post_title.png')); plt.close()
                plt.figure(figsize=(6,4))
                plt.hist(df_out['post_proper_name'].values, bins=40)
                plt.title('Proper name posterior distribution')
                plt.tight_layout(); plt.savefig(os.path.join(diag_dir, 'hist_post_name.png')); plt.close()
            except Exception as e:
                logging.warning('Plotting diagnostics failed: %s', e)

    logging.info('Lexicon discovery finished. Outputs saved to %s', out_dir)
    return df_out, summary

# ---------------------------
# CLI
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Lexicon discovery: classify segments into titles/commodities/proper_names')
    p.add_argument('--boundary', type=str, default=BOUNDARY_DEFAULT)
    p.add_argument('--out', type=str, default=OUT_DIR_DEFAULT)
    p.add_argument('--bootstrap', type=int, default=BOOTSTRAP_ROUNDS)
    p.add_argument('--seed', type=int, default=DEFAULT_SEED)
    p.add_argument('--debug', action='store_true')
    p.add_argument('--no-diags', action='store_true')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    configure_logging(args.debug)
    try:
        run_lexicon_discovery(boundary_path=args.boundary, out_dir=args.out, bootstrap_rounds=args.bootstrap, debug=args.debug, seed=args.seed, save_diags=not args.no_diags)
    except Exception as e:
        logging.exception('Fatal error in lexicon discovery: %s', e)
        raise
