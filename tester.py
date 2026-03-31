#!/usr/bin/env python3
import os
import random
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import ndcg_score
import matplotlib.pyplot as plt

# ---------------- USER CONFIG ----------------
CSV_PATH = "workout_fitness_tracker_data.csv"

MIN_USER_INTERACTIONS = 1   # kept conservative; script adapts automatically
MIN_ITEM_INTERACTIONS = 3

MAX_USERS = 2000
MAX_ITEMS = 1000
LEAVE_LAST = 1              # if users have >1 interactions we use leave-last
COLD_START_TEST_FRAC = 0.20 # fraction of users to hold out when each user has only a single interaction

ALS_FACTORS = 64
ALS_ITERS = 15
BPR_FACTORS = 64
BPR_ITERS = 50
# ---------------------------------------------

df = pd.read_csv(CSV_PATH)
df.columns = [c.strip() for c in df.columns]

expected_cols = ['User ID', 'Workout Type', 'Workout Duration (mins)', 'Calories Burned', 'Workout Intensity']
for c in expected_cols:
    if c not in df.columns:
        raise ValueError(f"Expected column '{c}' in CSV (found {df.columns.tolist()})")

df = df[expected_cols].copy()

# ---------- DURATION BUCKET ----------
def get_duration_bucket(d):
    try:
        d = float(d)
    except:
        d = 0.0
    if d <= 30:  return '0-30'
    if d <= 60:  return '31-60'
    if d <= 90:  return '61-90'
    return '91+'

df['Duration Bucket'] = df['Workout Duration (mins)'].apply(get_duration_bucket)

# ---------- COARSE ITEM DEFINITION ----------
workouts_df = (
    df.drop_duplicates(subset=['Workout Type', 'Duration Bucket', 'Workout Intensity'])
      .reset_index(drop=True)
)

workouts_df = workouts_df.rename(columns={
    'Workout Type': 'name',
    'Duration Bucket': 'duration_bucket',
    'Workout Intensity': 'intensity'
})

workouts_df['workout_id'] = workouts_df.index

# ---------- MERGE BACK ----------
user_logs = df.merge(
    workouts_df[['name', 'duration_bucket', 'intensity', 'workout_id']],
    left_on=['Workout Type', 'Duration Bucket', 'Workout Intensity'],
    right_on=['name', 'duration_bucket', 'intensity'],
    how='left'
)

user_logs = user_logs.rename(columns={'User ID': 'user_id'})
user_logs = user_logs.sort_values('user_id').reset_index(drop=True)
user_logs['Session Index'] = user_logs.groupby('user_id').cumcount()

# ---------- STAT DIAGNOSTICS ----------
user_counts = user_logs['user_id'].value_counts()
item_counts = user_logs['workout_id'].value_counts()

print("Unique users:", user_logs['user_id'].nunique())
print("User counts ≥{}: {}".format(MIN_USER_INTERACTIONS, (user_counts >= MIN_USER_INTERACTIONS).sum()))
print("Unique items:", user_logs['workout_id'].nunique())
print("Item counts ≥{}: {}".format(MIN_ITEM_INTERACTIONS, (item_counts >= MIN_ITEM_INTERACTIONS).sum()))
print("Example item frequency:")
print(item_counts.head())

# ---------- FILTER ----------
active_users = user_counts[user_counts >= MIN_USER_INTERACTIONS].index.tolist()
popular_items = item_counts[item_counts >= MIN_ITEM_INTERACTIONS].index.tolist()

active_users = active_users[:MAX_USERS]
popular_items = popular_items[:MAX_ITEMS]

filtered_logs = user_logs[
    user_logs['user_id'].isin(active_users) &
    user_logs['workout_id'].isin(popular_items)
].copy()

if filtered_logs.empty:
    raise ValueError("No data after filtering; relax MIN thresholds or redefine item granularity.")

print(f"Filtered data: users={filtered_logs['user_id'].nunique()}, items={filtered_logs['workout_id'].nunique()}, interactions={len(filtered_logs)}")

# ---------- Decide splitting strategy ----------
per_user_counts = filtered_logs['user_id'].value_counts()
all_single_interaction = per_user_counts.max() == 1 and per_user_counts.min() == 1

if all_single_interaction:
    # Cold-start holdout: sample fraction of users as test users (their single interaction is the test)
    print("All users have exactly one interaction. Using cold-start holdout split.")
    user_list = sorted(filtered_logs['user_id'].unique().tolist())
    n_test_users = max(1, int(len(user_list) * COLD_START_TEST_FRAC))
    random.seed(42)
    test_users = set(random.sample(user_list, n_test_users))
    train_df = filtered_logs[~filtered_logs['user_id'].isin(test_users)].copy()
    test_df = filtered_logs[filtered_logs['user_id'].isin(test_users)].copy()
else:
    # Leave-last per-user (original behavior)
    print("Using leave-last per-user split for users with >1 interactions.")
    train_logs = []
    test_logs = []
    for uid, g in filtered_logs.groupby('user_id'):
        gs = g.sort_values('Session Index').reset_index(drop=True)
        if len(gs) <= LEAVE_LAST:
            train_logs.append(gs)
            test_logs.append(pd.DataFrame(columns=gs.columns))
        else:
            train_logs.append(gs.iloc[:-LEAVE_LAST])
            test_logs.append(gs.iloc[-LEAVE_LAST:])
    train_df = pd.concat(train_logs, ignore_index=True)
    test_df = pd.concat(test_logs, ignore_index=True)

# quick diagnostics
print("Train interactions:", len(train_df), "Test interactions:", len(test_df))
print("Train users:", train_df['user_id'].nunique(), "Test users:", test_df['user_id'].nunique())

# ---------- USER/ITEM INDEXING ----------
unique_users = sorted(train_df['user_id'].unique().tolist()
                      + list(set(test_df['user_id'].unique()) - set(train_df['user_id'].unique())))
user_to_idx = {u: i for i, u in enumerate(unique_users)}
idx_to_user = {i: u for u, i in user_to_idx.items()}

unique_items = sorted(filtered_logs['workout_id'].unique().tolist())
item_to_idx = {i: pos for pos, i in enumerate(unique_items)}
idx_to_item = {v: k for k, v in item_to_idx.items()}

n_users = len(unique_users)
n_items = len(unique_items)

print(f"Matrix size: n_users={n_users}, n_items={n_items}")

def build_interaction_matrix(df_inter):
    rows, cols, data = [], [], []
    for _, r in df_inter.iterrows():
        u = user_to_idx[r['user_id']]
        i = item_to_idx[int(r['workout_id'])]
        rows.append(u)
        cols.append(i)
        data.append(1.0)
    return csr_matrix((data, (rows, cols)), shape=(n_users, n_items))

train_mat = build_interaction_matrix(train_df)
test_mat = build_interaction_matrix(test_df)

# ---------- MODEL SETUP ----------
use_implicit = False
try:
    import implicit
    from implicit.als import AlternatingLeastSquares
    from implicit.bpr import BayesianPersonalizedRanking
    use_implicit = True
    print("Using implicit library.")
except Exception:
    print("implicit library not available; using fallback methods.")

# ---------- ALS ----------
if use_implicit:
    als = AlternatingLeastSquares(factors=ALS_FACTORS, regularization=0.1,
                                  iterations=ALS_ITERS, random_state=42)
    # implicit expects item-user matrix (items x users)
    als.fit(train_mat.T.tocsr())

    def als_recommend(user_idx, K=10):
        recs = als.recommend(user_idx, train_mat, N=K, filter_already_liked_items=True)
        return [(int(item_idx), float(score)) for item_idx, score in recs]

else:
    # SVD fallback (approximate)
    from scipy.sparse.linalg import svds
    # choose k safely: must be < min(n_users, n_items)
    max_k = min(40, n_items - 1, n_users - 1)
    if max_k < 1:
        max_k = 1
    k = max_k
    try:
        u, s, vt = svds(train_mat.asfptype(), k=k)
        u = u[:, ::-1]
        s = s[::-1]
        vt = vt[::-1, :]
        user_factors = u.dot(np.diag(np.sqrt(s)))
        item_factors = (np.diag(np.sqrt(s)).dot(vt)).T
    except Exception as e:
        # fallback to random factors if svds fails
        print("svds failed:", e)
        n_factors = min(32, n_items, n_users)
        user_factors = np.random.normal(scale=0.01, size=(n_users, n_factors))
        item_factors = np.random.normal(scale=0.01, size=(n_items, n_factors))

    def als_recommend(user_idx, K=10):
        scores = item_factors.dot(user_factors[user_idx])
        seen = set(train_mat[user_idx].nonzero()[1].tolist())
        candidates = [(i, float(scores[i])) for i in range(n_items) if i not in seen]
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:K]

# ---------- BPR ----------
if use_implicit:
    bpr = BayesianPersonalizedRanking(factors=BPR_FACTORS, learning_rate=0.01,
                                      regularization=0.01, iterations=BPR_ITERS)
    bpr.fit(train_mat.T.tocsr())

    def bpr_recommend(user_idx, K=10):
        recs = bpr.recommend(user_idx, train_mat, N=K, filter_already_liked_items=True)
        return [(int(item_idx), float(score)) for item_idx, score in recs]

else:
    # Simple BPR-ish training fallback (small factor count and epochs)
    n_factors = 32
    U = np.random.normal(scale=0.01, size=(n_users, n_factors))
    V = np.random.normal(scale=0.01, size=(n_items, n_factors))

    user_pos = {u: set(train_mat[u].nonzero()[1].tolist()) for u in range(n_users)}
    all_items = set(range(n_items))

    def sample_triplet():
        u = random.randrange(n_users)
        if len(user_pos[u]) == 0:
            return None
        i = random.choice(list(user_pos[u]))
        neg = list(all_items - user_pos[u])
        if not neg:
            return None
        j = random.choice(neg)
        return u, i, j

    epochs = 5
    lr = 0.05
    reg = 0.01

    for _ in range(epochs):
        for _ in range(1000):
            tup = sample_triplet()
            if not tup:
                continue
            u, i, j = tup
            xu, vi, vj = U[u], V[i], V[j]
            x = xu.dot(vi - vj)
            sig = 1 / (1 + np.exp(-x))
            grad_u = (1 - sig) * (vi - vj) - reg * xu
            grad_vi = (1 - sig) * xu - reg * vi
            grad_vj = -(1 - sig) * xu - reg * vj
            U[u] += lr * grad_u
            V[i] += lr * grad_vi
            V[j] += lr * grad_vj

    def bpr_recommend(user_idx, K=10):
        scores = V.dot(U[user_idx])
        seen = set(train_mat[user_idx].nonzero()[1].tolist())
        candidates = [(i, float(scores[i])) for i in range(n_items) if i not in seen]
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:K]

# ---------- METRICS ----------
def precision_recall_ndcg_at_k(recommender, k=10, users=None):
    precisions, recalls, ndcgs = [], [], []
    users = list(range(n_users)) if users is None else users
    users_evaluated = 0
    for u in users:
        true_items = test_mat[u].nonzero()[1].tolist()
        if len(true_items) == 0:
            continue
        recs = recommender(u, K=k)
        rec_items = [it for it, sc in recs]
        # pad to length k
        if len(rec_items) < k:
            rec_items += [-1] * (k - len(rec_items))
        rel = [1 if it in true_items else 0 for it in rec_items]
        scores = [sc for _, sc in recs] + [0.0] * (k - len(recs))
        # compute metrics for this user
        prec = sum(rel[:k]) / k
        rec = sum(rel[:k]) / max(1, len(true_items))
        # ndcg_score expects arrays (n_samples, n_candidates)
        try:
            ndcg_val = float(ndcg_score([rel], [scores])) if any(rel[:k]) else 0.0
        except Exception:
            ndcg_val = 0.0
        precisions.append(prec)
        recalls.append(rec)
        ndcgs.append(ndcg_val)
        users_evaluated += 1

    if users_evaluated == 0:
        return {
            'Precision@K': 0.0,
            'Recall@K': 0.0,
            'NDCG@K': 0.0,
            'Users_evaluated': 0
        }

    return {
        'Precision@K': float(np.mean(precisions)),
        'Recall@K': float(np.mean(recalls)),
        'NDCG@K': float(np.mean(ndcgs)),
        'Users_evaluated': users_evaluated
    }

ks = [3, 5, 10]
results = {}

for name, fn in [('ALS', als_recommend), ('BPR', bpr_recommend)]:
    model_res = {}
    print(f"\nEvaluating {name} ...")
    for k in ks:
        m = precision_recall_ndcg_at_k(fn, k=k)
        print(f"  {name} @ k={k}: {m}")
        model_res[f'k={k}'] = m
    results[name] = model_res

# ---------- EXAMPLE USER ----------
example_user_idx = 0
print("\nExample recommendations for user index 0:")
for name, fn in [('ALS', als_recommend), ('BPR', bpr_recommend)]:
    recs = fn(example_user_idx, K=10)
    mapped = []
    for item_idx, score in recs:
        wid = idx_to_item[item_idx]
        row = workouts_df[workouts_df['workout_id'] == wid].iloc[0].to_dict()
        mapped.append({
            'workout_id': wid,
            'name': row.get('name'),
            'duration_bucket': row.get('duration_bucket'),
            'intensity': row.get('intensity'),
            'score': score
        })
    print(f"\n{name} top recs:")
    print(pd.DataFrame(mapped))

# ---------- SAVE SUMMARY ----------
summary_rows = []
for model_name, model_res in results.items():
    for k_label, m in model_res.items():
        summary_rows.append({'model': model_name, 'k': k_label, **m})

summary_df = pd.DataFrame(summary_rows)
out_dir = "out"
os.makedirs(out_dir, exist_ok=True)
out_csv = os.path.join(out_dir, "recommender_evaluation_summary.csv")
summary_df.to_csv(out_csv, index=False)
print(f"\nSaved summary to {out_csv}")


def plot_recommender_metrics(summary_df: pd.DataFrame, output_dir: str):
    """Create per-model metric curves similar to metric_eval_graph.py."""
    metric_columns = ["Precision@K", "Recall@K", "NDCG@K"]
    for model_name in summary_df["model"].unique():
        model_df = summary_df[summary_df["model"] == model_name].copy()
        model_df["k_value"] = (
            model_df["k"].str.replace("k=", "", regex=False).astype(int)
        )
        model_df = model_df.sort_values("k_value")

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for metric, marker in zip(metric_columns, ["o", "s", "^"]):
            ax.plot(
                model_df["k_value"],
                model_df[metric],
                marker=marker,
                linewidth=2,
                markersize=8,
                label=f"{metric}",
            )

        ax.set_xlabel("K (Number of Recommendations)", fontsize=12)
        ax.set_ylabel("Metric Value", fontsize=12)
        ax.set_title(
            f"{model_name} Recommendation Metrics", fontsize=14, fontweight="bold"
        )
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(model_df["k_value"])
        plt.tight_layout()

        output_path = os.path.join(
            output_dir, f"{model_name.lower()}_metrics_visualization.png"
        )
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {model_name} visualization to {output_path}")


plot_recommender_metrics(summary_df, out_dir)
