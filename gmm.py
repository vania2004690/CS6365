import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import ndcg_score, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

from data_frame import workouts_df, user_logs


workouts_df = workouts_df.rename(columns={
    'Workout ID': 'workout_id',
    'Workout Type': 'type',
    'Workout Duration (mins)': 'duration_min',
    'Workout Intensity': 'intensity',
    'Calories Burned': 'est_calories',
    'Duration Bucket': 'duration_bucket'
})

user_logs = user_logs.rename(columns={
    'User ID': 'user_id',
    'Workout ID': 'workout_id',
    'Session Index': 'session_index'
})

cat_cols = ['type', 'intensity', 'duration_bucket']
num_cols = ['duration_min', 'est_calories']

ohe = OneHotEncoder(sparse_output=False)
cat_feats = ohe.fit_transform(workouts_df[cat_cols])

scaler = StandardScaler()
num_feats = scaler.fit_transform(workouts_df[num_cols])

features = np.hstack([num_feats, cat_feats])

pca = PCA(n_components=5)
reduced_features = pca.fit_transform(features)
workouts_reduced_df = pd.DataFrame(reduced_features, columns=[f'pc{i+1}' for i in range(5)])
workouts_reduced_df['workout_id'] = workouts_df['workout_id']

logs_df = user_logs.copy()
train_logs, test_logs = [], []
for uid, group in logs_df.groupby('user_id'):
    group_sorted = group.sort_values('session_index').reset_index(drop=True)
    if len(group_sorted) <= 2:
        train = group_sorted
        test = pd.DataFrame(columns=group_sorted.columns)
    else:
        train = group_sorted.iloc[:-2]
        test = group_sorted.iloc[-2:]
    train_logs.append(train)
    test_logs.append(test)
train_df = pd.concat(train_logs, ignore_index=True)
test_df = pd.concat(test_logs, ignore_index=True)


workout_matrix = workouts_reduced_df.set_index('workout_id').values[:, :-1]
workout_ids = workouts_reduced_df['workout_id'].values
wid_to_idx = {wid: idx for idx, wid in enumerate(workout_ids)}
wid_to_vec = {wid: vec for wid, vec in zip(workout_ids, workout_matrix)}

user_profiles = {}
for uid, group in train_df.groupby('user_id'):
    done_wids = group['workout_id'].values
    indices = [wid_to_idx[wid] for wid in done_wids if wid in wid_to_idx]
    if indices:
        vecs = workout_matrix[indices]
        user_profiles[uid] = vecs.mean(axis=0)

user_profile_matrix = np.vstack(list(user_profiles.values()))
user_ids_list = list(user_profiles.keys())


n_components = 3
gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
gmm.fit(user_profile_matrix)

user_cluster_probs = gmm.predict_proba(user_profile_matrix)
user_prob_map = {uid: probs for uid, probs in zip(user_ids_list, user_cluster_probs)}

hard_assignments = gmm.predict(user_profile_matrix)
hard_assignment_map = {uid: c for uid, c in zip(user_ids_list, hard_assignments)}

cluster_workouts_map = {}
for cluster_id in range(n_components):
    cluster_users = [u for u, c in hard_assignment_map.items() if c == cluster_id]
    cluster_workouts_map[cluster_id] = set(train_df[train_df['user_id'].isin(cluster_users)]['workout_id'].unique())

def recommend_gmm(uid, K=10):
    if uid not in user_profiles:
        return []
    
    profile = user_profiles[uid]
    probs = user_prob_map[uid] 
    
    sims = cosine_similarity(profile.reshape(1, -1), workout_matrix).flatten()
    
    candidates = []
    for i, wid in enumerate(workout_ids):
        base_score = sims[i]
        
        boost = 0
        for cluster_id in range(n_components):
            if wid in cluster_workouts_map[cluster_id]:
                boost += probs[cluster_id] * 0.05 
        
        final_score = base_score + boost
        candidates.append((int(wid), float(final_score)))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:K]

def evaluate_ranking(k=10):
    test_users = test_df['user_id'].unique()
    if len(test_users) > 1000:
        np.random.seed(42)
        test_users = np.random.choice(test_users, size=1000, replace=False)
    
    precisions, recalls, ndcgs = [], [], []
    
    for uid in test_users:
        recs = recommend_gmm(uid, K=k)
        if not recs:
            continue
        
        rec_wids = [wid for wid, _ in recs]
        rec_types = workouts_df[workouts_df['workout_id'].isin(rec_wids)]['type'].tolist()
        
        true_wids = list(test_df[test_df['user_id'] == uid]['workout_id'].unique())
        true_types = workouts_df[workouts_df['workout_id'].isin(true_wids)]['type'].tolist()
        
        if len(true_types) == 0:
            continue
        
        relevance = [1 if rec_type in true_types else 0 for rec_type in rec_types]
        prec = sum(relevance) / k
        rec = sum(relevance) / len(true_types)
        scores = [s for _, s in recs]
        ndcg_val = ndcg_score([relevance], [scores]) if any(relevance) else 0
        
        precisions.append(prec)
        recalls.append(rec)
        ndcgs.append(ndcg_val)
    
    return np.mean(precisions), np.mean(recalls), np.mean(ndcgs)

def evaluate_classification():
    print("\nClassification Metrics (Real vs Random)")
    
    y_true = []
    y_scores = []
    all_wids = list(workouts_df['workout_id'].values)

    test_users = test_df['user_id'].unique()
    if len(test_users) > 500:
        np.random.seed(42)
        test_users = np.random.choice(test_users, size=500, replace=False)

    for uid in test_users:
        if uid not in user_profiles: continue
        
        real_wids = test_df[test_df['user_id'] == uid]['workout_id'].unique()
        profile = user_profiles[uid].reshape(1, -1)
        probs = user_prob_map[uid]

        for wid in real_wids:
            if wid not in wid_to_vec: continue
            
            w_vec = wid_to_vec[wid].reshape(1, -1)
            cosine = cosine_similarity(profile, w_vec)[0][0]
            
            boost = 0
            for c in range(n_components):
                if wid in cluster_workouts_map[c]:
                    boost += probs[c] * 0.05
            
            y_true.append(1)
            y_scores.append(cosine + boost)

            neg_wid = random.choice(all_wids)
            while neg_wid in real_wids:
                neg_wid = random.choice(all_wids)
                
            neg_vec = wid_to_vec[neg_wid].reshape(1, -1)
            neg_cosine = cosine_similarity(profile, neg_vec)[0][0]
            
            neg_boost = 0
            for c in range(n_components):
                if neg_wid in cluster_workouts_map[c]:
                    neg_boost += probs[c] * 0.05
            
            y_true.append(0)
            y_scores.append(neg_cosine + neg_boost)

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    roc_auc = roc_auc_score(y_true, y_scores)
    
    best_f1 = 0
    best_thresh = 0
    thresholds = np.linspace(y_scores.min(), y_scores.max(), 100)
    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        if sum(y_pred) == 0: continue
        score = f1_score(y_true, y_pred)
        if score > best_f1:
            best_f1 = score
            best_thresh = t
            
    y_pred_optimal = (y_scores >= best_thresh).astype(int)

    print(f"accuracy: {accuracy_score(y_true, y_pred_optimal):.4f}")
    print(f"precision: {precision_score(y_true, y_pred_optimal):.4f}")
    print(f"recall: {recall_score(y_true, y_pred_optimal):.4f}")
    print(f"f1: {best_f1:.4f}")
    print(f"roc_auc: {roc_auc:.4f}")

if __name__ == "__main__":
    
    print("\n[Ranking Metrics]")
    k_vals = [3, 5, 10]
    for k in k_vals:
        p, r, n = evaluate_ranking(k)
        print(f"K={k} -> Precision: {p:.4f}, Recall: {r:.4f}, NDCG: {n:.4f}")
        
    evaluate_classification()
