import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import ndcg_score
import random

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

#feature encoding
cat_cols = ['type', 'intensity', 'duration_bucket']
num_cols = ['duration_min', 'est_calories']

ohe = OneHotEncoder(sparse_output=False)
cat_feats = ohe.fit_transform(workouts_df[cat_cols])

scaler = StandardScaler()
num_feats = scaler.fit_transform(workouts_df[num_cols])

features = np.hstack([num_feats, cat_feats])

#PCA
pca = PCA(n_components=5)
reduced_features = pca.fit_transform(features)
workouts_reduced_df = pd.DataFrame(reduced_features, columns=[f'pc{i+1}' for i in range(5)])
workouts_reduced_df['workout_id'] = workouts_df['workout_id']

logs_df = user_logs.copy()

#training/test data
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

#user profiles
workout_matrix = workouts_reduced_df.set_index('workout_id').values[:, :-1]
workout_ids = workouts_reduced_df['workout_id'].values
wid_to_idx = {wid: idx for idx, wid in enumerate(workout_ids)}

user_profiles = {}
for uid, group in train_df.groupby('user_id'):
    done_wids = group['workout_id'].values
    indices = [wid_to_idx[wid] for wid in done_wids if wid in wid_to_idx]
    if indices:
        vecs = workout_matrix[indices]
        user_profiles[uid] = vecs.mean(axis=0)

user_profile_matrix = np.vstack(list(user_profiles.values()))

#kmeans clustering
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
user_clusters = kmeans.fit_predict(user_profile_matrix)
cluster_map = {uid: int(cluster) for uid, cluster in zip(user_profiles.keys(), user_clusters)}

#cluster workouts
cluster_workouts_map = {}
for cluster_id in range(n_clusters):
    cluster_users = [u for u, c in cluster_map.items() if c == cluster_id]
    cluster_workouts_map[cluster_id] = set(train_df[train_df['user_id'].isin(cluster_users)]['workout_id'].unique())

#recommended workouts
def recommend_for_user(uid, K=10):
    if uid not in user_profiles:
        return []
    profile = user_profiles[uid]
    cluster_id = cluster_map[uid]
    cluster_workouts = cluster_workouts_map[cluster_id]

    #cosine similarity for all workouts
    sims = cosine_similarity(profile.reshape(1, -1), workout_matrix).flatten()
    candidates = []
    for wid, sim in zip(workout_ids, sims):
        score = sim + 0.05 if wid in cluster_workouts else sim
        candidates.append((int(wid), float(score)))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:K]

#evaluation metrics - based on workout TYPE instead of ID
def precision_recall_ndcg_at_k(k=10):
    test_users = test_df['user_id'].unique()
    # Sample 1000 users for faster evaluation
    if len(test_users) > 1000:
        np.random.seed(42)
        test_users = np.random.choice(test_users, size=1000, replace=False)
    
    precisions, recalls, ndcgs = [], [], []
    
    for uid in test_users:
        recs = recommend_for_user(uid, K=k)
        if not recs:
            continue
        
        rec_wids = [wid for wid, _ in recs]
        rec_types = workouts_df[workouts_df['workout_id'].isin(rec_wids)]['type'].tolist()
        
        true_wids = list(test_df[test_df['user_id'] == uid]['workout_id'].unique())
        true_types = workouts_df[workouts_df['workout_id'].isin(true_wids)]['type'].tolist()
        
        if len(true_types) == 0:
            continue
        
        # Binary relevance based on workout TYPE match
        relevance = [1 if rec_type in true_types else 0 for rec_type in rec_types]
        prec = sum(relevance) / k
        rec = sum(relevance) / len(true_types)
        scores = [s for _, s in recs]
        ndcg_val = ndcg_score([relevance], [scores]) if any(relevance) else 0
        
        precisions.append(prec)
        recalls.append(rec)
        ndcgs.append(ndcg_val)
    
    if not precisions:
        return {
            'Precision@K': 0.0,
            'Recall@K': 0.0,
            'NDCG@K': 0.0,
            'Users_evaluated': 0
        }
    return {
        'Precision@K': np.mean(precisions),
        'Recall@K': np.mean(recalls),
        'NDCG@K': np.mean(ndcgs),
        'Users_evaluated': len(precisions)
    }

metrics = {
    'k=3': precision_recall_ndcg_at_k(3),
    'k=5': precision_recall_ndcg_at_k(5),
    'k=10': precision_recall_ndcg_at_k(10)
}
metrics_df = pd.DataFrame(metrics).T
print("\nEvaluation metrics:\n")
print(metrics_df)
