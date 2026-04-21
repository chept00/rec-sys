import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle

import cornac
from cornac.models import SVD, WMF, PMF,CTR, CDL, BPR,EASE
from cornac.eval_methods import RatioSplit
from cornac.metrics import NDCG, Precision, Recall

DATA_PATH=r'C:\Users\dcheruiyot2\OneDrive - KPMG\Projects\rec-sys\dataset\utility\reviews.csv'

def load_data(path):
    df = pd.read_csv(path)
    df= df.drop(['Unnamed: 0'], axis=1)

    review_counts = df.groupby('reviewerID').size()
    active_users=review_counts[review_counts >= 3].index
    df = df[df['reviewerID'].isin(active_users)].copy()

    df['customerReview'] = df['customerReview'].astype(float)
    return df.reset_index(drop=True)

def build_dataset(df):
    df['feedback'] = (df['customerReview'] >= 4).astype(float)
    triplets = list(zip(
        df['reviewerID'].tolist(),
        df['ASIN'].tolist(),
        df['feedback'].tolist()
    ))

    return triplets

def train_model(triplets):
    eval_method = RatioSplit(
        data=triplets,
        test_size=0.2,
        rating_threshold=1.0,
        seed=42,
        verbose=True
    )

    wmf_model=WMF(
        k=100,
        max_iter=50,
        learning_rate=0.001,
        lambda_u=0.01,
        lambda_v=0.01,
        a=1.0,
        b=0.001,
        seed=42,
        verbose=42,
    )

    wmf_model.fit(eval_method.train_set)
    return wmf_model,  eval_method

def recommend_for_user(user_id, model, eval_method, n=5):
    if user_id not in eval_method.train_set.uid_map:
        return []
    
    user_idx = eval_method.train_set.uid_map[user_id]
    scores = model.score(user_idx)
    known_items = set(eval_method.train_set.matrix[user_idx].nonzero()[1])

    item_scores = [
        (item_idx, score) for item_idx, score in enumerate(scores) if item_idx not in known_items
    ]

    item_scores.sort(key=lambda x: x[1], reverse=True)
    idx_to_iid = {v: k for k, v in eval_method.train_set.iid_map.items()}
    return [idx_to_iid[idx] for idx, _ in item_scores[:n] if idx in idx_to_iid]

def evaluate(wmf_model, eval_method, n=10):
    experiment = cornac.Experiment(
        eval_method=eval_method,
        models=[wmf_model],
        metrics=[
            Precision(k=n),
            Recall(k=n),
            NDCG(k=n)
        ],
        user_based=True
    )
    experiment.run()
    return experiment

if __name__ == "__main__":
    df = load_data(DATA_PATH)
    triplets = build_dataset(df)

    wmf_model,  eval_method = train_model(triplets)
    with open("models/wmf_model.pkl", "wb") as f:
        pickle.dump((wmf_model, eval_method),f)

    evaluate(wmf_model,  eval_method)

    test_user = df['reviewerID'].iloc[0]
    recs = recommend_for_user(test_user, wmf_model, eval_method)
    for asin in recs:
        name = df[df['ASIN'] == asin]['ProductName'].iloc[0]
        print(name)