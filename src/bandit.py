import pandas as pd
import numpy as np
import random
import vowpalwabbit 
import re
from vowpalwabbit import pyvw


def _clean(text):
    return re.sub(r"[|:\s]+","_", str(text).lower())

def build_shared_context(context):
    location = _clean(context.get('location','unknown'))
    time_of_day = _clean(context.get('time_of_day','unknown'))
    device = _clean(context.get('device','unknown'))
    intent = _clean(context.get('intent','unknown'))
    is_weekend = _clean(context.get('is_weekend','unknown'))
    return (
        f"shared |user "
        f"location_{location}=1 "
        f"time_{time_of_day}=1 "
        f"device_{device}=1 "
        f"intent_{intent}=1 "
        f"is_weekend={is_weekend}"
    )

def build_arm_line(arm_index, product, cost=None, prob=None):
    category = _clean(product.get('category','unknown'))
    rating = round(float(product.get('avg_rating', 3.0)), 2)
    log_count = round(float(product.get('log_review_count', 1.0)), 3)
    words = product.get('name_words','')
    label = f"{arm_index}:{cost}:{prob}" if cost is not None else str(arm_index)

    return (
        f"{label} |item "
        f"cat_{category}=1 "
        f"rating_{rating} "
        f"log_count={log_count} "
        f"{words}"
    ).strip()

def build_vw_example(context, products, chosen_index=None, cost=None, prob=None):
    lines = [build_shared_context(context)]
    for i, product in enumerate(products):
        if chosen_index is not None and i == chosen_index:
            lines.append(build_arm_line(i,product=product, cost=cost, prob=prob))
        else:
            lines.append(build_arm_line(i, product))
    return '\n'.join(lines)

def sample_pmf(scores):    
    r=random.random()
    cummulative=0.0
    for index, prob in enumerate(scores):
        cummulative += prob
        if r <= cummulative:
            return index, prob
    return len(scores)-1, scores[-1]


class ContextualBandit:
    def __init__(self, lmdba=0.5):
        self.lmdba = lmdba
        self.model = pyvw.Workspace(
            f"--cb_explore_adf --softmax --lambda {lmdba} -q sa --quiet"
        )
        self.total_interactions = 0
        self.total_reward = 0

    def predict(self, context, products):
        example = build_vw_example(context,products)
        scores = self.model.predict(example)
        chosen_index, prob = sample_pmf(scores)
        return chosen_index, prob, scores
    
    def learn(self, context, products,chosen_index, reward, prob):
        cost = -reward
        example = build_vw_example(context, products, chosen_index, cost,prob)
        self.model.learn(example)
        self.total_interactions +=1
        self.total_reward += reward

    @property
    def rate(self):
        if self.total_interactions == 0:
            return 0.0
        return self.total_reward/self.total_interactions

    def offline_train(self, interactions, products_df, top_k=5):
        product_lookup = products_df.set_index('ASIN').to_dict('index')
        all_asins = list(product_lookup.keys())

        category_to_asins = {}
        for asin, product in product_lookup.items():
            cat = product.get('category','unknown')
            category_to_asins.setdefault(cat, []).append(asin)

        all_categories = list(category_to_asins.keys())

        for _, row in interactions.iterrows():
            chosen_asin = row['ASIN']
            if chosen_asin not in product_lookup:
                continue

            context = {
                'location': row['reviewLocation'],
                'time_of_day': row['time_of_day'],
                'device': row['device'],
                'intent': row['intent'],
                'is_weekend': row['is_weekend'],
            }

            chosen_category = product_lookup[chosen_asin].get('category','unknown')
            other_categories = [c for c in all_categories if c != chosen_category]

            sampled_cats = np.random.choice(
                other_categories, size=min(top_k-1, len(other_categories)), replace=False
            )

            other_asins = [
                np.random.choice(category_to_asins[cat]) for cat in sampled_cats
            ]

            # others = [a for a in all_asins if a != chosen_asin]

            # sampled = np.random.choice(others, size=min(top_k-1, len(others)), replace=False).tolist()
            arm_asins = [chosen_asin] +other_asins
            products = [product_lookup[a] for a in arm_asins]

            prob = 1.0/len(products)

            self.learn(context,products,chosen_index=0, reward=float(row['reward']), prob=prob)
