import pandas as pd
import numpy as np
def load_data(path):
    df = pd.read_csv(path)
    return df

def filter_top_products(df, n):
    top = (df.groupby('ASIN')['reviewerID'].count().nlargest(n).index)
    filtered = df[df['ASIN'].isin(top)].copy()
    print(len(filtered))
    return filtered

def add_weekend(df):
    df['reviewDate'] = pd.to_datetime(df['reviewDate'])
    df['is_weekend'] = df['reviewDate'].dt.dayofweek.isin([5,6]).astype(int)
    return df

CATEGORY_CONTEXT_BIAS = {
    'kitchen':{'time_of_day':'morning', 'device':'mobile','intent':'browsing'},
    'personal care':{'time_of_day':'morning', 'device':'mobile','intent':'browsing'},
    'bathroom':{'time_of_day':'morning', 'device':'mobile','intent':'browsing'},
    'books':{'time_of_day':'evening', 'device':'tablet','intent':'browsing'},
    'bedroom':{'time_of_day':'evening', 'device':'desktop','intent':'browsing'},
    'living room':{'time_of_day':'evening', 'device':'desktop','intent':'browsing'},
    'children':{'time_of_day':'afternoon', 'device':'mobile','intent':'browsing'},
    'fashion':{'time_of_day':'afternoon', 'device':'mobile','intent':'browsing'},
    'travel essentials':{'time_of_day':'afternoon', 'device':'mobile','intent':'browsing'},
    'electronic devices':{'time_of_day':'afternoon', 'device':'desktop','intent':'buying'},
    'computer components':{'time_of_day':'afternoon', 'device':'desktop','intent':'buying'},
    'peripheral devices':{'time_of_day':'afternoon', 'device':'desktop','intent':'buying'},
    'office supplies':{'time_of_day':'afternoon', 'device':'desktop','intent':'buying'},
    'mobile accessories':{'time_of_day':'afternoon', 'device':'mobile','intent':'buying'},
    'cleaning material':{'time_of_day':'morning', 'device':'mobile','intent':'buying'},
    'unknown':{'time_of_day':'afternoon', 'device':'mobile','intent':'browsing'},
}
BIAS_STRENGTH = 0.7

def _assign_context_value(category, field, options, rng):
    cat_lower = str(category).lower().strip()
    bias = CATEGORY_CONTEXT_BIAS.get(cat_lower,{})
    biased_value =bias.get(field)

    if biased_value and rng.random() < BIAS_STRENGTH:
        return biased_value   
    return rng.choice(options)

def add_synthetic_context(df, seed=42):
    rng = np.random.default_rng(seed)
    time_options = ['morning','afternoon','evening','night']
    intent_options = ['browsing','buying']

    df['time_of_day'] = [
        _assign_context_value(cat,'time_of_day', time_options, rng)
        for cat in df['category']
    ]
    df['device'] = [
        _assign_context_value(cat,'device', time_options, rng)
        for cat in df['category']
    ]
    df['intent'] = [
        _assign_context_value(cat,'intent', time_options, rng)
        for cat in df['category']
    ]

    return df

 
def compute_reward(row):
    # 'Books', 'Unknown', 'Office Supplies', 'Personal Care',
    #    'Cleaning Material', 'Bedroom', 'Children', 'Living Room',
    #    'Fashion', 'Peripheral Devices', 'Kitchen', 'Computer Components',
    #    'Mobile Accessories', 'Travel Essentials', 'Electronic Devices',
    #    'Bathroom'
    category=str(row.get('category','')).lower().strip()
    time_of_day=str(row.get('time_of_day','')).lower().strip()
    device=str(row.get('device','')).lower().strip()
    intent=str(row.get('intent','')).lower().strip()
    is_weekend=str(row.get('is_weekend',0)).lower().strip()
    rating=str(row.get('customerReview',3)).lower().strip()

    if rating==5:
        logit=1.0
    elif rating==4:
        logit=0.0
    else:
        logit = -1.5

    if time_of_day == 'morning' and category in ('kitchen','personal care','bathroom'):
        logit += 2.5

    if time_of_day in ('evening','ight') and category in ('books', 'bedroom','living room'):
        logit += 2.0

    if is_weekend == 1 and category in ('children','fashion', 'living room'):
        logit += 2.0

    if device == 'mobile' and category in ('mobile accessories', 'fashion', 'personal care'):
        logit +=1.5

    if device == 'desktop' and category in ('computer components', 'peripheral devices','electronic devices'):
        logit + 1.5

    if intent == 'buying' and category in ('electronic devices','computer components','office supplies'):
        logit += 1.8

    if intent == 'browsing' and category in ('books', 'fashion', 'travel essentials'):
        logit += 1.5

    prob = 1/(1+np.exp(-logit))
    return int(np.random.random() < prob)

def add_reward(df):
    product_avg = df.groupby('ASIN')['customerReview'].transform('mean')
    product_median = df.groupby('ASIN')['customerReview'].transform('median')
    # df['reward'] = (df['customerReview']== 5).astype(int)
    df['reward'] = df.apply(compute_reward, axis=1)
    print(f"Reward rate: {df['reward'].mean()}")
    return df

def build_product_features(df):
    products = (df.groupby('ASIN').agg(
        ProductName=('ProductName', 'first'),
        category=('category', 'first'),
        image=('image', 'first'),
        avg_rating=('customerReview', 'mean'),
        review_count=('customerReview', 'count'),
    ).reset_index())

    products['avg_rating'] = products['avg_rating'].round(2)
    products['log_review_count'] = np.log1p(products['review_count']).round(3)

    products['name_words'] = (
        products['ProductName'].str.lower().
        str.replace('_',' ', regex=False).str.replace(r'[^a-z0-9]',' ', regex=True).str.strip()
        )
    print(f"Products: {len(products)}")
    return products

def build_interactions(df,products, seed=42):
    interactions = df[['reviewerID', 'ASIN','reviewLocation','is_weekend','time_of_day','device','intent','reward']].merge(
        products[['ASIN','category','avg_rating','log_review_count','name_words']], on='ASIN',how='left'
    )

    interactions = interactions.sample(frac=1, random_state=seed).reset_index(drop=True)
    return interactions

def save(products, interactions):
    products.to_csv('output/products.csv', index=False)
    interactions.to_csv('output/interactions.csv', index=False)

def main():
    file_path = r'C:\Users\dcheruiyot2\OneDrive - KPMG\Projects\rec-sys\dataset\utility\reviews.csv'
    df=load_data(file_path)
    df=filter_top_products(df,200)
    df=add_weekend(df)
    df=add_synthetic_context(df)
    df=add_reward(df)
    products = build_product_features(df)
    interactions = build_interactions(df, products)
    save(products, interactions)

if __name__=="__main__":
    main()

