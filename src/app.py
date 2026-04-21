import pandas as pd
import numpy as np
import random
import streamlit as st
import warnings
warnings.filterwarnings("ignore")
from bandit import ContextualBandit

WARMUP_THRESHOLD = 5
st.set_page_config(
    page_title='Product Recommender',
    layout='wide'
)

@st.cache_data
def load_data():
    products=pd.read_csv('output/products.csv')
    interactions=pd.read_csv('output/interactions.csv')
    return products



def get_bandit() :
    if 'bandit' not in st.session_state:
        # st.write('creating new bandit..')
        st.session_state['bandit'] = ContextualBandit(lmdba=0.5)

    return st.session_state.bandit

def get_warmup_clicks():
    if 'warmup_clicks' not in st.session_state:
        st.session_state['warmup_clicks'] = 0
    return st.session_state['warmup_clicks']

def get_warmup_products(products, n=5):
    if 'warmup_products' not in st.session_state:
        st.session_state['warmup_products'] = (
            products.sample(n, random_state=42).reset_index(drop=True)
        )
    return st.session_state['warmup_products']

def increment_warmup():
    st.session_state['warmup_clicks'] = get_warmup_clicks() + 1

def is_warmed_up():
    return get_warmup_clicks() >= WARMUP_THRESHOLD

def get_random_products(products, n=5):
    return products.sample(n).reset_index(drop=True)

# def is_pretrained():
#     return st.session_state.get('pretrained', False)

# def mark_pretrained():
#     st.session_state.pretrained = True

# def maybe_pretrain(bandit, interactions, products):
#     if is_pretrained():
#         return
#     with st.spinner('pretraining..'):
#         bandit.offline_train(interactions, products, top_k=5)
#     mark_pretrained()

def get_top5(bandit, context, products):
    all_products = products.to_dict('records')
    _, prob, scores = bandit.predict(context, all_products)
    results = products.copy()
    results['score'] = scores
    top5=results.nlargest(5,'score').reset_index(drop=True)
    return top5, all_products

def format_name(name_words):
    return str(name_words)



def record_feedback(bandit, context, all_products, chosen_asin, reward):
    product_lookup = {p['ASIN']: p for p in all_products}
    all_asins = list(product_lookup.keys())

    if chosen_asin not in product_lookup:
        return
    
    others = [a for a in all_asins if a != chosen_asin]
    sampled = random.sample(others, min(4,len(others)))
    arm_products =[product_lookup[a] for a in [chosen_asin] + sampled]
    prob=1.0/len(arm_products)

    bandit.learn(context, arm_products,chosen_index=0, reward=reward, prob=prob)

def show_product_row(row, bandit, context, all_products, phase):
    col_img, col_info, col_feedback = st.columns([1,3,1])
    with col_img:
        if pd.notna(row.get('image')):
            st.image(row['image'], width='stretch')
        else:
            st.write('No image')

    with col_info:
        st.write(f"***{format_name(row['name_words'])}**")
        st.write(f"Category: {row['category']}")
        st.write(f"Avg Rating: {row['avg_rating']:.1f}  | Reviews {int(row['review_count'])}")
        # if phase == 'recommend':
            # st.write(f"Score: {row['score']}")

    with col_feedback:
        st.write("Feedback")
        if st.button("Satisfied", key=f"sat_{phase}_{row['ASIN']}"):
            record_feedback(
                bandit, context, all_products, 
                chosen_asin=row['ASIN'], reward=1.0
            )
            increment_warmup()
            st.rerun()

        if st.button("Not Satisfied", key=f"not_{phase}_{row['ASIN']}"):
            record_feedback(
                bandit, context, all_products, 
                chosen_asin=row['ASIN'], reward=0.0
            )
            increment_warmup()
            st.rerun()
        pass
    st.divider()


def main():
    st.title('Product recommender')

    products = load_data()
    all_products = products.to_dict('records')
    bandit = get_bandit()
    warmup_clicks = get_warmup_clicks()

    # maybe_pretrain(bandit=bandit, interactions=interactions, products=products)
    st.sidebar.header('User Context')

    location=st.sidebar.selectbox("Location", sorted(products['location'].dropna().unique()) if 'locatoin' in products.columns else 'United States')
    time_of_day=st.sidebar.selectbox("Time of day", ['morning', 'afternoon','evening','night'])
    device=st.sidebar.selectbox("Device", ['mobile', 'desktop','table'])
    intent=st.sidebar.selectbox("Intent", ['buying', 'browsing'])
    is_weekend=st.sidebar.toggle("Weekend", value=False)

    context = {
        "location": location,
        "time_of_day": time_of_day,
        "device": device,
        "intent": intent,
        "is_weekend": int(is_weekend),
    }
    

    st.sidebar.divider()
    st.sidebar.metric("Total Interactions", bandit.total_interactions)
    # st.sidebar.metric("Rate", round(bandit.rate, 6))
    
    if st.sidebar.button('Reset Model'):
        st.session_state.pop('bandit', None)
        st.session_state.pop('warmup_clicks', None)
        st.session_state.pop('warmup_products', None)
        st.rerun()

    if not is_warmed_up():
        remaining = WARMUP_THRESHOLD - warmup_clicks
        st.subheader('getting to know you')
        st.progress(warmup_clicks/WARMUP_THRESHOLD)

        random_products = get_warmup_products(products, n=5)
        for _, row in random_products.iterrows():
            show_product_row(row, bandit, context, all_products, phase='warmup')
    else:
        st.subheader('top 5 recommendations')
        top5, _ = get_top5(bandit, context, products)
        for _, row in top5.iterrows():
            show_product_row(row, bandit, context, all_products, phase='recommend')




if __name__ == '__main__':
    main()