"""
Explainable Interactive Visual Analytics Dashboard
Multimodal Analysis: Network + Text + Temporal
Course: COMP-834 Advanced Data Visualization — Assignment 3
PAK-AUSTRIA Fachhochschule, Spring 2026
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import warnings
import random
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & STYLES
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Visual Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
/* Dark background for the whole app */
.stApp { background-color: #0d0d1a; color: #e8e8f0; }
/* KPI cards */
.kpi-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #2a3a6a;
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
    text-align: center;
}
.kpi-value { font-size: 1.9rem; font-weight: 700; color: #e94560; }
.kpi-label { font-size: 0.82rem; color: #8899cc; margin-top: 0.25rem; letter-spacing: 0.04em; }
/* Section headers */
.sec-header {
    font-size: 1.2rem; font-weight: 600; color: #56c0f0;
    border-bottom: 2px solid #1a3a5c; padding-bottom: 0.4rem; margin-bottom: 1rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# DATA GENERATION  (cached – runs once per session)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Generating synthetic dataset …")
def generate_data(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    n_users = 700
    n_products = 130

    categories = [
        "Laptop", "Phone", "Tablet", "Headphones", "Camera",
        "Speaker", "Smartwatch", "Monitor", "Keyboard", "Mouse",
    ]
    locations = [
        "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
        "San Diego", "Dallas", "San Jose", "Austin", "Seattle",
    ]

    products_df = pd.DataFrame([{
        "product_id": f"P{i:03d}",
        "product_name": f"{categories[i % len(categories)]} Pro {i + 1}",
        "category": categories[i % len(categories)],
        "price": round(random.uniform(20, 2000), 2),
    } for i in range(n_products)])

    users_df = pd.DataFrame([{
        "user_id": f"U{i:04d}",
        "username": f"user_{i + 1}",
        "location": random.choice(locations),
    } for i in range(n_users)])

    pos_templates = [
        "Excellent product! The {f} is amazing and works perfectly.",
        "Highly recommended! Great {f} and very fast delivery.",
        "Outstanding quality! The {f} exceeded all my expectations.",
        "Best purchase ever. The {f} is absolutely top notch.",
        "Wonderful item! The {f} is exactly what I needed.",
        "Love this! The {f} works flawlessly every single time.",
        "Perfect for my needs. The {f} is impressive and reliable.",
        "Great value for money. The {f} is outstanding performance.",
        "Five stars! The {f} is fantastic and very durable indeed.",
        "Absolutely brilliant product. The {f} is superb and well made.",
    ]
    neg_templates = [
        "Disappointed with this. The {f} is terrible quality.",
        "Poor quality item. The {f} broke after just a week of use.",
        "Not worth the money. The {f} does not work as expected.",
        "Terrible experience overall. The {f} is very misleading.",
        "Would not recommend to anyone. The {f} is completely defective.",
        "Very unhappy with this purchase. The {f} stopped working soon.",
        "Bad product overall. The {f} is nothing like the description.",
        "Waste of money. The {f} is completely useless junk.",
        "Very poor quality. The {f} broke immediately upon first use.",
        "Stay away! The {f} is badly made and totally unreliable.",
    ]
    neu_templates = [
        "Okay product. The {f} is average and does the job fine.",
        "Does what it says. The {f} is acceptable for the price paid.",
        "Average product overall. The {f} could definitely be better.",
        "Not bad but not great. The {f} is quite mediocre honestly.",
        "Decent enough product. The {f} works sometimes.",
        "Mixed feelings about this. The {f} is rather inconsistent.",
        "It is what it is. The {f} is neither great nor terrible.",
        "Middle of the road product. The {f} is just barely okay.",
    ]
    features = [
        "battery life", "build quality", "performance", "design",
        "display", "sound quality", "camera", "software",
        "connectivity", "durability", "ease of use", "value for money",
    ]

    start_dt = datetime(2022, 1, 1)
    total_days = (datetime(2024, 12, 31) - start_dt).days
    pid_to_cat = products_df.set_index("product_id")["category"].to_dict()

    reviews = []
    used = set()

    for uid in users_df["user_id"]:
        k = random.randint(9, 26)
        for pid in random.sample(list(products_df["product_id"]), min(k, n_products)):
            if (uid, pid) in used:
                continue
            used.add((uid, pid))

            rating = random.choices([1, 2, 3, 4, 5], weights=[5, 10, 15, 35, 35])[0]
            f = random.choice(features)

            if rating >= 4:
                text = random.choice(pos_templates).format(f=f)
                sentiment = "positive"
                score = round(random.uniform(0.35, 1.0), 3)
            elif rating <= 2:
                text = random.choice(neg_templates).format(f=f)
                sentiment = "negative"
                score = round(random.uniform(-1.0, -0.35), 3)
            else:
                text = random.choice(neu_templates).format(f=f)
                sentiment = "neutral"
                score = round(random.uniform(-0.28, 0.28), 3)

            ts = start_dt + timedelta(days=random.randint(0, total_days))

            reviews.append({
                "review_id": f"R{len(reviews):05d}",
                "user_id": uid,
                "product_id": pid,
                "category": pid_to_cat[pid],
                "rating": rating,
                "review_text": text,
                "sentiment": sentiment,
                "sentiment_score": score,
                "timestamp": ts,
                "helpful_votes": random.randint(0, 100),
            })

    rev_df = pd.DataFrame(reviews)
    rev_df["timestamp"] = pd.to_datetime(rev_df["timestamp"])
    rev_df["year_month"] = rev_df["timestamp"].dt.to_period("M").astype(str)
    rev_df["month_date"] = rev_df["timestamp"].dt.to_period("M").dt.to_timestamp()
    rev_df["year"] = rev_df["timestamp"].dt.year
    rev_df["month"] = rev_df["timestamp"].dt.month

    return rev_df, users_df, products_df


# ─────────────────────────────────────────────────────────────────────────────
# NETWORK BUILDER  (cached per unique filtered dataframe + params)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Building user co-review network …")
def build_network(reviews_df: pd.DataFrame, min_common: int = 2, max_nodes: int = 160):
    top_users = reviews_df["user_id"].value_counts().head(max_nodes).index.tolist()
    df = reviews_df[reviews_df["user_id"].isin(top_users)]

    user_prods   = df.groupby("user_id")["product_id"].apply(set).to_dict()
    user_sent    = df.groupby("user_id")["sentiment_score"].mean().to_dict()
    user_reviews = df.groupby("user_id").size().to_dict()
    user_rating  = df.groupby("user_id")["rating"].mean().to_dict()

    G = nx.Graph()
    for uid in top_users:
        G.add_node(
            uid,
            review_count=user_reviews.get(uid, 0),
            avg_sentiment=round(user_sent.get(uid, 0), 4),
            avg_rating=round(user_rating.get(uid, 0), 2),
        )

    for i in range(len(top_users)):
        for j in range(i + 1, len(top_users)):
            u1, u2 = top_users[i], top_users[j]
            common = len(user_prods.get(u1, set()) & user_prods.get(u2, set()))
            if common >= min_common:
                G.add_edge(u1, u2, weight=common)

    return G


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar(rev_df: pd.DataFrame, prod_df: pd.DataFrame):
    with st.sidebar:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1f/Python_logo_01.svg/48px-Python_logo_01.svg.png",
            width=40,
        )
        st.markdown("## ⚙️ Controls & Filters")

        st.markdown("### 📅 Date Range")
        min_d = rev_df["timestamp"].min().date()
        max_d = rev_df["timestamp"].max().date()
        date_range = st.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)

        st.markdown("### 🏷️ Category")
        cats = ["All"] + sorted(prod_df["category"].unique().tolist())
        sel_cat = st.selectbox("Category", cats)

        st.markdown("### 😊 Sentiment")
        sel_sent = st.multiselect(
            "Sentiment", ["positive", "neutral", "negative"],
            default=["positive", "neutral", "negative"],
        )

        st.markdown("### ⭐ Min Rating")
        min_rating = st.slider("Min rating", 1, 5, 1)

        st.markdown("---")
        st.markdown("### 🔗 Network")
        min_common = st.slider("Min shared products (edge threshold)", 1, 6, 2)
        max_nodes  = st.slider("Max nodes to display", 50, 200, 150)

        st.markdown("---")
        st.markdown("### 📊 Full Dataset Stats")
        st.metric("Total Reviews",  f"{len(rev_df):,}")
        st.metric("Unique Users",   rev_df["user_id"].nunique())
        st.metric("Unique Products", rev_df["product_id"].nunique())
        st.metric("Date Span", f"Jan 2022 – Dec 2024")

    return date_range, sel_cat, sel_sent, min_rating, min_common, max_nodes


def apply_filters(rev_df, date_range, sel_cat, sel_sent, min_rating):
    df = rev_df.copy()
    if len(date_range) == 2:
        df = df[
            (df["timestamp"].dt.date >= date_range[0])
            & (df["timestamp"].dt.date <= date_range[1])
        ]
    if sel_cat != "All":
        df = df[df["category"] == sel_cat]
    if sel_sent:
        df = df[df["sentiment"].isin(sel_sent)]
    df = df[df["rating"] >= min_rating]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1  ─  OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────

def tab_overview(df: pd.DataFrame):
    st.markdown('<div class="sec-header">📊 Dashboard Overview</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        (f"{len(df):,}", "Total Reviews"),
        (f"{df['user_id'].nunique():,}", "Unique Users"),
        (f"{df['product_id'].nunique():,}", "Products"),
        (f"{df['rating'].mean():.2f} ⭐", "Avg Rating"),
        (f"{(df['sentiment']=='positive').mean()*100:.0f}%", "Positive Rate"),
    ]
    for col, (val, lbl) in zip([c1, c2, c3, c4, c5], kpis):
        col.markdown(
            f'<div class="kpi-card"><div class="kpi-value">{val}</div>'
            f'<div class="kpi-label">{lbl}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        rc = df["rating"].value_counts().sort_index()
        fig = go.Figure(go.Bar(
            x=[f"{'★'*i} ({i})" for i in rc.index],
            y=rc.values,
            marker_color=["#e74c3c","#e67e22","#f1c40f","#2ecc71","#27ae60"],
            text=rc.values, textposition="outside",
        ))
        fig.update_layout(
            title="Rating Distribution", xaxis_title="Rating", yaxis_title="Count",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="white", height=340,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        sc = df["sentiment"].value_counts()
        fig = go.Figure(go.Pie(
            labels=sc.index, values=sc.values, hole=0.46,
            marker_colors=[
                "#2ecc71" if l == "positive" else ("#e74c3c" if l == "negative" else "#f39c12")
                for l in sc.index
            ],
        ))
        fig.update_layout(
            title="Sentiment Distribution",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="white", height=340,
        )
        st.plotly_chart(fig, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        cc = df.groupby("category").size().sort_values()
        fig = go.Figure(go.Bar(
            x=cc.values, y=cc.index, orientation="h",
            marker_color="#3498db", text=cc.values, textposition="outside",
        ))
        fig.update_layout(
            title="Reviews by Category", xaxis_title="Count",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="white", height=340,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        ar = df.groupby("category")["rating"].mean().sort_values()
        fig = go.Figure(go.Bar(
            x=ar.values, y=ar.index, orientation="h",
            marker_color="#9b59b6",
            text=[f"{v:.2f}" for v in ar.values], textposition="outside",
        ))
        fig.update_layout(
            title="Avg Rating by Category", xaxis_title="Avg Rating", xaxis_range=[0, 5.6],
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="white", height=340,
        )
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2  ─  NETWORK ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def _network_layout(G, algo: str):
    if algo == "Spring (Force-directed)":
        return nx.spring_layout(G, seed=42, k=0.55)
    elif algo == "Circular":
        return nx.circular_layout(G)
    elif algo == "Kamada-Kawai":
        try:
            return nx.kamada_kawai_layout(G)
        except Exception:
            return nx.spring_layout(G, seed=42)
    else:
        return nx.random_layout(G, seed=42)


def plot_network_plotly(G, layout_algo="Spring (Force-directed)", color_by="community"):
    if G.number_of_nodes() == 0:
        return None

    pos = _network_layout(G, layout_algo)

    # Community detection
    try:
        comms = list(nx.community.greedy_modularity_communities(G))
        node_comm = {n: i for i, c in enumerate(comms) for n in c}
    except Exception:
        node_comm = {n: 0 for n in G.nodes()}

    # Edges
    ex, ey = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        ex += [x0, x1, None]; ey += [y0, y1, None]

    edge_trace = go.Scatter(
        x=ex, y=ey, mode="lines",
        line=dict(width=0.5, color="rgba(150,160,200,0.25)"),
        hoverinfo="none", showlegend=False,
    )

    nx_vals = [pos[n][0] for n in G.nodes()]
    ny_vals = [pos[n][1] for n in G.nodes()]

    if color_by == "sentiment":
        colors = [G.nodes[n].get("avg_sentiment", 0) for n in G.nodes()]
        colorscale = "RdYlGn"
        cmin, cmax = -1, 1
        cbar_title = "Avg Sentiment"
    else:
        colors = [node_comm.get(n, 0) for n in G.nodes()]
        colorscale = "Viridis"
        cmin, cmax = None, None
        cbar_title = "Community"

    sizes = [7 + G.degree(n) * 1.4 for n in G.nodes()]

    hover = [
        f"User: {n}<br>Degree: {G.degree(n)}<br>"
        f"Reviews: {G.nodes[n].get('review_count',0)}<br>"
        f"Avg Sentiment: {G.nodes[n].get('avg_sentiment',0):.3f}<br>"
        f"Avg Rating: {G.nodes[n].get('avg_rating',0):.2f}<br>"
        f"Community: {node_comm.get(n, '—')}"
        for n in G.nodes()
    ]

    node_trace = go.Scatter(
        x=nx_vals, y=ny_vals, mode="markers",
        hoverinfo="text", text=hover,
        marker=dict(
            size=sizes, color=colors,
            colorscale=colorscale, cmin=cmin, cmax=cmax,
            showscale=True,
            colorbar=dict(title=cbar_title, thickness=14),
            line=dict(width=0.5, color="white"),
        ),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False, hovermode="closest",
        margin=dict(b=10, l=0, r=0, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="rgba(10,10,28,1)",
        paper_bgcolor="rgba(10,10,28,1)",
        font_color="white", height=560,
    )
    return fig


def plot_adjacency_matrix(G, n_show=60):
    nodes = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:n_show]
    sub = G.subgraph(nodes)
    adj = nx.to_pandas_adjacency(sub)
    deg_order = sorted(adj.columns, key=lambda n: G.degree(n), reverse=True)
    adj = adj.loc[deg_order, deg_order]
    labels = [n[-4:] for n in adj.columns]

    fig = go.Figure(go.Heatmap(
        z=adj.values, x=labels, y=labels,
        colorscale="Blues", showscale=True,
        colorbar=dict(title="Connected"),
        hovertemplate="Row: %{y}<br>Col: %{x}<br>Edge: %{z}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Adjacency Matrix — Top {n_show} Nodes (sorted by degree)",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(14,14,34,1)",
        font_color="white", height=560,
        xaxis=dict(tickfont=dict(size=7)),
        yaxis=dict(tickfont=dict(size=7)),
    )
    return fig


def tab_network(df: pd.DataFrame, min_common: int, max_nodes: int):
    st.markdown('<div class="sec-header">🔗 Network Analysis</div>', unsafe_allow_html=True)

    G = build_network(df, min_common=min_common, max_nodes=max_nodes)

    if G.number_of_nodes() == 0:
        st.error("Network is empty — lower 'Min shared products' in the sidebar.")
        return

    # KPIs
    try:
        comms = list(nx.community.greedy_modularity_communities(G))
        mod = nx.community.modularity(G, comms)
        n_comms = len(comms)
    except Exception:
        mod, n_comms = None, None

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Nodes",   G.number_of_nodes())
    c2.metric("Edges",   G.number_of_edges())
    c3.metric("Density", f"{nx.density(G):.5f}")
    c4.metric("Avg Degree", f"{np.mean([d for _,d in G.degree()]):.1f}")
    c5.metric("Communities", n_comms if n_comms else "—")

    st.markdown("---")

    vcol1, vcol2 = st.columns([3, 1])
    with vcol2:
        view   = st.radio("View", ["Node-Link Diagram", "Adjacency Matrix"])
        layout = st.selectbox("Layout", ["Spring (Force-directed)", "Circular", "Kamada-Kawai", "Random"])
        color_by = st.radio("Color nodes by", ["community", "sentiment"])
        if view == "Adjacency Matrix":
            n_show = st.slider("Nodes in matrix", 20, 100, 60)

    with vcol1:
        if view == "Node-Link Diagram":
            fig = plot_network_plotly(G, layout_algo=layout, color_by=color_by)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Nodes = users. Edges = shared product reviews ≥ threshold. "
                "Hover over a node for full details. Size ∝ degree."
            )
        else:
            fig = plot_adjacency_matrix(G, n_show=n_show)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Blue cells = connected users. Ordered by degree (highest first). "
                "Compare with node-link — matrix scales better for dense graphs."
            )

    st.markdown("---")
    col_deg, col_comm = st.columns(2)

    with col_deg:
        degs = [d for _, d in G.degree()]
        fig = go.Figure(go.Histogram(
            x=degs, nbinsx=25, marker_color="#3498db",
            marker_line=dict(color="white", width=0.4),
        ))
        fig.update_layout(
            title="Degree Distribution", xaxis_title="Degree", yaxis_title="Count",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="white", height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_comm:
        if n_comms and comms:
            sizes_c = sorted([len(c) for c in comms], reverse=True)
            labels_c = [f"C{i+1}" for i in range(len(sizes_c))]
            fig = go.Figure(go.Bar(
                x=labels_c, y=sizes_c, marker_color="#9b59b6",
                text=sizes_c, textposition="outside",
            ))
            fig.update_layout(
                title=f"Community Sizes  (modularity = {mod:.4f})" if mod else "Community Sizes",
                xaxis_title="Community", yaxis_title="Members",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="white", height=300,
            )
            st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3  ─  TEXT ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────

_STOP = {
    "the","a","an","is","it","its","and","or","but","in","on","for","with",
    "this","that","my","i","me","be","to","of","very","so","just","not","no",
    "after","from","what","ever","all","has","have","been","are","was","were",
    "their","they","you","your","our","us","we","he","she","his","her","at",
    "by","as","if","do","did","will","would","could","should","am","into","each",
}


@st.cache_data(show_spinner=False)
def make_wordcloud(texts_key: str, texts: tuple):
    text = " ".join(texts)
    wc = WordCloud(
        width=820, height=400, background_color="#0d0d1a",
        colormap="plasma", max_words=120, stopwords=_STOP,
        prefer_horizontal=0.75,
    ).generate(text)
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#0d0d1a")
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig


@st.cache_data(show_spinner="Running LDA …")
def run_lda(texts: tuple, n_topics: int = 5, n_words: int = 8):
    vec = CountVectorizer(max_features=600, stop_words="english", min_df=2)
    dtm = vec.fit_transform(list(texts))
    vocab = vec.get_feature_names_out()
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=12)
    lda.fit(dtm)
    topics = []
    for i, comp in enumerate(lda.components_):
        idx = comp.argsort()[-n_words:][::-1]
        words   = [vocab[j] for j in idx]
        weights = [float(comp[j]) for j in idx]
        topics.append({"topic": i + 1, "words": words, "weights": weights})
    return topics


def tab_text(df: pd.DataFrame):
    st.markdown('<div class="sec-header">📝 Text Analytics</div>', unsafe_allow_html=True)

    wc_filter = st.radio(
        "Word Cloud — filter by sentiment",
        ["All Reviews", "Positive Only", "Negative Only", "Neutral Only"],
        horizontal=True,
    )
    filter_map = {
        "All Reviews":    df["review_text"].tolist(),
        "Positive Only":  df[df["sentiment"] == "positive"]["review_text"].tolist(),
        "Negative Only":  df[df["sentiment"] == "negative"]["review_text"].tolist(),
        "Neutral Only":   df[df["sentiment"] == "neutral"]["review_text"].tolist(),
    }
    texts = filter_map[wc_filter]
    if not texts:
        st.warning("No reviews match the selected filter.")
        return

    col_wc, col_freq = st.columns(2)

    with col_wc:
        st.markdown("**Word Cloud**")
        fig_wc = make_wordcloud(wc_filter, tuple(texts))
        st.pyplot(fig_wc, use_container_width=True)
        plt.close("all")

    with col_freq:
        st.markdown("**Top-20 Words by Frequency**")
        all_words = [
            w.lower().strip(".,!?;:'\"")
            for t in texts
            for w in t.split()
            if w.lower().strip(".,!?;:'\"") not in _STOP and len(w) > 2
        ]
        freq = Counter(all_words).most_common(20)
        if freq:
            ws, cs = zip(*freq)
            fig = go.Figure(go.Bar(
                x=list(cs), y=list(ws), orientation="h",
                marker_color="#e74c3c", text=list(cs), textposition="outside",
            ))
            fig.update_layout(
                title="Top 20 Most Frequent Words", xaxis_title="Frequency",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="white", height=440,
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        sc_df = df.groupby(["category","sentiment"]).size().reset_index(name="count")
        fig = px.bar(
            sc_df, x="category", y="count", color="sentiment",
            color_discrete_map={"positive":"#2ecc71","neutral":"#f39c12","negative":"#e74c3c"},
            title="Sentiment Breakdown by Category", barmode="stack",
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="white", height=370, xaxis_tickangle=-40,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = go.Figure()
        for s, c in [("positive","#2ecc71"),("neutral","#f39c12"),("negative","#e74c3c")]:
            sub = df[df["sentiment"] == s]["sentiment_score"]
            fig.add_trace(go.Violin(
                y=sub, name=s.capitalize(),
                fillcolor=c, opacity=0.7,
                box_visible=True, meanline_visible=True,
            ))
        fig.update_layout(
            title="Sentiment Score Distribution (Violin)",
            yaxis_title="Sentiment Score",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="white", height=370,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("**Topic Modeling — LDA**")
    n_topics = st.slider("Number of topics", 3, 8, 5, key="lda_slider")

    try:
        topics = run_lda(tuple(df["review_text"].tolist()), n_topics=n_topics)
        cols = st.columns(min(n_topics, 5))
        for i, topic in enumerate(topics[:5]):
            with cols[i]:
                total_w = sum(topic["weights"])
                st.markdown(f"**Topic {topic['topic']}**")
                for w, wt in zip(topic["words"], topic["weights"]):
                    pct = wt / total_w * 100
                    st.markdown(f"- {w} &nbsp; `{pct:.1f}%`", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"LDA error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4  ─  TEMPORAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def tab_temporal(df: pd.DataFrame):
    st.markdown('<div class="sec-header">📅 Temporal Analysis</div>', unsafe_allow_html=True)

    monthly = (
        df.groupby("month_date")
        .agg(count=("review_id","count"),
             avg_sent=("sentiment_score","mean"),
             avg_rating=("rating","mean"))
        .reset_index()
    )

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure(go.Scatter(
            x=monthly["month_date"], y=monthly["count"],
            mode="lines+markers", line=dict(color="#3498db", width=2),
            fill="tozeroy", fillcolor="rgba(52,152,219,0.18)",
            marker=dict(size=4),
        ))
        fig.update_layout(
            title="Monthly Review Volume", xaxis_title="Month", yaxis_title="Reviews",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="white", height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure(go.Scatter(
            x=monthly["month_date"], y=monthly["avg_sent"],
            mode="lines+markers", line=dict(color="#2ecc71", width=2),
            marker=dict(size=4),
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.35)")
        fig.update_layout(
            title="Monthly Avg Sentiment Score", xaxis_title="Month", yaxis_title="Score",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="white", height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fig = go.Figure(go.Scatter(
            x=monthly["month_date"], y=monthly["avg_rating"],
            mode="lines+markers", line=dict(color="#e74c3c", width=2),
            fill="tozeroy", fillcolor="rgba(231,76,60,0.15)",
            marker=dict(size=4),
        ))
        fig.update_layout(
            title="Monthly Avg Rating", xaxis_title="Month", yaxis_title="Rating",
            yaxis_range=[1, 5.2],
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="white", height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        sm = df.groupby(["month_date","sentiment"]).size().reset_index(name="n")
        pivot = sm.pivot(index="month_date", columns="sentiment", values="n").fillna(0)
        fig = go.Figure()
        for s, color, fill in [
            ("positive", "#2ecc71", "rgba(46,204,113,0.55)"),
            ("neutral",  "#f39c12", "rgba(243,156,18,0.55)"),
            ("negative", "#e74c3c", "rgba(231,76,60,0.55)"),
        ]:
            if s in pivot.columns:
                fig.add_trace(go.Scatter(
                    x=pivot.index, y=pivot[s], mode="lines",
                    name=s.capitalize(),
                    line=dict(color=color, width=1.5),
                    stackgroup="one", fillcolor=fill,
                ))
        fig.update_layout(
            title="Sentiment Trend Over Time (Stacked Area)",
            xaxis_title="Month", yaxis_title="Reviews",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="white", height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("**Category × Month Heatmap**")
    hd = df.groupby(["year_month","category"]).size().reset_index(name="n")
    ph = hd.pivot(index="year_month", columns="category", values="n").fillna(0)
    ph = ph.tail(30)

    fig = go.Figure(go.Heatmap(
        z=ph.values.T, x=ph.index.tolist(), y=ph.columns.tolist(),
        colorscale="Viridis", colorbar=dict(title="Reviews"),
        hovertemplate="Month: %{x}<br>Category: %{y}<br>Reviews: %{z}<extra></extra>",
    ))
    fig.update_layout(
        title="Review Heatmap — Category × Month",
        xaxis_title="Month", yaxis_title="Category",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(14,14,34,1)",
        font_color="white", height=380,
        xaxis=dict(tickfont=dict(size=8), tickangle=-45),
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5  ─  INTEGRATION  (Sentiment-Aware Network + Brushing & Linking)
# ─────────────────────────────────────────────────────────────────────────────

def tab_integration(df: pd.DataFrame, min_common: int, max_nodes: int):
    st.markdown(
        '<div class="sec-header">🔗📝 Integrated Analysis — Sentiment-Aware Network</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Node **color** encodes average user sentiment (🟢 positive → 🔴 negative). "
        "Node **size** encodes review activity. Edges = shared reviews."
    )

    G = build_network(df, min_common=min_common, max_nodes=max_nodes)
    if G.number_of_nodes() == 0:
        st.error("Network is empty — lower 'Min shared products' in the sidebar.")
        return

    fig = plot_network_plotly(G, layout_algo="Spring (Force-directed)", color_by="sentiment")
    if fig:
        fig.update_layout(title="Sentiment-Aware User Co-Review Network", height=580)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🖱️ Brushing & Linking — Explore by Category")
    st.markdown("Select a product category to cross-filter the review panel and user sentiment bar.")

    col1, col2 = st.columns([1, 1])

    with col1:
        sel_cat = st.selectbox("Product Category", sorted(df["category"].unique()), key="integ_cat")
        cdf = df[df["category"] == sel_cat]

        top_u = (
            cdf.groupby("user_id")
            .agg(reviews=("review_id","count"), avg_sent=("sentiment_score","mean"))
            .sort_values("reviews", ascending=False)
            .head(15)
            .reset_index()
        )

        bar_colors = [
            "#2ecc71" if s > 0.1 else ("#e74c3c" if s < -0.1 else "#f39c12")
            for s in top_u["avg_sent"]
        ]
        fig = go.Figure(go.Bar(
            x=top_u["user_id"], y=top_u["reviews"],
            marker_color=bar_colors,
            text=[f"{s:.2f}" for s in top_u["avg_sent"]],
            textposition="outside",
            hovertemplate="User: %{x}<br>Reviews: %{y}<br>Avg Sentiment: %{text}<extra></extra>",
        ))
        fig.update_layout(
            title=f"Top Active Users — {sel_cat}",
            xaxis_title="User", yaxis_title="Reviews",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="white", height=360,
            xaxis=dict(tickangle=-40, tickfont=dict(size=8)),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(f"**Latest Reviews — {sel_cat}**")
        sample = (
            cdf.sort_values("timestamp", ascending=False)
            .head(7)[["user_id","rating","sentiment","review_text","timestamp"]]
        )
        for _, row in sample.iterrows():
            ico = "🟢" if row["sentiment"] == "positive" else ("🔴" if row["sentiment"] == "negative" else "🟡")
            st.markdown(
                f"{ico} **{row['user_id']}** &nbsp;|&nbsp; ⭐ {row['rating']} &nbsp;|&nbsp; "
                f"{row['timestamp'].strftime('%b %Y')}",
                unsafe_allow_html=True,
            )
            st.markdown(f"> _{row['review_text']}_")
            st.markdown("---")

    # Temporal + Network linked view
    st.markdown("### 📅 Temporal Slice → Network Color Shift")
    st.markdown("Drag the year slider to see how user sentiment in this category evolved.")
    yr = st.slider("Year", 2022, 2024, 2023, key="integ_year")
    yr_df = df[(df["year"] == yr) & (df["category"] == sel_cat)]

    c3, c4 = st.columns(2)
    with c3:
        mc = yr_df.groupby("month")["sentiment_score"].mean().reset_index()
        fig = go.Figure(go.Bar(
            x=mc["month"], y=mc["sentiment_score"],
            marker_color=["#2ecc71" if v >= 0 else "#e74c3c" for v in mc["sentiment_score"]],
        ))
        fig.update_layout(
            title=f"Monthly Sentiment — {sel_cat} ({yr})",
            xaxis_title="Month", yaxis_title="Avg Sentiment Score",
            xaxis=dict(tickvals=list(range(1,13)), ticktext=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="white", height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        rc = yr_df["rating"].value_counts().sort_index()
        fig = go.Figure(go.Bar(
            x=rc.index, y=rc.values,
            marker_color=["#e74c3c","#e67e22","#f1c40f","#2ecc71","#27ae60"],
            text=rc.values, textposition="outside",
        ))
        fig.update_layout(
            title=f"Rating Distribution — {sel_cat} ({yr})",
            xaxis_title="Rating", yaxis_title="Count",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="white", height=300,
        )
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 6  ─  EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def tab_evaluation(df: pd.DataFrame, min_common: int, max_nodes: int):
    st.markdown('<div class="sec-header">📏 Evaluation & Metrics</div>', unsafe_allow_html=True)

    G = build_network(df, min_common=min_common, max_nodes=max_nodes)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Graph Metrics")
        if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
            avg_deg = np.mean([d for _, d in G.degree()])
            max_deg = max(d for _, d in G.degree())
            try:
                comms = list(nx.community.greedy_modularity_communities(G))
                mod = nx.community.modularity(G, comms)
                n_comms = len(comms)
            except Exception:
                mod, n_comms = None, None
            try:
                avg_clust = nx.average_clustering(G)
            except Exception:
                avg_clust = None
            try:
                lcc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
                avg_path = nx.average_shortest_path_length(lcc)
                diam = nx.diameter(lcc)
            except Exception:
                avg_path, diam = None, None

            rows = {
                "Nodes":  G.number_of_nodes(),
                "Edges":  G.number_of_edges(),
                "Graph Density":   f"{nx.density(G):.5f}",
                "Avg Degree":      f"{avg_deg:.2f}",
                "Max Degree":      max_deg,
                "Avg Clustering":  f"{avg_clust:.4f}" if avg_clust else "N/A",
                "Modularity":      f"{mod:.4f}" if mod else "N/A",
                "Communities":     n_comms if n_comms else "N/A",
                "Avg Path Length (LCC)": f"{avg_path:.3f}" if avg_path else "N/A",
                "Diameter (LCC)":  diam if diam else "N/A",
            }
            st.dataframe(
                pd.DataFrame(list(rows.items()), columns=["Metric", "Value"]),
                use_container_width=True, hide_index=True,
            )

    with col2:
        st.markdown("### Text Analytics Metrics")
        sd = df["sentiment"].value_counts(normalize=True) * 100
        rows_t = {
            "Total Reviews":    len(df),
            "Positive Reviews": f"{sd.get('positive',0):.1f}%",
            "Neutral Reviews":  f"{sd.get('neutral',0):.1f}%",
            "Negative Reviews": f"{sd.get('negative',0):.1f}%",
            "Avg Sentiment Score": f"{df['sentiment_score'].mean():.4f}",
            "Std Dev Score":    f"{df['sentiment_score'].std():.4f}",
            "Avg Words/Review": f"{df['review_text'].apply(lambda x: len(x.split())).mean():.1f}",
            "Avg Rating":       f"{df['rating'].mean():.2f}",
            "Rating Std Dev":   f"{df['rating'].std():.2f}",
        }
        st.dataframe(
            pd.DataFrame(list(rows_t.items()), columns=["Metric", "Value"]),
            use_container_width=True, hide_index=True,
        )

    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### Node-Link vs Adjacency Matrix")
        cmp = pd.DataFrame({
            "Criterion": [
                "Scalability (>500 nodes)", "Community visibility",
                "Edge weight clarity", "Dense graph readability",
                "Individual node query", "Interaction support",
            ],
            "Node-Link": ["❌ Poor","✅ Excellent","✅ Good","❌ Hairball","✅ Hover","✅ Excellent"],
            "Adjacency Matrix": ["✅ Good","✅ Good (reorder)","✅ Excellent","✅ Excellent","⚠️ Moderate","⚠️ Moderate"],
        })
        st.dataframe(cmp, use_container_width=True, hide_index=True)

    with col4:
        st.markdown("### Static vs Interactive Views")
        si = pd.DataFrame({
            "Criterion": [
                "Insight depth", "Data exploration", "Anomaly detection",
                "Reproducibility", "Cognitive load", "Best use case",
            ],
            "Static": ["Limited","Poor","Poor","✅ Excellent","Low","Reports / publications"],
            "Interactive": ["High","✅ Excellent","✅ Excellent","Moderate","Moderate","Exploration & analysis"],
        })
        st.dataframe(si, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Simulated User Study (SUS Scale)")
    sus = pd.DataFrame({
        "Evaluator":            [f"Evaluator {i+1}" for i in range(8)],
        "SUS Score (/100)":     [82, 78, 85, 90, 76, 88, 83, 79],
        "Task Completion (%)":  [95, 88, 100, 100, 85, 97, 92, 90],
        "Avg Task Time (s)":    [42,  55,  38,  31,  60,  39,  44,  52],
        "User Satisfaction":    ["Good","Good","Excellent","Excellent","Good","Excellent","Good","Good"],
    })
    st.dataframe(sus, use_container_width=True, hide_index=True)

    avg_sus  = sus["SUS Score (/100)"].mean()
    avg_comp = sus["Task Completion (%)"].mean()
    avg_time = sus["Avg Task Time (s)"].mean()
    grade    = "Excellent" if avg_sus >= 85 else ("Good" if avg_sus >= 70 else "Acceptable")

    m1, m2, m3 = st.columns(3)
    m1.metric("Avg SUS Score", f"{avg_sus:.1f} / 100", delta=grade)
    m2.metric("Avg Task Completion", f"{avg_comp:.1f}%")
    m3.metric("Avg Task Time", f"{avg_time:.1f} s")

    # Centrality top users
    st.markdown("---")
    st.markdown("### Top-10 Users by Centrality (Betweenness)")
    if G.number_of_nodes() > 2:
        try:
            bc = nx.betweenness_centrality(G, normalized=True)
            top_bc = sorted(bc.items(), key=lambda x: x[1], reverse=True)[:10]
            bc_df = pd.DataFrame(top_bc, columns=["User", "Betweenness Centrality"])
            bc_df["Betweenness Centrality"] = bc_df["Betweenness Centrality"].round(5)
            fig = go.Figure(go.Bar(
                x=bc_df["Betweenness Centrality"], y=bc_df["User"],
                orientation="h", marker_color="#e94560",
                text=bc_df["Betweenness Centrality"].astype(str),
                textposition="outside",
            ))
            fig.update_layout(
                title="Top 10 Users — Betweenness Centrality",
                xaxis_title="Betweenness Centrality",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="white", height=340,
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info(f"Centrality computation skipped: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.title("📊 Explainable Interactive Visual Analytics Framework")
    st.markdown(
        "**Multimodal Data Analysis: Network + Text + Temporal**  \n"
        "E-Commerce Product Reviews Dataset &nbsp;|&nbsp; "
        "Course: COMP-834 Advanced Data Visualization &nbsp;|&nbsp; Assignment 3"
    )
    st.markdown("---")

    rev_df, users_df, prod_df = generate_data()

    date_range, sel_cat, sel_sent, min_rating, min_common, max_nodes = \
        render_sidebar(rev_df, prod_df)

    filtered = apply_filters(rev_df, date_range, sel_cat, sel_sent, min_rating)

    if filtered.empty:
        st.warning("⚠️ No data matches the current filters. Adjust the sidebar settings.")
        return

    t1, t2, t3, t4, t5, t6 = st.tabs([
        "📊 Overview",
        "🔗 Network Analysis",
        "📝 Text Analytics",
        "📅 Temporal Analysis",
        "🔗📝 Integration",
        "📏 Evaluation",
    ])

    with t1: tab_overview(filtered)
    with t2: tab_network(filtered, min_common, max_nodes)
    with t3: tab_text(filtered)
    with t4: tab_temporal(filtered)
    with t5: tab_integration(filtered, min_common, max_nodes)
    with t6: tab_evaluation(filtered, min_common, max_nodes)


if __name__ == "__main__":
    main()
