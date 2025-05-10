import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from textblob import TextBlob

# ========== ページ設定 & CSS ==========
st.set_page_config(page_title="@cosme Review Insight", page_icon="💄", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Montserrat', sans-serif; }
    body { background: #FAF8FF; }
    h1,h2,h3 { color:#7B1FA2; font-weight:600; }
    .stButton>button, .stDownloadButton>button {
        background:#7B1FA2; color:#fff; border:none; border-radius:8px;
        padding:0.5rem 1.2rem; font-weight:600; transition:0.3s;
    }
    .stButton>button:hover, .stDownloadButton>button:hover { background:#9B4DCC; }
    .metric-card { background:#fff; border-radius:12px; box-shadow:0 4px 14px rgba(0,0,0,0.06); padding:1rem; margin:0.5rem 0; }
    </style>
    """, unsafe_allow_html=True)

# ========== サイドバー ==========
with st.sidebar:
    st.header("🔍 レビュー分析設定")
    url_input = st.text_input("@cosmeレビューURL", placeholder="例: https://www.cosme.net/products/10240630/review/")
    max_pages = st.slider("最大ページ数", 1, 10, 3)
    submitted = st.button("分析開始")
    st.info("※URLは /review/ で終わるようにしてください")

# ========== データ取得関数 ==========
def get_reviews(url: str, max_pages: int) -> pd.DataFrame:
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0", "Accept-Language": "ja-JP,ja;q=0.9"})
    reviews = []
    for page in range(1, max_pages + 1):
        resp = session.get(f"{url}?page={page}", timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        items = soup.select("#product-review-list > div")
        if not items:
            break
        for item in items:
            # 評価スコア
            star = item.select_one("div.body div.rating.clearfix p.reviewer-rating")
            rating = float(re.sub(r"[^0-9.]", "", star.text)) if star else None
            # プロフィール
            prof = item.select_one("div.head div.reviewer-info")
            profile = prof.get_text(" ", strip=True) if prof else ""
            # 本文
            body_tag = item.select_one("div.body p:not(.reviewer-rating):not(.mobile-date)")
            text = body_tag.get_text(strip=True) if body_tag else ""
            # 日付
            date_tag = item.select_one("div.body div.rating.clearfix p.mobile-date")
            date_txt = date_tag.get_text(strip=True) if date_tag else ""
            reviews.append({"評価": rating, "属性": profile, "本文": text, "日付": date_txt})
    return pd.DataFrame(reviews)

# ========== メイン ==========
st.title("💄 @cosme Review Insight")
st.caption("迅速にレビューを取得・分析。最大ページ数指定で高速化可能。")

if submitted and url_input:
    with st.spinner("レビュー取得中…"):
        df = get_reviews(url_input, max_pages)
    # ----- デバッグ情報 -----
    st.subheader("🛠 デバッグ情報")
    st.write("DataFrame の先頭5件:")
    st.write(df.head())
    st.write("年代 列のユニーク値:", df["年代"].unique() if "年代" in df.columns else "カラムなし")
    st.write("欠損数:")
    if all(col in df.columns for col in ["年代","性別","肌質"]):
        st.write(df[["年代","性別","肌質"]].isna().sum())
    else:
        st.write("年代/性別/肌質 カラムが揃っていません")
    st.write("全カラム一覧:", df.columns.tolist())
    # ----- ここまで -----

    if df.empty:
        st.error("⚠️ レビューが取得できませんでした。URL／ページ数を確認してください。")
    else:
        st.success(f"✅ {len(df)} 件のレビューを取得しました！")
        # プロフィール分解
        if "属性" in df.columns:
            df[["年代","性別","肌質"]] = df["属性"].str.extract(r"(\d+代)\s+(男性|女性)\s*(.*)")
        # メトリクス
        c1, c2, c3 = st.columns(3)
        c1.metric("平均評価", f"{df['評価'].mean():.2f}")
        c2.metric("ポジティブ率", f"{(df['評価']>=5).mean()*100:.1f}%")
        c3.metric("レビュー数", f"{len(df)}")
        # グラフ
        g1, g2 = st.columns(2)
        with g1:
            st.subheader("年代別平均評価")
            fig, ax = plt.subplots()
            if "年代" in df.columns:
                df.groupby("年代")["評価"].mean().plot.bar(ax=ax, edgecolor="black")
            st.pyplot(fig)
        with g2:
            st.subheader("肌質別平均評価")
            fig2, ax2 = plt.subplots()
            if "肌質" in df.columns:
                df.groupby("肌質")["評価"].mean().plot.bar(ax=ax2, edgecolor="black")
            st.pyplot(fig2)
        # 感情スコア
        st.subheader("感情スコア分布")
        df['sentiment'] = df['本文'].apply(lambda x: TextBlob(x).sentiment.polarity)
        fig3, ax3 = plt.subplots()
        df['sentiment'].hist(bins=20, ax=ax3)
        st.pyplot(fig3)
        # セグメント
        st.subheader("年代×クラスタ 分布")
        tfidf = TfidfVectorizer(max_features=30, stop_words="japanese")
        X = tfidf.fit_transform(df['本文'])
        km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X)
        df['クラスタ'] = km.labels_
        seg = pd.crosstab(df['年代'], df['クラスタ'])
        st.dataframe(seg)
        # ダウンロード
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("CSVダウンロード", data=csv, file_name="cosme_reviews.csv", mime="text/csv")
