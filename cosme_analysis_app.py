import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
import streamlit as st
import matplotlib.pyplot as plt
# 日本語フォント設定
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Yu Gothic','Meiryo','TakaoPGothic','Noto Sans CJK JP']
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from textblob import TextBlob

# ===== Streamlit ページ設定 & CSS =====
st.set_page_config(page_title="@cosme Review Insight", page_icon="💄", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Montserrat', sans-serif; }
    body { background: #FAF8FF; padding:1rem; }
    h1,h2,h3 { color:#7B1FA2; font-weight:600; }
    .stButton>button, .stDownloadButton>button { background:#7B1FA2; color:#fff; border:none; border-radius:8px; padding:0.5rem 1.2rem; font-weight:600; }
    .stButton>button:hover, .stDownloadButton>button:hover { background:#9B4DCC; }
    </style>
    """, unsafe_allow_html=True)

# ===== サイドバー設定 =====
with st.sidebar:
    st.header("🔍 レビュー分析設定")
    url_input = st.text_input("@cosmeレビューURL", placeholder="https://www.cosme.net/products/10240630/review/")
    max_pages = st.slider("最大ページ数", 1, 5, 2)
    submitted = st.button("分析開始")
    st.info("※ URLは /review/ で終わるページを入力してください")

# ===== レビュー取得関数 =====
def get_reviews(url: str, max_pages: int) -> pd.DataFrame:
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "ja-JP,ja;q=0.9"
    })
    reviews = []
    for page in range(1, max_pages + 1):
        resp = session.get(f"{url}?page={page}", timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        items = soup.select("#product-review-list > div")
        if not items:
            break
        for item in items:
            star = item.select_one("div.body div.rating.clearfix p.reviewer-rating")
            rating = float(re.sub(r"[^0-9.]", "", star.text)) if star else None
            prof = item.select_one("div.head div.reviewer-info")
            profile = prof.get_text(" ", strip=True) if prof else ""
            body = item.select_one("div.body > p:not(.reviewer-rating):not(.mobile-date)")
            text = body.get_text(strip=True) if body else ""
            date = item.select_one("div.body div.rating.clearfix p.mobile-date")
            date_txt = date.get_text(strip=True) if date else ""
            reviews.append({"評価": rating, "属性": profile, "本文": text, "日付": date_txt})
    return pd.DataFrame(reviews)

# ===== メイン =====
st.title("💄 @cosme Review Insight")
st.caption("迅速にコスメレビューを取得・分析します。最大ページ数で速度調整可能。日本語フォント対応済み。")

if submitted and url_input:
    with st.spinner("レビュー取得中…"):
        df = get_reviews(url_input, max_pages)

    if df.empty:
        st.error("⚠️ レビューが取得できませんでした。URLまたはページ数を確認してください。")
    else:
        st.success(f"✅ {len(df)} 件のレビューを取得しました！")
        # 属性分解：年齢・性別・肌質を抽出
        df['年齢'] = df['属性'].str.extract(r"(\d+)歳")[0].fillna('不明').apply(lambda x: x+'歳' if x!='不明' else x)
        df['性別'] = df['属性'].str.extract(r"(男性|女性)")[0].fillna('不明')
        df['肌質'] = df['属性'].str.extract(r"(乾燥肌|脂性肌|普通肌|混合肌)")[0].fillna('不明')

        # メトリクスカード
        c1, c2, c3 = st.columns(3)
        c1.metric("平均評価", f"{df['評価'].mean():.2f}")
        c2.metric("ポジティブ率", f"{(df['評価']>=5).mean()*100:.1f}%")
        c3.metric("レビュー数", f"{len(df)} 件")

        # 年代別/肌質別評価グラフ
        g1, g2 = st.columns(2)
        with g1:
            st.subheader("年代別平均評価")
            fig, ax = plt.subplots()
            df.groupby("年齢")["評価"].mean().plot.bar(ax=ax, edgecolor="black")
            ax.set_ylabel("平均評価")
            st.pyplot(fig)
        with g2:
            st.subheader("肌質別平均評価")
            fig2, ax2 = plt.subplots()
            df.groupby("肌質")["評価"].mean().plot.bar(ax=ax2, edgecolor="black")
            ax2.set_ylabel("平均評価")
            st.pyplot(fig2)

        # 感情スコア分布
        st.subheader("感情スコア分布")
        df['sentiment'] = df['本文'].apply(lambda x: TextBlob(x).sentiment.polarity)
        fig3, ax3 = plt.subplots()
        df['sentiment'].hist(bins=20, color="#26A69A", edgecolor="white", ax=ax3)
        ax3.set_xlabel("Polarity")
        ax3.set_ylabel("件数")
        st.pyplot(fig3)

        # クラスタリング
        st.subheader("レビュー本文クラスタリング (3 clusters)")
        tfidf = TfidfVectorizer(max_features=30)
        X = tfidf.fit_transform(df['本文'])
        km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X)
        df['クラスタ'] = km.labels_
        st.dataframe(df[['年齢','性別','肌質','評価','クラスタ']])

        # セグメント分布
        st.subheader("年代×クラスタ 分布")
        seg = pd.crosstab(df['年齢'], df['クラスタ'])
        st.dataframe(seg)

        # CSVダウンロード
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("CSVダウンロード", data=csv, file_name="cosme_reviews.csv", mime="text/csv")
