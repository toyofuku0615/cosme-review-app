import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from textblob import TextBlob
import time

# =====================
# ページ設定 & CSS
# =====================
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

# =====================
# サイドバー
# =====================
with st.sidebar:
    st.header("🔍 レビューURLを入力")
    with st.form("sidebar_form"):
        url_input = st.text_input("@cosmeレビューURL", placeholder="例: https://www.cosme.net/products/10240630/review/")
        submitted = st.form_submit_button("分析する")
    st.info("※ URLは /review/ で終わるページを入力してください")

# =====================
# レビュー取得関数（静的HTML + 幅広いセレクタ検出）
# =====================
def get_reviews(url: str) -> pd.DataFrame:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept-Language": "ja-JP,ja;q=0.9"
    }
    reviews = []
    page = 1
    while True:
        resp = requests.get(f"{url}?page={page}", headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        # ルートとなるレビュー要素
        items = soup.select("#product-review-list > div")
        if not items:
            break
        for item in items:
            # 星評価
            star_tag = item.select_one("div.body > div.rating.clearfix > p.reviewer-rating.rtg-4")
            rating = None
            if star_tag:
                try:
                    rating = float(re.sub(r"[^0-9]", "", star_tag.get_text()))
                except:
                    rating = None
            # プロフィール
            prof_tag = item.select_one("div.head > div.reviewer-info")
            profile_txt = prof_tag.get_text(" ", strip=True) if prof_tag else ""
            # 本文（抜粋 or 全文）
            body_tag = item.select_one("div.body > p")
            full_txt = body_tag.get_text(strip=True) if body_tag else ""
            # 日付
            date_tag = item.select_one("div.body > div.rating.clearfix > p.mobile-date")
            date_txt = date_tag.get_text(strip=True) if date_tag else ""
            reviews.append({"評価": rating, "属性": profile_txt, "本文": full_txt, "日付": date_txt})
        page += 1
    return pd.DataFrame(reviews)

# =====================
# メイン表示
# =====================
st.title("💄 @cosme Review Insight")
st.caption("年代×肌質で読み解くコスメレビュー ─ 感情分析・クラスタリング・セグメント分析")

if submitted and url_input:
    with st.spinner("レビュー取得中…"):
        df = get_reviews(url_input)
    if df.empty:
        st.error("⚠️ レビューが取得できませんでした。URLを確認してください。")
    else:
        st.success(f"✅ {len(df)} 件のレビューを取得しました！")
        # プロフィール分解
        df[["年代","性別","肌質"]] = df["属性"].str.extract(r"(\d+代)\s+(男性|女性)?\s*(.*)")
        # メトリクスカード
        c1,c2,c3 = st.columns(3)
        c1.markdown(f"""<div class='metric-card'><h3>平均評価</h3><p><strong>{df['評価'].mean():.2f}</strong></p></div>""", unsafe_allow_html=True)
        c2.markdown(f"""<div class='metric-card'><h3>ポジティブ率</h3><p><strong>{(df['評価']>=5).mean()*100:.1f}%</strong></p></div>""", unsafe_allow_html=True)
        c3.markdown(f"""<div class='metric-card'><h3>レビュー数</h3><p><strong>{len(df)}</strong></p></div>""", unsafe_allow_html=True)
        # グラフ
        g1,g2 = st.columns(2)
        with g1:
            st.subheader("年代別平均評価")
            fig,ax = plt.subplots(); df.groupby("年代")["評価"].mean().plot.bar(ax=ax, edgecolor="black"); st.pyplot(fig)
        with g2:
            st.subheader("肌質別平均評価")
            fig2,ax2 = plt.subplots(); df.groupby("肌質")["評価"].mean().plot.bar(ax=ax2, edgecolor="black"); st.pyplot(fig2)
        # 感情スコア分布
        st.subheader("感情スコア分布")
        df['sentiment'] = df['本文'].apply(lambda x: TextBlob(x).sentiment.polarity)
        fig3,ax3 = plt.subplots(); df['sentiment'].hist(bins=30,ax=ax3); st.pyplot(fig3)
        # 時系列トレンド
        st.subheader("評価の時系列トレンド")
        df['レビュー日'] = pd.to_datetime(df['日付'], errors='coerce')
        trend = df.dropna(subset=['レビュー日']).groupby('レビュー日')['評価'].mean()
        fig4,ax4 = plt.subplots(); trend.plot(marker='o',ax=ax4); st.pyplot(fig4)
        # クラスタリング
        st.subheader("レビュー本文クラスタリング (3 clusters)")
        tfidf = TfidfVectorizer(max_features=50,stop_words="japanese"); X = tfidf.fit_transform(df['本文'])
        km = KMeans(n_clusters=3,random_state=42,n_init=10).fit(X); df['クラスタ']=km.labels_
        st.dataframe(df[['年代','性別','肌質','評価','クラスタ']])
        # セグメント
        st.subheader("年代×クラスタ 分布")
        seg = pd.crosstab(df['年代'],df['クラスタ'])
        st.dataframe(seg.style.background_gradient(cmap='PuRd'))
        # ダウンロード
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("CSVダウンロード", data=csv, file_name="cosme_reviews.csv", mime="text/csv")
