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
# ページ設定 & ハイクラスCSS
# =====================
st.set_page_config(page_title="@cosme Review Insight", page_icon="💄", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Montserrat', sans-serif; }
    body { background: linear-gradient(135deg, #FAF8FF 0%, #FFF 40%); }
    h1,h2,h3 { color:#7B1FA2; font-weight:600; }
    .stButton>button, .stDownloadButton>button {
        background:#7B1FA2; color:#fff; border:none; border-radius:8px;
        padding:0.5rem 1.2rem; font-weight:600; transition:0.3s;
    }
    .stButton>button:hover, .stDownloadButton>button:hover { background:#9B4DCC; }
    .metric-card {
        background:#fff; border-radius:12px; box-shadow:0 4px 14px rgba(0,0,0,0.06);
        padding:1rem; margin:0.5rem 0;
    }
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
# レビュー取得関数（最新HTML構造対応）
# =====================
def get_reviews(url: str) -> pd.DataFrame:
    headers = {"User-Agent": "Mozilla/5.0"}
    reviews = []
    page = 1
    while True:
        resp = requests.get(f"{url}?page={page}", headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        items = soup.select(".p-reviewList__item")
        if not items:
            break
        for item in items:
            # 評価スコア（小数対応）
            score_tag = item.select_one(".bl-reviewRating__score")
            rating = float(score_tag.text) if score_tag else None
            # プロフィール
            prof_tag = item.select_one(".p-reviewList__profile")
            profile_txt = prof_tag.get_text(" ", strip=True) if prof_tag else ""
            # 抜粋テキスト
            excerpt_tag = item.select_one(".p-reviewList__comment")
            excerpt = excerpt_tag.get_text(" ", strip=True) if excerpt_tag else ""
            # 『続きを読む』リンクで全文取得
            more = item.select_one("a.p-reviewList__moreLink")
            full_txt = excerpt
            if more and more.has_attr("href"):
                time.sleep(0.3)
                det = requests.get(more['href'], headers=headers, timeout=10)
                det_soup = BeautifulSoup(det.text, "html.parser")
                txt_tag = det_soup.select_one(".p-reviewExpand__text")
                if txt_tag:
                    full_txt = txt_tag.get_text(" ", strip=True)
            # 日付
            date_tag = item.select_one(".p-reviewList__date")
            date_txt = date_tag.text.strip() if date_tag else ""
            reviews.append({
                "評価": rating,
                "属性": profile_txt,
                "本文": full_txt,
                "日付": date_txt
            })
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
        df[["年代","性別","肌質"]] = df["属性"].str.extract(r"(\d+代)\s+(男性|女性)\s+(.*)")
        # メトリクスカード
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""<div class='metric-card'><h3>平均評価</h3><p><strong>{:.2f}</strong></p></div>""".format(df['評価'].mean()), unsafe_allow_html=True)
        with col2:
            pos_rate = (df['評価'] >= 5).mean() * 100
            st.markdown("""<div class='metric-card'><h3>ポジティブ率</h3><p><strong>{:.1f}%</strong></p></div>""".format(pos_rate), unsafe_allow_html=True)
        with col3:
            st.markdown("""<div class='metric-card'><h3>クラスタ数</h3><p><strong>3</strong></p></div>""", unsafe_allow_html=True)
        # グラフ表示
        g1, g2 = st.columns(2)
        with g1:
            st.subheader("年代別平均評価")
            fig, ax = plt.subplots()
            df.groupby("年代")["評価"].mean().plot.bar(color="#9C27B0", edgecolor="black", ax=ax)
            ax.set_ylabel("平均評価")
            st.pyplot(fig)
        with g2:
            st.subheader("肌質別平均評価")
            fig2, ax2 = plt.subplots()
            df.groupby("肌質")["評価"].mean().plot.bar(color="#FFB300", edgecolor="black", ax=ax2)
            ax2.set_ylabel("平均評価")
            st.pyplot(fig2)
        # 感情スコア分布
        st.subheader("感情スコア分布")
        df['sentiment'] = df['本文'].apply(lambda x: TextBlob(x).sentiment.polarity)
        fig3, ax3 = plt.subplots()
        df['sentiment'].hist(bins=30, color="#26A69A", edgecolor="white", ax=ax3)
        ax3.set_xlabel("Polarity")
        ax3.set_ylabel("件数")
        st.pyplot(fig3)
        # 時系列トレンド
        st.subheader("評価の時系列トレンド")
        df['レビュー日'] = pd.to_datetime(df['日付'], errors='coerce')
        trend = df.dropna(subset=['レビュー日']).groupby('レビュー日')['評価'].mean()
        fig4, ax4 = plt.subplots()
        trend.plot(marker='o', color="#FF7043", ax=ax4)
        ax4.set_ylabel("平均評価")
        st.pyplot(fig4)
        # クラスタリング
        st.subheader("レビュー本文クラスタリング (3 clusters)")
        tfidf = TfidfVectorizer(max_features=50, stop_words="japanese")
        X = tfidf.fit_transform(df['本文'])
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X)
        df['クラスタ'] = kmeans.labels_
        st.dataframe(df[['年代','性別','肌質','評価','クラスタ']])
        # ユーザーセグメント
        st.subheader("年代 × クラスタ 分布")
        seg = pd.crosstab(df['年代'], df['クラスタ'])
        st.dataframe(seg.style.background_gradient(cmap='PuRd'))
        # CSVダウンロード
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("CSVダウンロード", data=csv, file_name="cosme_reviews.csv", mime="text/csv")
