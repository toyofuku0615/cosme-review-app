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
# ページ設定 & 高級感CSS
# =====================
st.set_page_config(page_title="@cosme Review Insight", page_icon="💄", layout="wide")

st.markdown(
    """
    <style>
    /* Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Montserrat', sans-serif;
    }
    body {
        background: linear-gradient(135deg,#FAF8FF 0%, #FFF 40%);
    }
    h1,h2,h3 {
        color:#7B1FA2;
        font-weight:600;
    }
    .stDownloadButton>button, .stButton>button {
        background:#7B1FA2;
        color:#fff;
        border:none;
        border-radius:8px;
        padding:0.5rem 1.2rem;
        font-weight:600;
        transition:0.3s;
    }
    .stDownloadButton>button:hover, .stButton>button:hover {
        background:#9B4DCC;
    }
    .metric-card {
        background:#ffffff; 
        border-radius:12px; 
        box-shadow:0 4px 14px rgba(0,0,0,0.06);
        padding:1.2rem;
        margin-bottom:1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# =====================
# サイドバー
# =====================
with st.sidebar:
    st.header("🔍 レビューURLを入力")
    with st.form("url_form"):
        url_input = st.text_input("@cosmeレビューURL", placeholder="https://www.cosme.net/products/10240630/review/")
        submitted = st.form_submit_button("分析する")
    st.info("※ URLは /review/ で終わるページを入力してください")

# =====================
# レビュー取得関数（最新版HTML構造）
# =====================

def get_reviews(url:str)->pd.DataFrame:
    headers = {"User-Agent":"Mozilla/5.0"}
    reviews = []
    page = 1
    while True:
        resp = requests.get(f"{url}?page={page}", headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        items = soup.select(".p-reviewList__item")
        if not items:
            break
        for item in items:
            # 星評価：整数か小数 例 <span class="bl-reviewRating__score">5.3</span>
            score_tag = item.select_one(".bl-reviewRating__score")
            rating = float(score_tag.text) if score_tag else None
            # プロフィール 例: "30代 女性 乾燥肌"
            profile = item.select_one(".p-reviewList__profile").get_text(" ", strip=True)
            # 本文抜粋
            excerpt = item.select_one(".p-reviewList__comment").get_text(" ", strip=True)
            # 全文リンク
            more = item.select_one("a.p-reviewList__moreLink")
            full_txt = excerpt
            if more and more.has_attr("href"):
                time.sleep(0.4)
                det = requests.get(more["href"], headers=headers, timeout=10)
                det_soup = BeautifulSoup(det.text, "html.parser")
                txt_tag = det_soup.select_one(".p-reviewExpand__text")
                if txt_tag:
                    full_txt = txt_tag.get_text(" ", strip=True)
            # 日付 例: 2024/12/01
            date_tag = item.select_one(".p-reviewList__date")
            date = date_tag.text.strip() if date_tag else ""
            reviews.append({"評価": rating, "属性": profile, "本文": full_txt, "日付": date})
        page += 1
    return pd.DataFrame(reviews)

# =====================
# メイン
# =====================
st.title("💄 @cosme Review Insight")
st.caption("年代×肌質で読み解くコスメレビュー ─ 感情分析 / クラスタリング / セグメント 可視化")

if submitted and url_input:
    with st.spinner("レビュー取得中..."):
        df = get_reviews(url_input)

    if df.empty:
        st.error("レビューが取得できませんでした。URLを確認してください。")
        st.stop()

    st.success(f"{len(df)} 件のレビューを取得しました！")

    # プロフィール分解
    df[["年代","性別","肌質"]] = df["属性"].str.extract(r"(\d+代)?\s+(男性|女性)?\s+(.*)")

    # -------- メトリクスカード --------
    with st.container():
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("平均評価", f"{df['評価'].mean():.2f}")
        col_b.metric("ポジティブ率", f"{(df['評価']>=5).mean()*100:.1f}%")
        col_c.metric("クラスタ数", "3")

    # -------- グラフ：年代 & 肌質 --------
    g1, g2 = st.columns(2)
    with g1:
        st.subheader("📊 年代別 平均評価")
        fig, ax = plt.subplots()
        df.groupby("年代")["評価"].mean().plot.bar(color="#9C27B0", edgecolor="black", ax=ax)
        ax.set_ylabel("平均評価"); st.pyplot(fig)
    with g2:
        st.subheader("📊 肌質別 平均評価")
        fig2, ax2 = plt.subplots()
        df.groupby("肌質")["評価"].mean().plot.bar(color="#FFB300", edgecolor="black", ax=ax2)
        ax2.set_ylabel("平均評価"); st.pyplot(fig2)

    # -------- 感情スコア分布 --------
    st.subheader("😊 感情スコア分布")
    df["sentiment"] = df["本文"].apply(lambda x: TextBlob(x).sentiment.polarity)
    fig3, ax3 = plt.subplots()
    df["sentiment"].hist(bins=30, color="#26A69A", edgecolor="white", ax=ax3)
    ax3.set_xlabel("Polarity"); ax3.set_ylabel("Count")
    st.pyplot(fig3)

    # -------- 時系列トレンド --------
    st.subheader("📈 時系列 平均評価トレンド")
    df["レビュー日"] = pd.to_datetime(df["日付"], errors="coerce")
    trend = df.dropna(subset=["レビュー日"]).groupby("レビュー日")["評価"].mean()
    fig4, ax4 = plt.subplots()
    trend.plot(ax=ax4, color="#FF7043", marker="o"); ax4.set_ylabel("平均評価")
    st.pyplot(fig4)

    # -------- クラスタリング --------
    st.subheader("👥 レビュー本文クラスタリング (3-clusters)")
    tfidf = TfidfVectorizer(max_features=50, stop_words="japanese"); X = tfidf.fit_transform(df["本文"])
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X)
    df["クラスタ"] = kmeans.labels_
    st.dataframe(df[["年代","肌質","性別","評価","クラスタ"]])

    # -------- セグメントクロス --------
    st.subheader("🔍 年代 × クラスタ")
    seg = pd.crosstab(df["年代"], df["クラスタ"])
    st.dataframe(seg.style.background_gradient(cmap="PuRd"))

    # -------- CSV DL --------
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("データCSVダウンロード", data=csv, file_name="cosme_reviews.csv", mime="text/csv")
