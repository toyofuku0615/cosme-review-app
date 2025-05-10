import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
import streamlit as st
import matplotlib.pyplot as plt
import japanize_matplotlib
japanize_matplotlib.japanize()
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from textblob import TextBlob

# ===== ページ設定 =====
st.set_page_config(page_title="@cosme Review Insight", page_icon="💄", layout="wide")

# ===== サイドバー設定 =====nwith st.sidebar:
    st.header("🔍 レビュー分析設定")
    url_input = st.text_input("@cosmeの商品ページ/レビューURLを入力", placeholder="例: https://www.cosme.net/products/10240630/review/")
    submitted = st.button("分析開始")

# ===== レビュー取得関数 =====
def get_reviews(url: str, max_pages: int = 3) -> pd.DataFrame:
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0", "Accept-Language": "ja-JP,jp;q=0.9"})
    # URLが商品ページなら /review/ を付与
    if not url.endswith("/review/"):
        url = url.rstrip("/") + "/review/"
    reviews = []
    for page in range(1, max_pages+1):
        resp = session.get(f"{url}?page={page}", timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        items = soup.select("#product-review-list > div")
        if not items:
            break
        for item in items:
            # 評価
            score_tag = item.select_one("div.body div.rating.clearfix p.reviewer-rating")
            rating = float(re.sub(r"[^0-9.]", "", score_tag.text)) if score_tag else None
            # 属性
            info_tag = item.select_one("div.head div.reviewer-info")
            profile_txt = info_tag.get_text(" ", strip=True) if info_tag else ""
            # 本文
            body_tag = item.select_one("div.body p:not(.reviewer-rating):not(.mobile-date)")
            body_txt = body_tag.get_text(strip=True) if body_tag else ""
            # 投稿日
            date_tag = item.select_one("div.body div.rating.clearfix p.mobile-date")
            date_txt = date_tag.text.strip() if date_tag else ""
            reviews.append({"評価": rating, "属性": profile_txt, "本文": body_txt, "日付": date_txt})
    return pd.DataFrame(reviews)

# ===== メイン処理 =====
st.title("💄 @cosme Review Insight")
st.write("迅速にレビューを取得・分析します。日本語グラフ対応済み。")

if submitted and url_input:
    with st.spinner("レビュー取得中…"):
        df = get_reviews(url_input)
    if df.empty:
        st.error("⚠️ レビューが取得できませんでした。URLを確認してください。")
        st.stop()
    st.success(f"✅ {len(df)} 件のレビューを取得しました！")
    # 属性分解：年代、性別、肌質
    df[['年代','性別','肌質']] = df['属性'].str.extract(r'(\d+代)\s*(男性|女性)\s*(.+)')
    df[['年代','性別','肌質']] = df[['年代','性別','肌質']].fillna('不明')
    # メトリクス表示
    col1,col2,col3 = st.columns(3)
    col1.metric("平均評価", f"{df['評価'].mean():.2f}")
    col2.metric("最高評価率", f"{(df['評価']>=5).mean()*100:.1f}%")
    col3.metric("レビュー数", f"{len(df)}")
    # グラフ：年代別平均評価
    st.subheader("📊 年代別平均評価")
    fig1, ax1 = plt.subplots()
    df.groupby('年代')['評価'].mean().plot(kind='bar', ax=ax1)
    ax1.set_xlabel('年代')
    ax1.set_ylabel('平均評価')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig1)
    # グラフ：性別別平均評価
    st.subheader("📊 性別別平均評価")
    fig1b, ax1b = plt.subplots()
    df.groupby('性別')['評価'].mean().plot(kind='bar', ax=ax1b)
    ax1b.set_xlabel('性別')
    ax1b.set_ylabel('平均評価')
    plt.xticks(rotation=0)
    st.pyplot(fig1b)
    # グラフ：肌質別平均評価
    st.subheader("📊 肌質別平均評価")
    fig2, ax2 = plt.subplots()
    df.groupby('肌質')['評価'].mean().plot(kind='bar', ax=ax2)
    ax2.set_xlabel('肌質')
    ax2.set_ylabel('平均評価')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig2)
    # 感情スコア分布
    st.subheader("😊 感情スコア分布")
    df['感情'] = df['本文'].apply(lambda x: TextBlob(x).sentiment.polarity)
    fig3, ax3 = plt.subplots()
    df['感情'].hist(bins=20, ax=ax3)
    ax3.set_xlabel('感情スコア')
    ax3.set_ylabel('件数')
    st.pyplot(fig3)
    # クラスタリング
    st.subheader("👥 レビュー本文クラスタリング (3 clusters)")
    tfidf = TfidfVectorizer(max_features=30)
    X = tfidf.fit_transform(df['本文'])
    km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X)
    df['クラスタ'] = km.labels_
    st.dataframe(df[['年代','性別','肌質','評価','クラスタ']])
    # 年代×クラスタ分布
    st.subheader("🔍 年代 × クラスタ 分布")
    seg = pd.crosstab(df['年代'], df['クラスタ'])
    st.dataframe(seg)
    # CSVダウンロード
    csv = df.to_csv(index=False).encode('utf-8-sig')
    st.download_button("CSVダウンロード", data=csv, file_name="cosme_reviews.csv", mime="text/csv")
