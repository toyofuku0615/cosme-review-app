import pandas as pd
import requests
from bs4 import BeautifulSoup
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from textblob import TextBlob
import datetime

# --- ページ設定 ---
st.set_page_config(page_title="💄 @cosmeレビュー分析サイト", layout="wide", page_icon="💖")

# カスタムCSSで装飾
st.markdown("""
<style>
body {background-color: #FFF7F8;}
.stButton>button {background-color: #E91E63; color: white;}
.stForm label {font-size: 1.1em;}
h1, h2, h3 {color: #C2185B;}
</style>
""", unsafe_allow_html=True)

# サイドバー
with st.sidebar:
    st.header("🔍 レビュー分析")
    st.write("@cosmeレビューURLを入力して、ボタンを押してください。")
    with st.form("sidebar_form"):
        url_input = st.text_input("レビューURL", placeholder="例: https://www.cosme.net/product/product_id/10104342/review")
        submitted = st.form_submit_button("🔎 分析開始")
    st.markdown("---")
    st.write("#### 📖 使い方")
    st.write("1. 上記にURLを入力\n2. 分析開始をクリック\n3. 結果をお楽しみください！")
    st.markdown("---")
    st.write("### 🤝 サポート")
    st.write("不具合や要望はご連絡ください。")

# メインエリア
st.title("💄 @cosmeレビュー分析サイト")
st.write("**年代・肌質別の評価傾向**、**感情分析**、**クラスタリング**、**ユーザーセグメント**を一画面で確認できます。")

# レビュー取得関数
def get_reviews(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")
    reviews = []

    for block in soup.select(".c-section-review__item"):
        star = block.select_one(".c-section-review__rating i")
        rating = int(star["class"][-1].replace("star", "")) if star and "star" in star["class"][-1] else None
        profile_txt = block.select_one(".c-section-review__profile").get_text(strip=True) if block.select_one(".c-section-review__profile") else ""
        excerpt = block.select_one(".c-section-review__text").get_text(strip=True) if block.select_one(".c-section-review__text") else ""
        more = block.select_one("a.c-section-review__readMore")
        full_txt = excerpt
        if more:
            detail = requests.get(more["href"], headers=headers)
            dsoup = BeautifulSoup(detail.text, "html.parser")
            full = dsoup.select_one(".c-section-reviewDetail__text").get_text(strip=True)
            full_txt = full if full else excerpt
        date_txt = block.select_one(".c-section-review__date").get_text(strip=True) if block.select_one(".c-section-review__date") else ""
        reviews.append({"評価": rating, "属性": profile_txt, "本文": full_txt, "日付": date_txt})
    return pd.DataFrame(reviews)

# 分析実行
if submitted and url_input:
    with st.spinner("レビューを取得中..."):
        df = get_reviews(url_input)
    if df.empty:
        st.error("⚠️ レビューが見つかりません。URLをご確認ください。")
    else:
        st.success(f"✅ {len(df)} 件のレビューを取得しました！")

        # 属性分解
        df[["年代","肌質","性別"]] = df["属性"].str.extract(r"(\d+代)?・(.*?)・(.*?)$")

        # レイアウト：2段組
        col1, col2 = st.columns(2)

        # 年代別評価平均
        with col1:
            st.subheader("📊 年代別評価平均")
            fig1, ax1 = plt.subplots()
            df.groupby("年代")["評価"].mean().plot(kind="bar", ax=ax1, edgecolor="black")
            ax1.set_ylabel("平均評価")
            st.pyplot(fig1)

        # 肌質別評価平均
        with col2:
            st.subheader("📊 肌質別評価平均")
            fig2, ax2 = plt.subplots()
            df.groupby("肌質")["評価"].mean().plot(kind="bar", ax=ax2, edgecolor="black")
            ax2.set_ylabel("平均評価")
            st.pyplot(fig2)

        # 感情分析
        st.subheader("😊 感情分析（ポジ/ネガ分類）")
        df["感情スコア"] = df["本文"].apply(lambda x: TextBlob(x).sentiment.polarity)
        fig3, ax3 = plt.subplots()
        df["感情スコア"].hist(bins=20, ax=ax3)
        ax3.set_xlabel("感情スコア")
        ax3.set_ylabel("レビュー数")
        st.pyplot(fig3)

        # 時系列トレンド
        st.subheader("📈 評価の時系列トレンド")
        df["レビュー日"] = pd.to_datetime(df["日付"], errors='coerce')
        trend = df.dropna(subset=["レビュー日"]).groupby("レビュー日")["評価"].mean()
        fig4, ax4 = plt.subplots()
        trend.plot(ax=ax4, marker='o')
        ax4.set_ylabel("平均評価")
        st.pyplot(fig4)

        # クラスタリング
        st.subheader("👥 属性別レビュークラスタリング")
        tfidf = TfidfVectorizer(max_features=30, stop_words="japanese")
        X = tfidf.fit_transform(df["本文"])
        km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X)
        df["クラスタ"] = km.labels_
        st.dataframe(df[["年代","肌質","性別","評価","クラスタ"]])

        # ユーザーセグメント
        st.subheader("🔍 ユーザーセグメント（年代×クラスタ）")
        seg = pd.crosstab(df["年代"], df["クラスタ"])
        st.dataframe(seg)

        # CSVダウンロード
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("CSVをダウンロード", data=csv, file_name="cosme_reviews.csv", mime="text/csv")
