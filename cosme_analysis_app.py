import pandas as pd
import requests
from bs4 import BeautifulSoup
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from textblob import TextBlob
import datetime
import time

# --- ページ設定 ---
st.set_page_config(page_title="💄 @cosmeレビュー分析サイト", layout="wide", page_icon="💖")

# カスタムCSSで装飾
st.markdown("""
<style>
body {background-color: #FFF7F8;}
.stButton>button {background-color: #E91E63; color: #fff; border: none;}
.stSidebar .st-form {padding: 1em; background: #fff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
h1, h2, h3 {color: #C2185B;}
.stDownloadButton>button {background-color: #C2185B; color: #fff;}
</style>
""", unsafe_allow_html=True)

# サイドバー入力フォーム
with st.sidebar:
    st.header("🔍 レビュー分析")
    st.write("@cosmeのレビューURLを入力し、分析開始をクリックしてください。")
    with st.form("sidebar_form"):
        url_input = st.text_input("レビューURL", placeholder="例: https://www.cosme.net/products/10240630/review")
        submitted = st.form_submit_button("🔎 分析開始")
    st.markdown("---")

# メインエリア
st.title("💄 @cosmeレビュー分析サイト")
st.write("**年代・肌質別の評価傾向**、**感情分析**、**クラスタリング**、**ユーザーセグメント**を一画面で確認できます。")

# レビュー取得ロジック（ページネーション＆全文取得対応）
def get_reviews(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    reviews = []
    page = 1
    while True:
        # ページ番号付与
        paged_url = f"{url}?pageno={page}"
        resp = requests.get(paged_url, headers=headers)
        soup = BeautifulSoup(resp.text, "html.parser")
        items = soup.select(".c-section-review__item")
        if not items:
            break
        for item in items:
            # 評価
            star = item.select_one(".c-section-review__rating i")
            rating = None
            if star and star.has_attr('class'):
                for cls in star['class']:
                    if cls.startswith('star'):
                        try:
                            rating = int(re.sub(r'[^0-9]', '', cls))
                        except:
                            pass
            # 属性
            profile = item.select_one(".c-section-review__profile")
            profile_txt = profile.get_text(strip=True) if profile else ""
            # テキスト
            excerpt = item.select_one(".c-section-review__text").get_text(strip=True) if item.select_one(".c-section-review__text") else ""
            more = item.select_one("a.c-section-review__readMore")
            full_txt = excerpt
            if more and more.has_attr('href'):
                # 全文ページ取得
                time.sleep(0.5)
                det = requests.get(more['href'], headers=headers)
                dsoup = BeautifulSoup(det.text, 'html.parser')
                detail = dsoup.select_one(".c-section-reviewDetail__text")
                if detail:
                    full_txt = detail.get_text(strip=True)
            # 日付
            date_elem = item.select_one(".c-section-review__date")
            date_txt = date_elem.get_text(strip=True) if date_elem else ""
            reviews.append({"評価": rating, "属性": profile_txt, "本文": full_txt, "日付": date_txt})
        page += 1
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
        # レイアウト
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 年代別評価平均")
            fig1, ax1 = plt.subplots()
            df.groupby("年代")["評価"].mean().plot(kind="bar", ax=ax1, edgecolor="black")
            ax1.set_ylabel("平均評価")
            st.pyplot(fig1)
        with col2:
            st.subheader("📊 肌質別評価平均")
            fig2, ax2 = plt.subplots()
            df.groupby("肌質")["評価"].mean().plot(kind="bar", ax=ax2, edgecolor="black")
            ax2.set_ylabel("平均評価")
            st.pyplot(fig2)
        st.subheader("😊 感情分析（ポジ/ネガ分類）")
        df["感情スコア"] = df["本文"].apply(lambda x: TextBlob(x).sentiment.polarity)
        fig3, ax3 = plt.subplots()
        df["感情スコア"].hist(bins=20, ax=ax3)
        ax3.set_xlabel("感情スコア")
        ax3.set_ylabel("レビュー数")
        st.pyplot(fig3)
        st.subheader("📈 評価の時系列トレンド")
        df["レビュー日"] = pd.to_datetime(df["日付"], errors='coerce')
        trend = df.dropna(subset=["レビュー日"]).groupby("レビュー日")["評価"].mean()
        fig4, ax4 = plt.subplots()
        trend.plot(ax=ax4, marker='o')
        ax4.set_ylabel("平均評価")
        st.pyplot(fig4)
        st.subheader("👥 属性別レビュークラスタリング")
        tfidf = TfidfVectorizer(max_features=30, stop_words="japanese")
        X = tfidf.fit_transform(df["本文"])
        km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X)
        df["クラスタ"] = km.labels_
        st.dataframe(df[["年代","肌質","性別","評価","クラスタ"]])
        st.subheader("🔍 ユーザーセグメント（年代×クラスタ）")
        seg = pd.crosstab(df["年代"], df["クラスタ"])
        st.dataframe(seg)
        # CSVダウンロード
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("CSVをダウンロード", data=csv, file_name="cosme_reviews.csv", mime="text/csv")
