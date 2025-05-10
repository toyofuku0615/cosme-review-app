import pandas as pd
import requests
from bs4 import BeautifulSoup
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from textblob import TextBlob
import datetime

st.set_page_config(page_title="@cosmeレビュー分析アプリ", layout="wide")
st.title("💄 @cosmeレビュー分析サイト")
st.markdown("""
このサイトでは、@cosmeのレビューURLを入力することで、
年代別・肌質別の評価傾向やレビューの特徴語、感情傾向、ユーザークラスタを自動的に分析します。
""")

with st.form("url_form"):
    url_input = st.text_input("@cosmeのレビューURLを入力してください", placeholder="例: https://www.cosme.net/product/product_id/10104342/review")
    submitted = st.form_submit_button("検索")

def get_reviews(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")
    reviews = []

    for block in soup.select(".review__item"):
        rating = block.select_one(".review__rating")
        profile = block.select_one(".review__profile")
        text = block.select_one(".review__body")
        date = block.select_one(".review__date")

        reviews.append({
            "評価": int(rating.get_text(strip=True)[0]) if rating else None,
            "属性": profile.get_text(strip=True) if profile else "",
            "本文": text.get_text(strip=True) if text else "",
            "日付": date.get_text(strip=True) if date else ""
        })
    return pd.DataFrame(reviews)

if submitted and url_input:
    try:
        df = get_reviews(url_input)
        st.success(f"✅ {len(df)} 件のレビューを取得しました")

        df[["年代", "肌質", "性別"]] = df["属性"].str.extract(r"(\\d+代)?・(.*?)・(.*?)$")

        st.subheader("😊 感情分析（ポジティブ/ネガティブ）")
        df["感情スコア"] = df["本文"].apply(lambda x: TextBlob(x).sentiment.polarity if x else 0)
        fig3, ax3 = plt.subplots()
        df["感情スコア"].hist(bins=20, ax=ax3)
        ax3.set_title("感情スコアの分布")
        st.pyplot(fig3)

        if df["日付"].notna().all():
            try:
                df["レビュー日"] = pd.to_datetime(df["日付"], errors='coerce')
                df_sorted = df.sort_values("レビュー日")
                df_trend = df_sorted.groupby("レビュー日")["評価"].mean()

                st.subheader("📈 評価の時系列トレンド")
                fig4, ax4 = plt.subplots()
                df_trend.plot(ax=ax4)
                ax4.set_ylabel("平均評価")
                st.pyplot(fig4)
            except:
                st.warning("レビュー日が適切に取得できなかったため、時系列トレンドを表示できません")

        st.subheader("👥 属性別レビュー内容のクラスタリング")
        tfidf_vec = TfidfVectorizer(max_features=50, stop_words="japanese")
        tfidf_matrix = tfidf_vec.fit_transform(df["本文"].fillna(""))
        kmeans = KMeans(n_clusters=3, random_state=0, n_init=10).fit(tfidf_matrix)
        df["クラスタ"] = kmeans.labels_
        st.dataframe(df[["年代", "肌質", "性別", "評価", "クラスタ"]])

        st.subheader("🔍 ユーザーごとのセグメント（年代×クラスタ）")
        seg_table = pd.crosstab(df["年代"], df["クラスタ"])
        st.dataframe(seg_table)

    except Exception as e:
        st.error(f"❌ エラーが発生しました：{e}")
