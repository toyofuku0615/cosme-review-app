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

# ===== Streamlit設定 =====
st.set_page_config(page_title="@cosme Review Insight", page_icon="💄", layout="wide")

# ===== CSS =====
st.markdown(
    """
    <style>
    body { background-color: #FAF8FF; }
    h1, h2, h3 { color: #7B1FA2; font-weight: bold; }
    .stSidebar { background-color: #FFFFFF; padding: 1rem; border-radius: 8px; }
    .stButton>button, .stDownloadButton>button { background-color: #7B1FA2; color: white; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

# ===== サイドバー =====
with st.sidebar:
    st.header("🔍 レビュー分析設定")
    url_input = st.text_input("@cosmeの商品ページ/レビューURLを入力", placeholder="例: https://www.cosme.net/products/10240630/review/")
    max_pages = st.slider("最大ページ数", 1, 5, 3)
    submitted = st.button("分析開始")

# ===== レビュー取得 =====
def get_reviews(url: str, max_pages: int) -> pd.DataFrame:
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "ja-JP,ja;q=0.9"
    })
    # 商品ページURLをレビューURLに変換
    if not url.endswith("/review/"):
        url = url.rstrip("/") + "/review/"

    reviews = []
    for page in range(1, max_pages + 1):
        resp = session.get(f"{url}?page={page}", timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        items = soup.select("#product-review-list > div")
        if not items:
            break
        for item in items:
            # 評価
            score_tag = item.select_one("div.body div.rating.clearfix p.reviewer-rating")
            rating = None
            if score_tag:
                num = re.sub(r"[^0-9.]", "", score_tag.text)
                rating = float(num) if num else None
            # 年齢・肌質・性別: div.tag-list.clearfix li に順に入っている
            tags = item.select("div.body div.tag-list.clearfix li")
            tag_texts = [t.get_text(strip=True) for t in tags]
            age = tag_texts[0] if len(tag_texts) > 0 else "不明"
            skin = tag_texts[1] if len(tag_texts) > 1 else "不明"
            sex = tag_texts[2] if len(tag_texts) > 2 else "不明"
            # 本文
            body_tag = item.select_one("div.body p:not(.reviewer-rating):not(.mobile-date)")
            body_txt = body_tag.get_text(strip=True) if body_tag else ""
            # 日付
            date_tag = item.select_one("div.body div.rating.clearfix p.mobile-date")
            date_txt = date_tag.text.strip() if date_tag else ""
            reviews.append({"評価": rating, "年齢": age, "肌質": skin, "性別": sex, "本文": body_txt, "日付": date_txt})
    return pd.DataFrame(reviews)

# ===== メイン =====
st.title("💄 @cosme Review Insight")
st.write("迅速にレビューを取得・分析します。ページ数調整可能。日本語フォント対応。")

if submitted and url_input:
    with st.spinner("レビュー取得中…"):
        df = get_reviews(url_input, max_pages)
    if df.empty:
        st.error("⚠️ レビューが取得できませんでした。URLを確認してください。")
        st.stop()

    st.success(f"✅ {len(df)} 件のレビューを取得しました！")
    # 属性分解（年代・性別・肌質）
    parts = df['属性'].str.split(r'[・／\s]+', expand=True)
    df['年代'] = parts.iloc[:, 0].fillna('不明')
    df['性別'] = parts.iloc[:, 1].fillna('不明')
    df['肌質'] = parts.iloc[:, 2].fillna('不明')

    # メトリクス
    c1, c2, c3 = st.columns(3)
    c1.metric("平均評価", f"{df['評価'].mean():.2f}")
    c2.metric("ポジティブ率", f"{(df['評価']>=5).mean()*100:.1f}%")
    c3.metric("レビュー数", f"{len(df)}")

    # グラフ：年代別
    st.subheader("📊 年代別平均評価")
    fig1, ax1 = plt.subplots()
    df.groupby('年代')['評価'].mean().plot.bar(ax=ax1)
    ax1.set_xlabel('年代')
    ax1.set_ylabel('平均評価')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig1)

    # グラフ：性別別
    st.subheader("📊 性別別平均評価")
    fig2, ax2 = plt.subplots()
    df.groupby('性別')['評価'].mean().plot.bar(ax=ax2)
    ax2.set_xlabel('性別')
    ax2.set_ylabel('平均評価')
    plt.xticks(rotation=0)
    st.pyplot(fig2)

    # グラフ：肌質別
    st.subheader("📊 肌質別平均評価")
    fig3, ax3 = plt.subplots()
    df.groupby('肌質')['評価'].mean().plot.bar(ax=ax3)
    ax3.set_xlabel('肌質')
    ax3.set_ylabel('平均評価')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig3)

    # 感情スコア分布
    st.subheader("😊 感情スコア分布")
    df['感情'] = df['本文'].apply(lambda x: TextBlob(x).sentiment.polarity)
    fig4, ax4 = plt.subplots()
    df['感情'].hist(bins=20, ax=ax4)
    ax4.set_xlabel('感情スコア')
    ax4.set_ylabel('件数')
    st.pyplot(fig4)

    # クラスタリング
    st.subheader("👥 レビュークラスタリング (3 clusters)")
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
    st.download_button(
        label="CSVダウンロード",
        data=df.to_csv(index=False).encode('utf-8-sig'),
        file_name="cosme_reviews.csv",
        mime="text/csv"
    )
