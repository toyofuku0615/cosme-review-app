import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import time

def extract_reviews(asin_url):
    headers = {"User-Agent": "Mozilla/5.0"}
    reviews = []
    page = 1

    while True:
        url = f"{asin_url}?page={page}"
        resp = requests.get(url, headers=headers)
        soup = BeautifulSoup(resp.text, "html.parser")

        # レビューアイテムを抽出
        items = soup.select(".c-section-review__item")
        if not items:
            break

        for item in items:
            # 評価星
            star = item.select_one(".c-section-review__rating i")
            rating = int(star["class"][-1].replace("star", "")) if star else None

            # プロフィール（年代・性別・肌質）
            profile_txt = item.select_one(".c-section-review__profile").get_text(strip=True)

            # レビュー冒頭（省略されている場合がある）
            excerpt = item.select_one(".c-section-review__text").get_text(strip=True)

            # 「続きを読む」リンクがあれば全文を取得
            more_link = item.select_one("a.c-section-review__readMore")
            if more_link:
                time.sleep(0.5)
                detail = requests.get(more_link["href"], headers=headers)
                detail_soup = BeautifulSoup(detail.text, "html.parser")
                full_txt = detail_soup.select_one(".c-section-reviewDetail__text").get_text(strip=True)
            else:
                full_txt = excerpt

            # 投稿日
            date_txt = item.select_one(".c-section-review__date").get_text(strip=True)

            reviews.append({
                "評価": rating,
                "属性": profile_txt,
                "本文": full_txt,
                "日付": date_txt
            })

        page += 1
        # 最大ページ数など条件でブレイクしてもOK

    return pd.DataFrame(reviews)

if __name__ == "__main__":
    df = extract_reviews("https://www.cosme.net/products/10240630/review")
    print(df.head())
