from bs4 import BeautifulSoup
import requests
import pandas as pd
import os

class ScrapeData:
    def __init__(self,url,header):
        self.amazon_reviews=[]
        self.url=url
        self.header=header
        self.page=requests.get(url,headers=header)
        self.soup=BeautifulSoup(self.page.content, "html.parser")
        self.count=0
    
    def _scrapeReviews(self,soup):
        try:
            all_reviews=soup.find_all("div",attrs={"class":"a-expander-content reviewText review-text-content a-expander-partial-collapse-content"})
            for review in all_reviews:
                self.amazon_reviews.append(review.get_text())
                self.count+=1
                print(f"Scraped {self.count} reviews.")
        except AttributeError as e:
            print(f"Error parsing reviews: {e}")
            self.amazon_reviews.append("")

    def scrapeReviews(self):
        all_links=self.soup.find_all("a",attrs={'class':'a-link-normal s-underline-text s-underline-link-text s-link-style a-text-normal'})
        for link in all_links:
            product_link = link.get('href')
            if not product_link.startswith('https'):
                product_link="https://amazon.com"+ product_link
            new_page = requests.get(product_link,headers=self.header)
            new_soup = BeautifulSoup(new_page.content, "html.parser")
            self._scrapeReviews(new_soup)

if __name__=="__main__":
    url='https://www.amazon.com/s?k=redmi+earbuds&crid=3TZZRJIPFBP2E&sprefix=redmi+earbuds%2Caps%2C380&ref=nb_sb_noss_1'

    header=({'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36;','Accept-Language':'en-US, en;q=0.5'})

    obj=ScrapeData(url,header)
    
    obj.scrapeReviews()

    # Create the "artifacts" directory if it doesn't exist
    os.makedirs("artifacts", exist_ok=True)

    # Store the scraped data into a DataFrame
    amazon_df = pd.DataFrame(obj.amazon_reviews, columns=["Review"])
    
    # Save to CSV
    amazon_df.to_csv("artifacts/amazon_data.csv", index=False)
        