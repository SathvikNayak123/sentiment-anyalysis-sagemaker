from selenium.common.exceptions import StaleElementReferenceException, NoSuchElementException
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
import pandas as pd
import os

class ScrapeData:
    def __init__(self,url,path):
        self.amazon_reviews=[]
        self.count=0
        self.max_reviews=100
        self.base_url='https://amazon.com'
        self.page_url=url

        self.service = Service(executable_path=path)
        self.options = webdriver.ChromeOptions()
        self.driver = webdriver.Chrome(service=self.service, options=self.options)
        
    def get_product_links(self):
        all_products = self.driver.find_elements(By.XPATH, f"//a[@class='a-link-normal s-underline-text s-underline-link-text s-link-style a-text-normal']")
        all_products_links=[]
        for product in all_products:
            product_url = product.get_attribute("href")
            # Ensure the URL is correctly formatted
            if not product_url.startswith('https'):
                product_url = self.base_url + product_url
            all_products_links.append(product_url)
        return all_products_links

    def scrapeReviews(self,max_reviews):
        self.driver.get(self.page_url)
        while self.count < max_reviews:
            all_product_urls = self.get_product_links()
            for product_url in all_product_urls:
                try: 
                    self.driver.get(product_url)  # Navigate to each product page
                    time.sleep(2)

                    try:
                        # Click on 'See All Reviews' button
                        see_all_button = self.driver.find_element(By.XPATH, f"//a[@data-hook='see-all-reviews-link-foot']")
                        see_all_button.click()
                        print('See All Reviews clicked')
                        time.sleep(2)

                        while self.count < max_reviews:
                            all_reviews = self.driver.find_elements(By.XPATH, f"//span[@data-hook='review-body']")
                            for review in all_reviews:
                                self.amazon_reviews.append(review.text.strip())
                                self.count += 1
                                print(f"Scraped {self.count} reviews.")

                                if self.count >= max_reviews:
                                    print(f"Reached the limit of {max_reviews} reviews.")
                                    break
                            
                            if self.count >= max_reviews:
                                break
                            
                            try:
                                next_button = self.driver.find_element(By.XPATH, f"//li[@class='a-last']")
                                next_link = next_button.find_element(By.TAG_NAME, "a")
                                next_url = next_link.get_attribute("href")

                                # Navigate to the next review page
                                self.driver.get(next_url)
                                print('Next button clicked successfully')
                                time.sleep(2)

                            except NoSuchElementException:
                                print('No more pages for this product.')
                                break  # Exit the loop if there is no "Next" button

                    except NoSuchElementException as e:
                        print(f"Error finding See All Reviews button: {e}")
                        self.amazon_reviews.append("")
                    
                except StaleElementReferenceException as e:
                    print(f"StaleElementReferenceException encountered: {e}")
                    continue  # Skip to the next product in the list

                if self.count >= max_reviews:
                    print(f"Reached the limit of {max_reviews} reviews.")
                    break
            try:
                self.driver.get(self.page_url)
                next_page = self.driver.find_element(By.XPATH, f"//a[@class='s-pagination-item s-pagination-next s-pagination-button s-pagination-separator']")
                next_page_url = next_page.get_attribute("href")

                # Navigate to the next products page
                self.page_url = self.base_url + next_page_url
                print('Next Page')
                time.sleep(2)

            except NoSuchElementException:
                print('No more pages left')
                break  # Exit the loop if there is no "Next page" button

        self.driver.quit()

if __name__=="__main__":
    url='https://www.amazon.com/s?k=redmi+earbuds&crid=3TZZRJIPFBP2E&sprefix=redmi+earbuds%2Caps%2C380&ref=nb_sb_noss_1'

    path="chromedriver.exe"

    obj=ScrapeData(url,path)
    
    obj.scrapeReviews(11000)

    # Create the "artifacts" directory if it doesn't exist
    os.makedirs("artifacts", exist_ok=True)

    # Store the scraped data into a DataFrame
    amazon_df = pd.DataFrame(obj.amazon_reviews, columns=["Review"])
    
    # Save to CSV
    amazon_df.to_csv("artifacts/amazon_data.csv", index=False)
        