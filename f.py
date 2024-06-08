import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from wordcloud import WordCloud
from transformers import pipeline
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.by import By
from time import sleep
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Tokenize text
    tokens = nltk.word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join tokens back into string
    clean_text = ' '.join(tokens)
    return clean_text

def get_sentiment(texts):
    pipe = pipeline(task="sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", device=-1)
    preds = pipe(texts)

    response = dict()
    response["labels"] = [pred["label"] for pred in preds]
    response["scores"] = [pred["score"] for pred in preds]
    return response

def generate_wordcloud(texts):
    text = ' '.join(texts)
    wordcloud = WordCloud(width=800, height=400, background_color ='white', contour_width=0).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)


def app():
    st.title("Twitter Sentiment Analyzer")

    st.write("""
    This app allows you to analyze the sentiment of tweets from any specified Twitter user's timeline.
    """)

    twitter_handle = st.sidebar.text_input("Twitter Username:", "")
    # twitter_handle ='https://twitter.com/search?q=%40&src=typed_query'+twitter_handle
    twitter_handle = f"https://x.com/search?q={twitter_handle}&src=typed_query&f=top"

    if st.sidebar.button("Get Tweets"):
        tweet_data = []
        chrome_driver_path = "chromedriver.exe"
        service = Service(executable_path=chrome_driver_path)
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36')  # Example user agent
        options.add_argument('--window-size=1920,1080')
        driver = webdriver.Chrome(service=service, options=options)

     
        # Load the URL
        driver.get(twitter_handle)

        # Login automation
        wait = WebDriverWait(driver, 30)
        username_input = wait.until(EC.visibility_of_element_located((By.NAME, "text")))
        username_input.send_keys('Enter Username')

        next_button = driver.find_element(By.XPATH, '/html/body/div/div/div/div[1]/div/div/div/div/div/div/div[2]/div[2]/div/div/div[2]/div[2]/div/div/div/button[2]')
        next_button.click()

        password_input = wait.until(EC.visibility_of_element_located((By.NAME, "password")))
        password_input.send_keys('Enter PAssword')

        login_button = driver.find_element(By.XPATH, '/html/body/div/div/div/div[1]/div/div/div/div/div/div/div[2]/div[2]/div/div/div[2]/div[2]/div[2]/div/div[1]/div/div/button')
        login_button.click()

        # Scroll and fetch tweets
        num_scrolls = 3
        scroll_increment = 1000
        scroll_delay = 1

        for _ in range(num_scrolls):
            for _ in range(scroll_increment):
                driver.execute_script("window.scrollBy(0, 1);")
                sleep(0.0001)

            sleep(scroll_delay)

            tweets = driver.find_elements(By.CSS_SELECTOR, '[data-testid="tweet"]')
            for tweet in tweets:
                try:
                    tweet_text = tweet.find_element(By.CSS_SELECTOR, 'div[data-testid="tweetText"]').text
                    # tweet_reply = tweet.find_element(By.CSS_SELECTOR, 'div[data-testid="reply"]').text
                    retweets = tweet.find_element(By.CSS_SELECTOR, '[data-testid="retweet"] span').text
                    likes = tweet.find_element(By.CSS_SELECTOR, '[data-testid="like"] span').text
                    # Preprocess likes to remove 'K' suffix
                    likes = likes.replace('K', '000') if 'K' in likes else likes
                    # Preprocess tweet text
                    cleaned_text = preprocess_text(tweet_text)
                    tweet_data.append({
                        "text": cleaned_text,
                        "retweets": retweets,
                        "likes": likes
                    })
                except Exception as e:
                    print(f"Error: {e}")
                    continue

        driver.close()

        st.success("Tweets Scraped Successfully!")

        # Check if tweet data is available
        if tweet_data:
            st.subheader("First Few Tweets")
            for tweet in tweet_data[:5]:
                st.write(tweet["text"])

            st.subheader("Sentiment Analysis Graph")
            tweet_texts = [tweet["text"] for tweet in tweet_data]  # Extract text data from tweet_data list
            preds = get_sentiment(tweet_texts)  # Pass tweet_texts to get_sentiment function

            # Count sentiment labels
            sentiment_counts = {
                "positive": preds["labels"].count("positive"),
                "negative": preds["labels"].count("negative"),
                "neutral": preds["labels"].count("neutral")
            }

            fig = go.Figure(go.Bar(
                x=list(sentiment_counts.keys()),
                y=list(sentiment_counts.values()),
                text=list(sentiment_counts.values()),
                textposition='auto',
                marker=dict(color=['green', 'red', 'blue'])
            ))
            fig.update_layout(title="Sentiment Analysis", xaxis_title="Sentiment Label", yaxis_title="Sentiment Count")
            st.plotly_chart(fig)

            st.subheader("Word Cloud")
            generate_wordcloud([tweet["text"] for tweet in tweet_data])

        else:
            st.warning("No tweets found!")

if __name__ == "__main__":
    app()
