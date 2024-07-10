# views.py in your Django app (e.g., in the website directory)
from django.shortcuts import render
import csv
import os
import pandas as pd
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def index_view(request):
    return render(request, 'index.html')

def events_view(request):
    # Load the CSV file into a DataFrame
    stories_path = os.path.join(BASE_DIR, 'data/stories.csv')
    articles_path = os.path.join(BASE_DIR, 'data/articles.csv')
    stories_df = pd.read_csv(stories_path)
    articles_df = pd.read_csv(articles_path)
    
    # Get the latest 10 entries from stories_df
    latest_stories_df = stories_df.iloc[-10:].iloc[::-1]  # Reverse the order to get latest first
    summary_values = latest_stories_df['text'].tolist()
    id_values = latest_stories_df['article_id'].tolist()
    
    # Initialize lists for other data
    scores = []
    tickers = []
    urls = []
    news_sources = []
    dates = []
    summaries = []
    contexts = []
    

    # Retrieve corresponding data from articles_df for each of the latest 10 entries
    for id_value in id_values:
        article_row = articles_df.loc[articles_df['article_id'] == id_value]
        stories_row = stories_df.loc[stories_df['article_id'] == id_value]
        scores.append(list(article_row['article_score'])[0])
        tickers.append(list(article_row['source_ticker'])[0])
        urls.append(list(article_row['url'])[0])
        dates.append(list(article_row['pub_date'])[0])
        summaries.append(str(list(stories_row['text'])[0]))
        contexts.append(list(stories_row['context'])[0])
        ns = list(article_row['news_source'])[0]
        ns = ns.replace(' ', '_')
        news_sources.append(ns)

    
    # Render the values in the template

    context = {
        'summaries': json.dumps(summaries),
        'scores': json.dumps(scores),
        'tickers': json.dumps(tickers),
        'urls': urls,
        'dates': dates,
        'news_sources': news_sources,
        'contexts': json.dumps(contexts)
    }
    return render(request, 'events.html', context)

def search_view(request):
    return render(request, 'search.html')