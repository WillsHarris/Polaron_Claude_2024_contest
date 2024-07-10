# views.py in your Django app (e.g., in the website directory)
from django.shortcuts import render
import csv
import os
import pandas as pd
import json
import sys
sys.path.append('../')
from llm import get_embedding
from django.http import HttpResponse, JsonResponse
import chromadb
import numpy as np

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

    # make the horizontal lines and dates html
    dates_html = ''
    for i in range(len(dates)):
        dates_html += f'''
            <div class="horizontal-line hline-{i+1}">
                <div class="datetime"> {dates[i]} </div>
                <div class="h-tick"></div>
            </div>
        '''

    
    # Generate HTML for the reel panes
    reel_panes_html = ''
    for i in range(len(summaries)):
        reel_panes_html += """
            <div class="reel-pane">
                event {i}
                <div>
                    <canvas id="myCanvas-{i}" width="8000" height="3000" style="position:relative; top: 23%; left: 7%;"></canvas>
                </div>
                <script>
                    var newsText = test;
                    var ticker = test;
                    var score = 0;
                    var post_format = 'below';
                    var img_path = test;
                    writeNewsPost(ticker, score, newsText, post_format, img_path, 'myCanvas-{i}');
                </script>
            </div>
        """.replace('{i}', str(i))
        
    
    


    # Render the values in the template
    context = {
        'summaries': json.dumps(summaries),
        'scores': json.dumps(scores),
        'tickers': json.dumps(tickers),
        'urls': urls,
        'dates': dates,
        'news_sources': news_sources,
        'contexts': json.dumps(contexts),
        'dates_html': dates_html,
        'reel_panes_html': reel_panes_html
    }
    return render(request, 'events.html', context)
    

def search_view(request):

    query = request.GET.get('search', '')
    articles_path = os.path.join(BASE_DIR, 'data/articles.csv')
    articles_df = pd.read_csv(articles_path)

    # get search embedding
    if query == "":
        return render(request, 'search.html')
    search_embedding = get_embedding(query)

    chroma_client = chromadb.PersistentClient(path="../website/data/")
    sums = chroma_client.get_collection("summaries")

    query_dict = sums.query(query_embeddings = search_embedding, 
                            n_results=5, 
                            include = ["distances", "metadatas"])

    distances = np.array(query_dict["distances"][0])
    metadata = np.array(query_dict["metadatas"][0])

    article_ids = {"article_id": [metadata[i]["article_id"] for i in range(len(metadata))],
                "distance": distances}
    search_results = pd.DataFrame(article_ids)
    full_search_results = pd.merge(search_results, articles_df, on="article_id")

    context = {
        'title': full_search_results['title'].tolist(),
        'summary': full_search_results['summary'].tolist(),
        'link': full_search_results['url'].tolist()
    }
    results = [{k: v[i] for k, v in context.items()} for i in range(len(context['title']))]

    return JsonResponse({'results': results})