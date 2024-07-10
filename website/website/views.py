# views.py in your Django app (e.g., in the website directory)
from django.shortcuts import render
import csv
import os

def index_view(request):
    return render(request, 'index.html')

def events_view(request):
    return render(request, 'events.html')

def search_view(request):
    return render(request, 'search.html')

def get_story_data(request):
    csv_path = os.path.join(os.path.dirname(__file__), 'data', 'articles.csv')
    summary_value = None
    with open(csv_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            summary_value = row['summary']  # Assuming 'summary' is the column name
    return render(request, 'events.html', {'summary': summary_value})