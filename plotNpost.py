import os
import shutil
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.gridspec import GridSpec
from matplotlib.dates import DateFormatter
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime, timedelta,time
import tweepy
from instagrapi import Client
from instagrapi.types import Usertag
import colorgram
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps
import json
import plotly.express as px
import plotly.io as pio
from bing_image_downloader import downloader
from concurrent.futures import ProcessPoolExecutor, as_completed
from selenium import webdriver
from selenium.webdriver import FirefoxOptions
from selenium.webdriver.chrome.options import Options
import time

from langchain.prompts import PromptTemplate

from langchain.output_parsers import ResponseSchema
from langchain_anthropic import ChatAnthropic
from langchain.output_parsers import StructuredOutputParser
from IPython.display import display

import yfinance as yf
import pytz
import pandas as pd

from llm import llm_get_summary_keyphrases, llm_get_summary_metadata, llm_summarize_news_update, create_social_text
from llm import extract_topics_to_expand_on, create_supplementary_note, get_company_wiki_link
from scraper import get_wikipedia_data
import warnings
from util import log_outcome
import traceback

# TEMPORARY
import tracemalloc
tracemalloc.start()

# paths
pwd = os.getcwd().split("/notebooks")[0] # this is to make it work both for broton.py and called from notebook
pwd = os.getcwd().split("/test")[0]      # ^

dir_path = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(dir_path, 'debug.json'), 'r') as file:
    debug_params = json.load(file)

with open( os.path.join(dir_path, 'config.json'), 'r') as file:
    config = json.load(file)

storage_path  = config['storage_path']
sources_path = config['sources_location']


smart_llm = config['smart_llm']
dumb_llm = config['dumb_llm']

X_consumer_key_fin = os.getenv('X_consumer_key_fin')
X_consumer_secret_fin = os.getenv('X_consumer_secret_fin')
X_access_token_fin = os.getenv('X_access_token_fin')
X_access_token_secret_fin = os.getenv('X_access_token_secret_fin')
X_bearer_token_fin = os.getenv('X_bearer_token_fin')

X_consumer_key_world = os.getenv('X_consumer_key_world')
X_consumer_secret_world = os.getenv('X_consumer_secret_world')
X_bearer_token_world = os.getenv('X_bearer_token_world')
X_access_token_world = os.getenv('X_access_token_world')
X_access_token_secret_world = os.getenv('X_access_token_secret_world')

# --------------

Insta_username = os.getenv('Insta_username')
Insta_password = os.getenv('Insta_password')

mpl.rcParams['contour.negative_linestyle'] = 'dashed'

#------------------------------------------------------------------------------------

# plot settings
max_score = 3
max_movement = 2
sigma = 1
hist_max = 50  # max number of scatter points on make_score_vs_movement_plot


logos_root = pwd + "/images/logos"
save_image_root = pwd + "/images"
temp_image_path = pwd + "/images/temp"
gauge_path = pwd + '/images/other/buy_sell_gauge/bearometer_X_X2_Y.png'

# fonts
font_path =  pwd + "/images/fonts/Lexend/Lexend-Regular.ttf" 
font_path_title = pwd + "/images/fonts/Helvetica/Helvetica-Bold.ttf"  
font_path_bold = pwd + "/images/fonts/Lexend/Lexend-SemiBold.ttf"  
font_path_light = pwd + '/images/fonts/Lexend/Lexend-Regular.ttf'
font_path_italics = pwd + "/images/fonts/Helvetica/Helvetica-Oblique.ttf"
font_path_tnr = pwd + '/images/fonts/Lexend/Lexend-Light.ttf' # "/images/fonts/Times_New_Roman/times_new_roman.ttf"

# for Download Logos Tools
articles = pd.read_csv(os.path.join(storage_path, 'articles.csv'))
assets_df = pd.read_csv(os.path.join(storage_path, 'assets.csv'))


with open(sources_path, 'r') as file:
    sources_dict = json.load(file)

#.................................API-posters.................................

def X_post(text='usa > switzerland', posts_df=None, image_paths=[], reply_id=None, account='finance', debug=False):
    warnings.filterwarnings("ignore", category=ResourceWarning, message=".*unclosed <ssl.SSLSocket.*>")
    try:
        if debug and not debug_params["social_posts"]:
            print("Not posting to social media. (debug mode)")
            return None, posts_df # don't actually post
            
        if account == 'finance':
            bearer_token=X_bearer_token_fin
            access_token=X_access_token_fin
            access_token_secret=X_access_token_secret_fin
            consumer_key=X_consumer_key_fin
            consumer_secret=X_consumer_secret_fin

        elif account == 'world':
            bearer_token=X_bearer_token_world
            access_token=X_access_token_world
            access_token_secret=X_access_token_secret_world
            consumer_key=X_consumer_key_world
            consumer_secret=X_consumer_secret_world
        
        api = tweepy.Client(bearer_token=bearer_token,
                            access_token=access_token,
                            access_token_secret=access_token_secret,
                            consumer_key=consumer_key,
                            consumer_secret=consumer_secret)
        
        
        auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
        oldapi = tweepy.API(auth)
        media = [oldapi.media_upload(path) for path in image_paths]

        if reply_id is not None:
            if len(media)>0: 
                response = api.create_tweet(text=text, media_ids=[m.media_id for m in media], in_reply_to_tweet_id=reply_id)
            else:
                response = api.create_tweet(text=text, in_reply_to_tweet_id=reply_id)
        else:
            if len(media)>0: 
                response = api.create_tweet(text=text, media_ids=[m.media_id for m in media])
            else: 
                response = api.create_tweet(text=text)
        tweet_id = response.data.get('id')
        tweet_text = response.data.get('text', text) 

        if posts_df is not None:
            # add a row to the posts_df 
            new_row = pd.DataFrame({"post_id": tweet_id,
                                    "date": datetime.now(pytz.timezone('America/New_York')).strftime('%Y-%m-%d %H:%M'),
                                    "thread_id": None,
                                    "article_id": None, 
                                    "post_text": tweet_text, 
                                    "image_text": None}, 
                                    index=[0])
            posts_df = pd.concat([posts_df, new_row], ignore_index=True)

        return tweet_id, posts_df
    
    except Exception as e:
        print("X post failed", e)

def remove_X_post(post_id, posts_df, account='finance'):
    try:
        if account == 'finance':
            bearer_token = X_bearer_token_fin
            access_token = X_access_token_fin
            access_token_secret = X_access_token_secret_fin
            consumer_key = X_consumer_key_fin
            consumer_secret = X_consumer_secret_fin

        elif account == 'world':
            bearer_token = X_bearer_token_world
            access_token = X_access_token_world
            access_token_secret = X_access_token_secret_world
            consumer_key = X_consumer_key_world
            consumer_secret = X_consumer_secret_world

        api = tweepy.Client(bearer_token=bearer_token,
                            access_token=access_token,
                            access_token_secret=access_token_secret,
                            consumer_key=consumer_key,
                            consumer_secret=consumer_secret)

        response = api.delete_tweet(post_id)

        # remove the row with the post_id from the posts_df
        posts_df = posts_df[posts_df['post_id'] != post_id]

        return posts_df

    except Exception as e:
        print("X post removal failed:", e)
        
def Insta_post(text='usa > france', image_paths = []):
    try:
        cl=Client()
        cl.login(Insta_username, Insta_password)
    except: print("login failed")

    user = cl.user_info_by_username(Insta_username)

    cl.album_upload(
        image_paths,
        caption=text,
        usertags=[Usertag(user=user, x=0.5, y=0.5)],
        extra_data={
        "like_and_view_counts_disabled": False,
        "disable_comments": False})

def llm_timeframe(summary, llm_name):
    # define the response schemas
    response_schemas = [ResponseSchema(
        name="timeframe",
        description="timeframe: 'near-term', 'mid-term' 'long-term'"
    )]


    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    template='''
    identify if the following news is likely to affect the stock price in the near-term (weeks), 
    mid-term (months) or long-term (years). 
    {format_instructions}
    text: {summary}  
    '''

    prompt = PromptTemplate(
        template=template,
        input_variables=["summary"],
        partial_variables={"format_instructions": format_instructions}
    )
  
    chat = ChatAnthropic(model=llm_name, temperature=0)

    _input = prompt.format_prompt(summary=summary)
    output = chat.invoke(_input.to_messages()).content
    llm_cost = llm_cost_cents(str(_input), str(output), llm_name)
    dict_out = output_parser.parse(output)
    timeframe = dict_out['timeframe']

    return timeframe, llm_cost

#.................................Tools.................................

def parse_article_from_dict(articles=None, article_idx=-1, filter_bool=False):
    llm_cost = 0
    if articles is None: 
        articles = pd.read_csv(os.path.join(storage_path, 'articles.csv'))
    if article_idx<0: # only articles that pass the filters
        if filter_bool:
            articles = articles[articles['qfactor_bool'] == True]
            articles = articles[articles['bfactor_str'] == 'breaking']
    ticker = articles.iloc[article_idx]['source_ticker']
    article_id = articles.iloc[article_idx]['article_id']
    summary = articles.iloc[article_idx]['summary']
    reason = articles.iloc[article_idx]['reason']
    date = articles.iloc[article_idx]['pub_date']
    article_score = articles.iloc[article_idx]['article_score']
    news_source = articles.iloc[article_idx]['news_source']
    url = articles.iloc[article_idx]['url']
    news_type = articles.iloc[article_idx]['news_type']
    timeframe, llm_cost = llm_timeframe(summary, llm_name=dumb_llm)
    bfactor_str = timeframe

    try: 
        summary_keyphrases, llm_cost_new = llm_get_summary_keyphrases(summary, llm_name=smart_llm)
        llm_cost += llm_cost_new
    except: 
        summary_keyphrases = []

    return ticker, article_id, summary, reason, date, article_score, summary_keyphrases, news_source, url, llm_cost, bfactor_str, news_type

def get_article_post_data_dict(articles, article_idx, filter_bool=False):
 
    dict_out = {}
    llm_cost = 0
    if articles is None:  # read articles if not provided
        articles = pd.read_csv(os.path.join(storage_path, 'articles.csv'))

    # optional filter
    if article_idx<0:
        if filter_bool:
            articles = articles[articles['qfactor_bool'] == True]
            articles = articles[articles['bfactor_str'] == 'breaking']

    # data from the articles.csv
    dict_out['ticker'] = articles.iloc[article_idx]['source_ticker']
    dict_out['article_id'] = articles.iloc[article_idx]['article_id']
    dict_out['summary'] = articles.iloc[article_idx]['summary']
    dict_out['reason'] = articles.iloc[article_idx]['reason']
    dict_out['pub_date'] = articles.iloc[article_idx]['pub_date']
    dict_out['article_score'] = articles.iloc[article_idx]['article_score']
    dict_out['news_source'] = articles.iloc[article_idx]['news_source']
    dict_out['url'] = articles.iloc[article_idx]['url']
    dict_out['news_type'] = articles.iloc[article_idx]['news_type']
    other_ticker = articles.iloc[article_idx]['other_ticker']
    try: other_ticker = eval(other_ticker)
    except: pass
    dict_out['other_ticker'] = other_ticker
    

    # MOVE TO PLOT FUNC
    # # get other ticker (if exists)
    # tickers_tracked_lst = assets_df['ticker'].to_list()
    # other_ticker_list = []
    # if other_ticker is not None:
    #     for t in other_ticker.keys():
    #         if t in tickers_tracked_lst:
    #             other_ticker_list.append(t)
    # other_ticker_bool = len(other_ticker_list) == 2
        

    # additinal metadata not in the articles.csv
    timeframe, llm_cost_new = llm_timeframe(dict_out['summary'], llm_name=dumb_llm)
    dict_out['timeframe'] = timeframe
    llm_cost += llm_cost_new

    co_name = assets_df[assets_df['ticker'] == articles.iloc[article_idx]['source_ticker']]['name'].to_list()[0]
    social_post_text, llm_cost_text = create_social_text(co_name, articles.iloc[article_idx]['summary'], smart_llm)
    llm_cost += llm_cost_text

    try:
        text_keyphrases, llm_cost_new = llm_get_summary_keyphrases(social_post_text, llm_name=smart_llm)
        llm_cost += llm_cost_new
    except: text_keyphrases = []

    dict_out['image_text'] = social_post_text
    dict_out['text_keyphrases'] = text_keyphrases

    return dict_out, llm_cost

def find_phrases_in_text_bool(text, phrase_list):
    bool_out = [False for i in range(len(text.split()))]

    for phrase in phrase_list:
        split_text = text.lower().split(phrase.lower())
        if len(split_text) > 1:
            phrase_start_idx = len(split_text[0].split())
            for i in range(len(phrase.split())):
                bool_out[i + phrase_start_idx] = True

    return bool_out

def color_distance(c1, c2):
    return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2 + (c1[2] - c2[2])**2)
   
def get_main_colors(image_path, num_colors=5):
    try:
        image = Image.open(image_path)
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        colors = colorgram.extract(image, num_colors)
        main_colors = [(color.rgb[0] / 255, color.rgb[1] / 255, color.rgb[2] / 255) for color in colors]
        color_threshold = 0.1
        main_colors_new = [c for c in main_colors if color_distance(c, (0, 0, 0)) > color_threshold and color_distance(c, (1, 1, 1)) > color_threshold]
        if main_colors_new == []: pass
        else: main_colors = main_colors_new
        if main_colors is None or  main_colors ==[]: main_colors=[(0,0,0)]

        return main_colors
    except Exception as e:
        print("An error occurred:", e)
        return None

def get_ticker_colors(tickers_list, logos_root):
    colors_file = "colors_data.json"

    # Try to load colors data from the file
    try:
        with open(colors_file, 'r') as file:
            saved_data = json.load(file)

        # Check if loaded data contains all tickers in tickers_list
        if set(tickers_list).issubset(saved_data.keys()):
            return [saved_data[ticker] if ticker != "USD" else "white" for ticker in tickers_list]
        else: pass 

    except FileNotFoundError: pass

    # Recreate the colors data
    colors_data = {ticker: get_main_colors(logos_root + f"/{ticker}.png")[0] if ticker != "USD" else "white" for ticker in tickers_list}

    # Save the colors data to the file
    with open(colors_file, 'w') as file:
        json.dump(colors_data, file)

    return [colors_data[ticker] for ticker in tickers_list]

def last_trade_info(tickers):

    articles = pd.read_csv(os.path.join(storage_path, 'articles.csv'))
    result = {}

    if not isinstance(tickers, list):
        tickers = [tickers]

    for ticker in tickers:
        # Filter articles for the given ticker
        ticker_articles = articles[articles['source_ticker'] == ticker]

        # Check if there are any trades for the given ticker
        if ticker_articles.empty:
            result[ticker] = {'pub_date': None, 'id': None}
        else:
            # Get the last trade information
            last_trade = ticker_articles.iloc[-1]
            trade_date = last_trade['pub_date']
            trade_id = last_trade['article_id']

            result[ticker] = {'pub_date': trade_date, 'id': trade_id}

    return result

#.................................bearometer builder.................................

bearometer_html_path = '/home/wills/broton/images/other/buy_sell_gauge/bearometer.html'
bearometer_html_path_X_Y = '/home/wills/broton/images/other/buy_sell_gauge/bearometer_X_X2_Y.html'


def process_ij(ij, ticker_lst=None, save_path='auto'):
    try: 
        i, j = ij
        i2=None
    except: 
        i, i2, j = ij
        assert ticker_lst is not None

    output = []
    output.append(f"Processing: {i}, {i2}, {j}")
    # Load html and fill in X and Y 
    with open(bearometer_html_path, 'r') as root_file:
        root_html_content = str(root_file.read())
        new_html_content = root_html_content.replace('var score = 0;', f'var score = {i};')
        new_html_content = new_html_content.replace('var j = 0;', f'var j = {j};')
        if i2 is not None:
            new_html_content = new_html_content.replace('var score2 = null;', f'var score2 = {i2};')
            new_html_content = new_html_content.replace('TICK1', ticker_lst[0])
            new_html_content = new_html_content.replace('TICK2', ticker_lst[1])

    # make file paths
    if save_path == 'auto':
        ij_html_path = bearometer_html_path_X_Y
        if i2 is not None: ij_html_path = ij_html_path.replace('X2', str(i2))
        else: ij_html_path = ij_html_path.replace('_X2', '')
        ij_html_path = ij_html_path.replace('X', str(i)).replace('Y', str(j))
        ij_png_path = ij_html_path.replace('.html', '.png')
    else:
        ij_html_path = save_path.replace('.png', '.html')
        ij_png_path = save_path

    # Save html
    with open(ij_html_path, 'w') as html_file:
        html_file.write(new_html_content)

    # Take screenshot
    opts = FirefoxOptions()
    opts.add_argument("--headless")
    driver = webdriver.Firefox(options=opts)
    driver.set_window_size(7*500, 3*360) 
    driver.get('https://google.com/') 
    driver.get('file://' + os.path.abspath(ij_html_path)) 
    time.sleep(1)
    driver.save_screenshot(ij_png_path) 
    driver.quit()

    # Crop image
    file_to_crop = ij_png_path
    image = Image.open(file_to_crop)
    width, height = image.size
    crop_pixels_height = int(0.01 * height)
    crop_pixels_width = int(0.004 * width)

    cropped_image = image.crop((crop_pixels_width, crop_pixels_height, width, height))
    cropped_image.save(file_to_crop)
    image.close()

    # Remove html
    os.remove(ij_html_path)

    return output

def update_bearometers(mode=None):
    with ProcessPoolExecutor() as executor:
        # make list of ijs
        scores = [-3, -2, -1, 0, 1, 2, 3]
        times = [0, 1, 2]
        ij_lst_1s = [(i, j) for i in scores for j in times] # one score
        #ij_lst_2s = [(i, i2, j) for i in scores for i2 in scores for j in times if i2>i] # two scores
        ij_lst = ij_lst_1s

        # process ijs
        futures = {executor.submit(process_ij, ij): ij for ij in ij_lst}
        for future in as_completed(futures):
            ij = futures[future]
            try:
                output = future.result()
            except Exception as exc:
                print(f'{ij} generated an exception: {exc}')
            else:
                print('\n'.join(output))

#.................................Pillow-Tools.................................

def add_white_borders(image_path, dimx, dimy):
    # Open the image
    original_image = Image.open(image_path)
    
    # Calculate the dimensions for the new image with borders
    current_width, current_height = original_image.size
    aspect_ratio = current_width / current_height
    
    if aspect_ratio > 1:
        new_height = dimy
        new_width = int(dimy * aspect_ratio)
    else:
        new_width = dimx
        new_height = int(dimx / aspect_ratio)

    # Calculate the position to center the image
    x_offset = (dimx - new_width) // 2
    y_offset = (dimy - new_height) // 2

    # Create a new blank image with white borders
    new_image = Image.new("RGB", (dimx, dimy), "white")

    # Paste the original image onto the new image with centering
    new_image.paste(original_image.resize((new_width, new_height), Image.ANTIALIAS), (x_offset, y_offset))
    new_image.save(image_path)
    return new_image

def wrap_text(draw, text, x_position, y_position, bold_bool_lst, font_bold, font_medium, font_size, size, max_y, max_x, text_height, x_justify='left', y_justify='top', resize_font=True):
    x_pos = x_position
    y_pos = y_position

    if bold_bool_lst is None:
        bold_bool_lst = [False for _ in text.split()]
    
    x_pos_lst = []
    y_pos_lst = []
    words = []
    fonts = []

    i = -1
    while i < len(text.split())-1:
        i += 1
        word = text.split()[i]
        if bold_bool_lst[i]:
            font = ImageFont.truetype(font_bold, font_size)
        else:
            font = ImageFont.truetype(font_medium, font_size)

        word_bbox = draw.textbbox((0, 0), word, font=font)
        word_width = word_bbox[2] - word_bbox[0]
        word_hight = word_bbox[3] - word_bbox[1]

        if x_pos + word_width > max_x:
            x_pos = x_position
            y_pos += text_height
        
        x_pos_next = x_pos + word_width + (draw.textbbox((0, 0), " ", font=font)[2])

        x_pos_lst.append(x_pos)
        y_pos_lst.append(y_pos)
        fonts.append(font)
        words.append(word)
        x_pos = x_pos_next

        if resize_font:
            while y_pos + word_hight > max_y: 
                x_pos = x_position
                y_pos = y_position

                if bold_bool_lst is None:
                    bold_bool_lst = [False for _ in text.split()]
                x_pos_lst = []
                y_pos_lst = []
                words = []
                fonts = []
                text_height = text_height * (font_size-1)/font_size
                font_size -= 1
                
                i = -1
                if font_size <= 0: break
                if bold_bool_lst[i]: font = ImageFont.truetype(font_bold, font_size)
                else: font = ImageFont.truetype(font_medium, font_size)
                word_bbox = draw.textbbox((0, 0), word, font=font)
                word_width = word_bbox[2] - word_bbox[0]
                


    word_height = draw.textbbox((0, 0), text.split()[0], font=font)[3]
    text_width = max_x - x_position 
    text_height = max(y_pos_lst) - y_position + word_height
    paragraph_height = y_pos_lst[-1] - y_pos_lst[0] + word_height


    # Justify text
    if x_justify == 'left':
        xsift = 0  
    elif x_justify == 'center':
        xsift = text_width // 2
    elif x_justify == 'right':
        xsift = text_width
    if y_justify == 'top':
        yshift = 0  
    elif y_justify == 'center':
        yshift = -1*(max((max_y-y_position) - paragraph_height, 0))//2
    elif y_justify == 'bottom':
        yshift = text_height


    x_pos_lst = [x - xsift for x in x_pos_lst]
    y_pos_lst = [y - yshift for y in y_pos_lst]

    for x_pos, y_pos, word, font in zip(x_pos_lst, y_pos_lst, words, fonts):
        draw.text((x_pos, y_pos), word, fill="black", font=font)

def line_xy(draw, val, mode='x', extent=None, spacer1=0, spacer2=0, color='black', line_width=8, end_caps=False):
    sval1 = val + spacer1
    sval2 = val - spacer2
    if extent is None:
        sizey = 1600
        sizex = 1600
        if  mode=='x': extent = [0, sizey]
        if  mode=='y': extent = [0, sizex]

    if  mode=='x':
        draw.line([(sval1, extent[0]), (sval1, extent[1])], fill="gray", width=line_width)
        draw.line([(sval2, extent[0]), (sval2, extent[1])], fill="gray", width=line_width)
        draw.line([(val, extent[0]), (val, extent[1])], fill=color, width=line_width)
    if  mode=='y':
        draw.line([(extent[0], sval1), (extent[1], sval1)], fill="gray", width=1)
        draw.line([(extent[0], sval2), (extent[1], sval2)], fill="gray", width=1)  
        draw.line([(extent[0], val), (extent[1], val)], fill=color, width=line_width)

def add_text(draw, xy, text, font_path, font_size, color='black', x_justify='center', y_justify='center'):
    font = ImageFont.truetype(font_path, font_size)
    
    def get_text_size(text, font):
        bbox = font.getbbox(text)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return width, height
    
    text_width, text_height = get_text_size(text, font)
    x, y = xy
    
    if x_justify == 'left': pass  
    elif x_justify == 'center': x -= text_width // 2
    elif x_justify == 'right': x -= text_width
    
    if y_justify == 'top': pass  
    elif y_justify == 'center': y -= text_height // 2
    elif y_justify == 'bottom': y -= text_height
    
    draw.text((x, y), text, font=font, fill=color)

def recolor_image(image_path, new_color):
    """
    Recolors black pixels to the specified RGBA color while preserving transparency.

    :param image_path: The path to the image to be recolored.
    :param new_color: The RGBA color to replace black with, as a tuple (R, G, B, A).
    """
    # Load the image
    image = Image.open(image_path)
    
    # Convert the image to RGBA if it's not already in that mode
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    # Convert image to numpy array
    data = np.array(image)
    
    # Identify black pixels (or nearly black) and non-transparent pixels
    # Black is assumed to be (0, 0, 0, A) where A>0 to avoid changing transparent pixels
    black_pixels_mask = (data[:, :, 0] == 0) & (data[:, :, 1] == 0) & (data[:, :, 2] == 0) & (data[:, :, 3] > 0)
    
    # Change black pixels to the new color
    data[..., :4][black_pixels_mask] = new_color  # Change only the color, not the alpha channel
    
    # Convert numpy array back to PIL Image
    new_image = Image.fromarray(data, 'RGBA')
    return new_image

def add_image(image, path, xy, image_size_max, x_justify='center', y_justify='center', stack_mode='auto', recolor = (0,0,0,255)):
    if type(path) is not list:
        im = Image.open(path).convert("RGBA")
        im.thumbnail(image_size_max)
        x, y = xy
        width, height = im.size
        if x_justify == 'left': pass  
        elif x_justify == 'center': x -= width // 2
        elif x_justify == 'right': x -= width
        
        if y_justify == 'top': pass  
        elif y_justify == 'center': y -= height // 2
        elif y_justify == 'bottom': y -= height
        
        image.paste(im, (x, y), mask=im)
    else: 
        height_total = 0
        width_total = 0
        image_max_x, image_max_y = image_size_max
        disp_lst_x = []
        disp_lst_y = []
        widths = []
        heights = []
        
        pad = 0.15
        for p in path:
            im = Image.open(p).convert("RGBA")
            im.thumbnail([image_max_x, image_max_y])
            width_total += im.size[0]
            height_total += im.size[1]
            disp_lst_x.append(im.size[0]*(1+pad))
            disp_lst_y.append(im.size[1]*(1+pad))
            widths.append(im.size[0])
            heights.append(im.size[1])
        
        # add padding between images only
        width_total *= (1+(len(path)-1)*pad) 
        height_total *= (1+(len(path)-1)*pad)
        
        areas = []
        imgs = []
        xs = []
        ys = []
        HVs = []
        # add images
        if stack_mode == 'auto':
            hvs = ["H", "V"]
        else:
            hvs = [stack_mode]
        for hv in hvs: # horizontal and vertical
            for i, p in enumerate(path):
                try:
                    im = recolor_image(p, recolor[i])
                except:
                    im = Image.open(p).convert("RGBA")
                
                # reset each image box to fit image_size_max
                if hv == 'H':
                    image_resize_max = np.array([image_max_x, image_max_y]) * min(image_max_x / width_total, image_max_y / heights[i])

                elif hv == 'V':
                    image_resize_max = np.array([image_max_x, image_max_y]) * min(image_max_y / height_total, image_max_x / widths[i])
            
                im.thumbnail(image_resize_max)
                width, height = im.size
                
                x, y = xy
                if hv == 'H':
                    x +=  int(((widths[i]//2 + sum(widths[:i]) + pad/(1+pad)*i*width_total) - width_total//2) * (image_max_x) / width_total)
                elif hv == 'V':
                    y +=  int(((heights[i]//2 + sum(heights[:i]) + pad/(1+pad)*i*height_total) - height_total//2) * image_max_y / height_total)


                if x_justify == 'left': pass  
                elif x_justify == 'center': x -= width // 2
                elif x_justify == 'right': x -= width
                
                if y_justify == 'top': pass  
                elif y_justify == 'center': y -= height // 2
                elif y_justify == 'bottom': y -= height

                areas.append(width*height)
                imgs.append(im)
                xs.append(x)
                ys.append(y)
                HVs.append(hv)

        # find the best orientation 
        total_areas = []
        for i in range(len(areas)//2):
            total_areas.append(areas[i] + areas[i+1])
        i_max = total_areas.index(max(total_areas))
        HV_best = hvs[i_max]

        # paste images
        for im, x, y, hv, in zip(imgs, xs, ys, HVs):
            if hv ==HV_best:
                image.paste(im, (x, y), mask=im)


    return width, height

#.............................Download-Logos-Tools...................................

def download_yahoo_image(co_name, temp_image_path, image_limit = 9, image_type = 'logo'):
   
    # Download images
    if image_type == 'logo':
        search_term = f'{co_name} logo'
    elif image_type == None:
        search_term = f'{co_name}'
    force_replace = False
    timeout = 20

    print('---------')
    downloader.download(search_term, 
                        limit=image_limit, 
                        output_dir=temp_image_path, 
                        force_replace=force_replace, 
                        timeout=timeout, 
                        adult_filter_off=True,
                        verbose=False)
    
    images_path = os.path.join(temp_image_path, search_term)
    print('TEST')
    print(images_path)
    images_path_new = os.path.join(temp_image_path, f'yahoo_downloads')

    # Check if the original images path exists
    if not os.path.exists(images_path):
        print(f"Directory does not exist: {images_path}")
        return None

    # Move images_path to images_path_new, replacing what was there before
    if os.path.exists(images_path_new):
        shutil.rmtree(images_path_new)
        print(f"Existing directory removed: {images_path_new}")

    try:
        shutil.move(images_path, images_path_new)
        print(f"Directory moved from {images_path} to {images_path_new}")
    except Exception as e:
        print(f"Error moving directory: {e}")
        return None
    print('TEST, TEST, TEST') 
    return images_path_new

def make_image_grid(image_root, names=None, show=True):
    downloaded_images = [f for f in os.listdir(image_root) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    try:
        downloaded_images = sorted(downloaded_images, key=lambda x: int(x.split('_')[1].split('.')[0]))
    except: pass

    if names is not None:
        # filter the images by the names
        downloaded_images = [f for f in downloaded_images if os.path.splitext(f)[0] in names]
        # delete duplicates
        downloaded_images = list(set(downloaded_images))
        # order the images by the names
        downloaded_images = sorted(downloaded_images, key=lambda x: names.index(os.path.splitext(x)[0]))
        
    # Show the images in a grid with DPI values
    if downloaded_images:
        num_cols = 3  # Adjust the number of rows and columns as needed
        num_rows = len(downloaded_images) // num_cols + min(len(downloaded_images)%num_cols, 1)
        _, axes = plt.subplots(num_rows, num_cols, figsize=(10, 8))

        im_paths = []
        for j, ax in enumerate(axes.flat):
            for im_path in downloaded_images:
                try:
                    if im_path not in im_paths:
                        im_paths.append(im_path)
                        
                        image_path = os.path.join(image_root, im_path)
                        img = Image.open(image_path)
                        img = Image.open(image_path)
                        if image_path.endswith('.jpg'):
                            os.remove(image_path)
                            image_path = image_path.replace('.jpg', '.png')
                            img.save(image_path, format='PNG')
                            img = Image.open(image_path)
                        assert image_path.endswith('.png')
                        # image size
                        ax.set_aspect('equal', adjustable='box')         
                        width, height = img.size
                        size = (int(width), int(height))
                        assert width > 400
                        assert height > 400
                        ax.imshow(img, aspect='equal') 
                        ax.axis('off')   
                        ax.text(0.5, 1.05, f"Image {1+downloaded_images.index(im_path)}. SIZE: {size} \n", transform=ax.transAxes, ha='center', va='center', fontsize=8)
                        break
                    
                except: pass

    else: print("No images downloaded.")
    if show: plt.show()
    else: 
        # save to image_root + 'image_grid.png'
        save_path = os.path.join(image_root, 'image_grid.png')
        plt.savefig(save_path)
        plt.close()
        return save_path

def copy_logo_to_logos_dir(image_root, image_numb, name, dest_dir=logos_root):

    file_name = f"Image_{image_numb}.png"
    new_name = name + '.png'
    new_name = new_name.replace(' ', '_')

    # Construct the source and destination file paths
    src_file_path = os.path.join(image_root, file_name)
    dest_file_path = os.path.join(dest_dir, new_name)

    # Open the image using Pillow
    image = Image.open(src_file_path).convert('RGBA')

    # Remove white background
    datas = image.getdata()
    newData = []
    for item in datas:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
    image.putdata(newData)
  
    # Crop the image to remove empty space
    cropped_image = image.crop(image.getbbox())

    # Save the cropped image, overwriting the original file
    cropped_image.save(dest_file_path)
    print('file saved', dest_file_path)
    return dest_file_path

def make_black_white(logo_path, mode=1, write=False, threshold=228):
    original_image = Image.open(logo_path).convert('RGBA')
    bw_image = original_image.convert('LA')
    darker_image = None
    if mode == 1:
        darker_image = bw_image.point(lambda p: p if p > threshold else 0)
    if write:
        bw_path = logo_path.replace('.png', '_bw.png')
        darker_image.save(bw_path)
    else: display(darker_image)

def check_if_logo_exists(BW=False, write=False, type='tickers'):
    
    if type == 'tickers':
        names = assets_df['ticker'].to_list()
    elif type == 'sources':
        names = [name.replace(' ', '_') for category in sources_dict.values() for company_info in category for name in company_info.keys()]

    for t in names:
        
        suffix = '.png'
        suffix_bd = '_bw.png'

        t_path = os.path.join(logos_root, t+suffix)
        t_path_bw = os.path.join(logos_root, t+suffix_bd)
        path = t_path
        if BW: path = t_path_bw
        if os.path.exists(path):
            print(path)
            pass
        else:
            
            if write:
                assert BW
                make_black_white(t_path, write = write)
                print('wrote bw file for ', t_path)
            else:
                print(f"The file '{t_path}' does not exist.")
                if BW: make_black_white(t_path)

def add_logo_from_wiki(company_name, ticker, llm_name, save_path):
    llm_cost = 0
    print(f'Adding logo for {company_name} to {save_path}.')

    try:
        wiki_url = None
        ticker_ = ticker
        if ticker_ == None: ticker_ = company_name

        wiki_url, llm_cost = get_company_wiki_link(company_name, ticker_, llm_name)
        temp_local_img_dir = 'temp_imgs'
        wiki_body, wiki_image_dict_lst = get_wikipedia_data(wiki_url, save_root=temp_local_img_dir)
        
        # for no wiki body found, return
        if wiki_body is None:
            print(f'ERROR: No Wikipedia for {company_name} found (url {wiki_url}).')
            return llm_cost
        
        # find the logo image name 
        logo_image_name = None
        for wiki_image_dict in wiki_image_dict_lst:
            if 'logo' in wiki_image_dict['image_name'].lower():
                logo_image_name = wiki_image_dict['image_name']
                break
            if 'logo' in wiki_image_dict['description'].lower():
                logo_image_name = wiki_image_dict['image_name']
                break

        # for no logo image name found, return
        if logo_image_name is None:
            print(f'ERROR: No Wikipedia logo for {company_name} found (url {wiki_url})')
            return llm_cost
        
        # check if the 'company' is infact a company, as indicated by the wiki body
        if 'company' not in wiki_body.lower():
            print(f'ERROR: {company_name} is not a company. Skipping (url {wiki_url}).')
            return llm_cost
        co_strs = ['inc.', 'co.',  'ltd.', 'corporation', 'limited']
        if not any(s in wiki_body.lower() for s in co_strs):
            print(f'ERROR: {company_name} is not a company. Skipping (url {wiki_url}).')
            return llm_cost
        

        # copy the logo image to the save path with ticker_.png
        if logo_image_name.endswith('.png'):
            logo_image_path_temp = os.path.join(temp_local_img_dir, logo_image_name)
            logo_image_path_new = os.path.join(save_path, f"{ticker_}.png")

            # copy the image to the save path
            shutil.copyfile(logo_image_path_temp, logo_image_path_new)

            # make black/white copy in save path
            make_black_white(logo_image_path_new, mode=1, write=True, threshold=228)

        else:
            print(f'Logo for {company_name} is not a png image. Skipping (url {wiki_url}).')
            return llm_cost

        # remove temp images directory
        shutil.rmtree(temp_local_img_dir)

        return llm_cost
    
    except Exception as e:
        print(f'ERROR: {e}')
        return llm_cost
    

#.................................Posting-Tools.................................

def make_X_caption(co_name, ticker, score, url, publisher, news_type=None, add_url_to_text=False):
    # Load sites_dict:
    dir_local = os.path.dirname(os.path.abspath(__file__))
    sites_dict_path = os.path.join(dir_local, 'sources.json')
    with open(sites_dict_path) as json_file:
        sites_dict = json.load(json_file)

    handle = ''
    for k in sites_dict.keys():
        if publisher == str(k):
            handle += ' '
            handle += sites_dict[k]['X_handle']


    descriptor = ['Very negative', 'Negative', 'Moderately negative', 'Neutral', 'Moderately positive', 'Positive', 'Very Positive'][int(score)+3]
    X_caption = f'''{descriptor} news about {co_name} ${ticker}, per{handle}''' 
    if add_url_to_text: X_caption += f"\nSource: {url}"
    
    if news_type is not None: # add news type to caption
        '''
        ['product', 'services', 'regulatory', 'legal', 'acquisition (of a company)', \
        'merger', 'partnership', 'labor', 'consumer', 'stock buyback', \
        'dividend', 'capital expenditure' , 'macroeconomic',  'management', \
        'earnings', 'financial', 'market analysis', 'opinion', 'other']

        '''
        if news_type in ['product', 'regulatory', 'legal']:
            X_caption_split = X_caption.split('news')
            X_caption = X_caption_split[0] + news_type + ' news' + X_caption_split[1]
    
    return X_caption

def week_in_review_text(articles_df):
    a_df = articles_df.copy()
    # Convert publication date to datetime
    a_df['pub_date'] = pd.to_datetime(a_df['pub_date'])
    
    # Filtering the DataFrame
    one_week_ago = datetime.now() - timedelta(weeks=1)
    filtered_df = a_df[(a_df['article_score'].notna()) & (a_df['pub_date'] > one_week_ago)]
    filtered_df = filtered_df[filtered_df['article_score'] != 0]

    # Calculate average scores
    average_scores = filtered_df.groupby('source_ticker')['article_score'].mean().reset_index()
    
    # Calculate the number of scores for each 'source_ticker'
    score_counts = filtered_df.groupby('source_ticker')['article_score'].count().reset_index()
    
    # Merge the two DataFrames on 'source_ticker'
    average_scores = average_scores.merge(score_counts, on='source_ticker', suffixes=('_mean', '_count'))
    
    # Rename the columns for clarity
    average_scores.columns = ['source_ticker', 'average_score', 'number_of_scores']
    average_scores = average_scores[average_scores['number_of_scores'] > 2].sort_values('average_score', ascending=False)

    # Prepare the post text
    post_text = 'WEEK IN REVIEW: Average News Sentiment (Scale: -1 to 1)\n\n'
    
    top_bottom_tickers = []

    # Add top 3 tickers and their scores
    top_tickers = []
    for index, (ticker, score, _) in enumerate(average_scores.head(3).values):
        row = f'${ticker}: {score/3:.1f}'
        if score > 0:
            row = row.replace(': ', ': +')

        row = row.replace(f'{score/3:.1f}', f'{score/3:.1f}'+(" "*(6-len(ticker))))
        top_tickers.append(row)
        top_bottom_tickers.append((ticker, index))
    
    # Add bottom 3 tickers and their scores
    bottom_tickers = []
    for index, (ticker, score, _) in enumerate(average_scores.tail(3).values, start=3):
        row = f'${ticker}: {score/3:.1f}'
        if score > 0:
            row = row.replace(': ', ': +')
        bottom_tickers.append(row)
        top_bottom_tickers.append((ticker, index))

    # Combine top and bottom tickers side by side
    post_text += '🔺Top Tickers:       🔻Bottom Tickers:\n'
    for i, (top, bottom) in enumerate(zip(top_tickers, bottom_tickers)):
        tic_len = len(average_scores.iloc[i]['source_ticker'])
        if tic_len==2:
            post_text += f'{top:<26}    {bottom}\n'
        elif tic_len==3:
            post_text += f'{top:<23}    {bottom}\n'
        elif tic_len==4:
            post_text += f'{top:<20}    {bottom}\n'
        elif tic_len==5:
            post_text += f'{top:<17}    {bottom}\n'
        else:
            post_text += f'{top:<24}    {bottom}\n'

    # Filter article_scores_dict to only include top and bottom tickers
    filtered_articles = filtered_df[filtered_df['source_ticker'].isin([ticker for ticker, _ in top_bottom_tickers])]
    article_scores_dict = {
        row['article_id']: {
            'article_score': row['article_score'],
            'source_ticker': row['source_ticker'],
            'ticker_position': next(index for ticker, index in top_bottom_tickers if ticker == row['source_ticker'])
        }
        for _, row in filtered_articles.iterrows()
    }

    return post_text, article_scores_dict

def short_summ_of_composite_news_for_ticker(news_list, ticker, llm_name, sentement=None):
    response_schemas = [
        ResponseSchema(
            name="summary",
            description="A very short sentence summarizing only the most important points in the news_list for the given ticker.",
            type = "string" 

        )
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    
    # build the prompt
    template="""
    Summarize a few most important points in the news_list for a given ticker in one extreamly short, direct statement (use as few words as possible).
    """ 
    if sentement is not None:
        template += f" Focus on points from the news that is {sentement}."
    
    template+="""
    {format_instructions}
    news_list = {news_list}
    ticker = {ticker}

    Do not include comments in the output JSON string.
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=['news_list', 'ticker'],
        partial_variables={"format_instructions": format_instructions} 
    )
    _input = prompt.format_prompt(news_list=news_list, ticker=ticker) 
    chat = ChatAnthropic(model=llm_name, temperature=0)
    output = chat.invoke(_input.to_messages()).content
    
    dict_out = output_parser.parse(output)
    
    # extract data from dict_out
    summary = dict_out['summary']

    return summary

def reduce_text(text, llm_name, mode='strict'):
    response_schemas = [
        ResponseSchema(
            name="summary",
            description="A much reduced text that maintains the original format.",
            type = "string" 

        )
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    
    # build the prompt
    if mode == 'normal':
        template="""
        Reduce the text by summarizing with fewer words. Keep the same formatting (number of lines, titles), but remove some information and reword for word-count efficiency.
        """ 
    else:
        template="""
        Reduce the text to as few words as possible. Keep the same formatting (number of lines, titles), but remove some information and reword for word-count efficiency.
        Note -- if the formatt is "$TICKER: summary text about ticker", no need to restate the company name or ticker in the summary text.
        """ 
    if mode == 'very strict':
        template = " Reduce the text in length"
    
    template+="""
    {format_instructions}
    text = {text}

    Do not include comments in the output JSON string.
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=['text'],
        partial_variables={"format_instructions": format_instructions} 
    )
    _input = prompt.format_prompt(text=text) 
    chat = ChatAnthropic(model=llm_name, temperature=0)
    output = chat.invoke(_input.to_messages()).content
    
    dict_out = output_parser.parse(output)
    
    # extract data from dict_out
    summary = dict_out['summary']

    return summary

def reduce_text_to_target(text, llm_name, len_target = 38, max_recursions=3, mode='strict'):
    modes = ['normal', 'strict', 'very strict']
    assert mode in modes

    text_len = len(text.split())
    if text_len<=len_target: return text
    r = 0
    strict_bool = False
    while len(text.split())>len_target:
        r+=1
        if r>max_recursions: break
        if r == max_recursions: mode = modes[min(len(modes), modes.index(mode))] # increse mode strength
        text = reduce_text(text, llm_name, mode)
        text_len = len(text.split())
        if text_len<=len_target: break
    
    if text_len>len_target:
        print(f'Could not reduce text length ({text_len}) to target ({len_target}):')
        print(text)
        text = None

    return text

def week_in_review_thread_texts(articles_df, article_scores_dict, llm_name):
    post_text_top = "Top Tickers:"
    for idx in [0,1,2]:
        ticker = [v['source_ticker'] for k,v in article_scores_dict.items() if v['ticker_position']==idx][0]
        id_lds = [k for k,v in article_scores_dict.items() if v['ticker_position']==idx]
        filtered_articles = articles_df[articles_df['article_id'].isin(id_lds)]
        news_list = str(filtered_articles['summary'].to_list())
        tic_summ = short_summ_of_composite_news_for_ticker(str(news_list), ticker, llm_name, sentement='positive')
        post_text_top += f"\n\n${ticker}: {tic_summ}"

    post_text_bottom = "Bottom Tickers:"
    for idx in [3, 4, 5]:
        ticker = [v['source_ticker'] for k,v in article_scores_dict.items() if v['ticker_position']==idx][0]
        id_lds = [k for k,v in article_scores_dict.items() if v['ticker_position']==idx]
        filtered_articles = articles_df[articles_df['article_id'].isin(id_lds)]
        news_list = str(filtered_articles['summary'].to_list())
        tic_summ = short_summ_of_composite_news_for_ticker(str(news_list), ticker, llm_name, sentement='negative')
        post_text_bottom += f"\n\n${ticker}: {tic_summ}"

    post_text_top = reduce_text_to_target(post_text_top, llm_name, len_target = 38, max_recursions=3)
    post_text_bottom = reduce_text_to_target(post_text_bottom, llm_name, len_target = 38, max_recursions=3)
    
    if post_text_bottom or post_text_top is None:
        return None, None

    return post_text_top, post_text_bottom

#.................................Plotting-functions.................................

def make_score_vs_movement_plot(scatter_news_score, scatter_movement, ticker, scatter_movement_hist=[], scatter_news_score_hist=[], shape="square", show_plot=False):

    # Fixing random state for reproducibility
    fpath = Path(font_path)

    # the x, y data
    x = scatter_movement_hist
    y = scatter_news_score_hist
    #x = [max(min(xi, max_movement*0.97), -max_movement*0.97) for xi in x]
    y = [max(min(yi, max_score*0.96), -max_score*0.96) for yi in y]

    if shape == "rectangle":
        fig = plt.figure(figsize=(4*16/16, 4*9/16))
        gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 8, 1], width_ratios=[8, 1, 1*9/16])
    else:
        fig = plt.figure(figsize=(4, 4))
        gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 8, 1], width_ratios=[8, 1, 1])
    

    ax_scatter = fig.add_subplot(gs[1:3, 0:2])
    ax_histx = fig.add_subplot(gs[0, 0:2], sharex=ax_scatter)
    ax_histy = fig.add_subplot(gs[1:3, 2], sharey=ax_scatter)

    # # the scatter plot:

    #Generate X and Y grids for contour plot
    news_score = np.linspace(-max_score, max_score, 1000).reshape(-1, 1)
    movement = np.linspace(-max_movement, max_movement, 1000).reshape(1, -1)
    X, Y = np.meshgrid(movement[0], news_score[:, 0])
    c = np.tanh(sigma * movement)
    score = news_score * (1 - np.maximum(c * np.sign(news_score), 0)**2) / max_score
    
    colors = [ (0.9, 0.1, 0.1), (0.8, 0.3, 0.3), (0.8, 0.8, 0.8), (0.3, 0.3, 0.8), (0.1, 0.1, 0.9)]  
    n_bins = [0, 0.25, 0.5, 0.75, 1]
    cmap = LinearSegmentedColormap.from_list("custom_colormap", list(zip(n_bins, colors)))

    ax_scatter.imshow(score, extent=[-max_movement, max_movement, -max_score, max_score], aspect='auto', origin="lower", cmap=cmap)

    # Add contour lines to the heatmap
    contour_levels = np.arange(-max_score+1, max_score, 1)/max_score
    contour_plot = ax_scatter.contour(X, Y, score, levels=contour_levels, colors='black', linewidths=0.4, linestyles = '--', alpha = 0.3, zorder=1)
    # Add labels to contour lines
    manual_locations = [(-max_movement*0.86 if i>(max_score-1) else max_movement*0.86, max_score*l) for i, l in enumerate(contour_levels)]
    ax_scatter.clabel(contour_plot, inline=True, fontsize=8, fmt='%1.2f', manual=manual_locations)

    x_scat,y_scat = x[:-1], y[:-1]
    ax_scatter.scatter(x_scat,y_scat, s=10, edgecolors='k', facecolors='none', linewidth=0.5, alpha=0.75)
    ax_scatter.scatter(scatter_movement, scatter_news_score, s=30, marker="*", edgecolors='k', facecolors='white', zorder=2, linewidths=0.75)

    # Add x,y histagrams
    x_bins = np.linspace(-max_movement, max_movement, 16)
    y_bins = np.linspace(-max_score, max_score,  2*max_score+2)
    ax_histx.hist(x, bins=x_bins, color='black', rwidth=0.92)  # Set color to black
    ax_histy.hist(y, bins=y_bins, orientation='horizontal', color='black', rwidth=0.92)  # Set color to black

    # Remove spacing
    if shape == "rectangle":
        plt.subplots_adjust(hspace=0.1, wspace=9/16*0.1)
    else:
        plt.subplots_adjust(hspace=0.1, wspace=0.1)

    # Remove hist frames 
    ax_histx.spines['top'].set_visible(False)
    ax_histx.spines['right'].set_visible(False)
    ax_histx.spines['left'].set_visible(False)

    ax_histy.spines['top'].set_visible(False)
    ax_histy.spines['right'].set_visible(False)
    ax_histy.spines['bottom'].set_visible(False)

    # Remove hist ticks
    ax_histx.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    ax_histy.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

    # xy labels
    ax_scatter.set_xlabel("Movement (normalized)", labelpad=6, font=fpath)
    ax_scatter.set_ylabel("News Score", labelpad=-2, font=fpath)

    # xy lims
    ax_scatter.set_ylim(-max_score, max_score)
    ax_scatter.set_xlim(-max_movement, max_movement)
   
    if shape == "rectangle":
        save_path = save_image_root + "/score_vs_movement_plot_rec.jpg"
        # Center the plot within the figure
        fig.subplots_adjust(right=0.85, bottom=0.05)
    else:
        save_path = save_image_root + "/score_vs_movement_plot.jpg"
    plt.savefig(save_path, dpi=400, bbox_inches='tight')  
    
    if shape=='rectangle':
        target_dimx = 1600
        target_dimy = 900
        add_white_borders(save_path, target_dimx, target_dimy)

    if show_plot: plt.show()
    plt.close()

def hande_other_tickers(ticker, score, other_ticker):
    try:
        # get list of tickers / co_names that we have logos for
        #logo_names = [_ for _ in os.listdir(logos_root)]
        #logo_names = [_.split('_bw.png')[0] for _ in logo_names if '_bw.png' in _]


        other_keys = list(other_ticker.keys())
        other_key = other_keys[0]
        if len(other_keys) == 1: # and other_key in logo_names:
            try: 
                other_score = other_ticker[other_key]['score']
                other_score = int(other_score)

                # update ticker and score to lists
                ticker = [ticker, other_key]
                score = [score, other_score]
                return ticker, score
            
            except: pass
    except: pass 
    return ticker, score

def make_summary_image(ticker, text, bfactor_str, date, score=None, phrase_list=[], news_source=None, other_ticker=None, shape='square', show_plot=False):

    # Define variables
    debug = False
    background_color = (255, 255, 255)
    news_source = "_".join(news_source.split(' '))
    sf = 4
    size = sf*1600
    sizex = sf*1600
    sizey = sf*1600

    text_height = sf*60
    y_position = sf*250

    if shape == "rectangle":
        sizex = sizex
        sizey = sizey*9//16
        #max_logo_size = (max_logo_size[0]*4/3, max_logo_size[1]*2/3)
        y_position = y_position*3//4

    default_font_size = sf*65
    text_height = sf*70
    small_font_size = default_font_size*6//9
    small_text_height = text_height*6//8

    # plot params ..............................
    xpad = sf*40
    ypad = sf*40
    sp = sf*18

    L = xpad        # Left
    R = sizex-xpad  # Right
    T = ypad        # Top
    B = sizey-ypad  # Bottom

    yfooter = B-ypad*3//2  # where main text ends

    x_1_by_3 = 2*(R-L)//8+L       # x 1/3
    x_2_by_3 = (6*(R-L))//11+L    # x 2/3
    y_1_by_3 = (B-T)//3+2*T         # y 1/3

    y_2_by_3_mid = (T + y_1_by_3)//2  # y midpoint of 2-3 block
    x_0_by_3_mid = (L + x_1_by_3)//2  # x midpoint of 0-1 block (ticker logo)
    y_0_by_3_mid = (T + y_1_by_3)//2  # y midpoint of 0-1 block (ticker logo)
    
    r_text_xlim = R-3*sp
    text_xlim = (x_1_by_3 + x_2_by_3)//2                    # x_thresh of main text (red line)
    text_xlim2 = (r_text_xlim + x_2_by_3)//2 + sizex//10    # x_thresh of reason text (blue line)
    
    # ..............................

    # 0.) make image
    image = Image.new("RGB", (sizex, sizey), background_color)
    draw = ImageDraw.Draw(image)

    if debug == True:
        line_xy(draw, L, mode='x', spacer1 = sp)
        line_xy(draw, R, mode='x', spacer2 = sp)
        line_xy(draw, T, mode='y', spacer1 = sp)
        line_xy(draw, (x_1_by_3+R)/2, mode='x')
        line_xy(draw, (x_1_by_3), mode='x')

        line_xy(draw, R, mode='x', spacer2 = sp)
        # line_xy(draw, r_text_xlim, mode='x', extent=[T, y_1_by_3])
        # line_xy(draw, x_1_by_3, mode='x', extent = [T, y_1_by_3], spacer1 = sp, spacer2 = sp)
        # line_xy(draw, x_2_by_3, mode='x', extent = [T, y_1_by_3], spacer1 = sp, spacer2 = sp)
        # line_xy(draw, y_1_by_3, mode='y', extent = [L, R], spacer1 = sp, spacer2 = sp)
        # line_xy(draw, y_2_by_3_mid, mode='y', extent = [x_1_by_3, R])
        # line_xy(draw, yfooter, mode='y', extent = [L, R], spacer2 = sp)
        # line_xy(draw, text_xlim, mode='x', extent = [yfooter, B], color = 'red')
        # line_xy(draw, yfooter, mode='y', extent = [text_xlim, text_xlim-sizex//20], color = 'red')
        # line_xy(draw, text_xlim2, mode='x', extent = [y_1_by_3-sp, y_1_by_3-3*sp], color = 'blue')
        # line_xy(draw, y_1_by_3-3*sp, mode='y', extent = [text_xlim2, text_xlim2-sizex//20], color = 'blue')
        #line_xy(draw, yfooter, mode='y', extent = [text_xlim, text_xlim-sizex//20], color = 'red')

    # 2.) Add main text
    bold_bool_lst = find_phrases_in_text_bool(text, phrase_list)
    x_position = L
    y_position = y_1_by_3 + 0*sp
    max_y = yfooter - 1*sp
    max_x = R
    wrap_text(draw, text, x_position, y_position, bold_bool_lst, font_path_bold, font_path, default_font_size, size, max_y, max_x, text_height, y_justify='center')

    # 4.) add ticker logo
    try:
        resize_dict = {'CVX': {'factor': 0.75},
                       'AAPL': {'factor': 1.0},
                       'VRTX': {'factor': 1.00},
                        'JNJ': {'factor': 0.95},
                        'HD': {'factor': 0.75},
                        'MA': {'factor': 0.95},
                        'GD': {'factor': 0.75},
                        'MPC': {'factor': 0.75},
                        'UBER':{'factor': 0.85},
                        'WFC': {'factor': 0.85},
                        'ABT': {'factor': 0.75}, 
                        'NEE': {'factor': 1.25},
                        'RIVN': {'factor': 0.75}, 
                        'V': {'factor': 0.85},
                        'TSLA': {'factor': 0.85},
                        'PG': {'factor': 0.85},
                        'EOG': {'factor': 0.85},
                        'ETH': {'factor': 0.95},
                        'INTC': {'factor': 0.95}
                        }

        try: resize_factor = resize_dict[ticker]['factor']
        except: resize_factor = 1
        
        # WORKING OLD:

        # if type(ticker) is not list: 
        #     path = logos_root + f"/{ticker}_bw.png"
        # else: # multiple tickers
        #     path = [logos_root + f"/{t}_bw.png" for t in ticker] 
       
        # Pull logos from wiki if they don't exist

        # handle other ticker --
        ticker, score = hande_other_tickers(ticker, score, other_ticker)

        if type(ticker) is not list: 
            ticker = [ticker]
            score = [score]
        
        for t in ticker:
            # get company name
            try: co_name = assets_df[assets_df['ticker'] == t]['name'].values[0]
            except: co_name = t
            # check if ticker logos exist
            if f"{t}_bw.png" not in os.listdir(logos_root):
                # add logo
                add_logo_from_wiki(co_name, t, smart_llm, logos_root)
                time.sleep(1)
        
        ticker = [t for t in ticker if f"{t}_bw.png" in os.listdir(logos_root)]
        score = [score[i] for i, t in enumerate(ticker) if f"{t}_bw.png" in os.listdir(logos_root)]
        path = [os.path.join(logos_root, f"{t}_bw.png") for t in ticker]      
        image_size_max = [(x_1_by_3-L)-3*sp, (y_1_by_3-T)-4*sp]
        image_size_max = [int(image_size_max[0]*resize_factor), int(image_size_max[1]*resize_factor)]
        
        xy = [x_0_by_3_mid, y_0_by_3_mid]


        black_rgba = [[0, 0, 0, 255], [0, 0, 0, 255]]

        add_image(image, path, xy, image_size_max, recolor = black_rgba)
        line_xy(draw, xy[0] + ((x_1_by_3-L)-3*sp)*3//4, mode='x', extent=[xy[1]-image_size_max[1]//2, xy[1]+image_size_max[1]//2], line_width=20)

        # try: resize_factor = resize_dict[ticker]['factor']
        # except: resize_factor = 1
        # path = logos_root + f"/{ticker}_bw.png"
        # image_size_max = [(x_1_by_3-L)-3*sp, (y_1_by_3-T)-3*sp]
        # image_size_max = [int(image_size_max[0]*resize_factor), int(image_size_max[1]*resize_factor)]
        # xy = [x_0_by_3_mid, y_0_by_3_mid]
        # add_image(image, path, xy, image_size_max)
        # line_xy(draw, xy[0] + ((x_1_by_3-L)-3*sp)*3//4, mode='x', extent=[4*sp, y_1_by_3-5*sp//2], line_width=20)

        # WORKING!
        # ticker_lst = ['AAPL', 'AMAZN']
        # y_div = (4*sp + y_1_by_3-5*sp//2)//2
        # for i, tic in enumerate(ticker_lst):
        #     try: resize_factor = resize_dict[tic]['factor']
        #     except: resize_factor = 1
        #     path = logos_root + f"/{tic}_bw.png"
        #     image_size_max = [((x_1_by_3-L)-3*sp), ((y_1_by_3-T)-3*sp)//len(ticker_lst)]
        #     image_size_max = [int(image_size_max[0]*resize_factor), int(image_size_max[1]*resize_factor)]
        #     xy = [x_0_by_3_mid, y_div + ((-1+2*i)*image_size_max[1])//2]
        #     image_size_max[1] = image_size_max[1]//1.2 # add y padding 
        #     add_image(image, path, xy, image_size_max)
        #     line_xy(draw, xy[0] + ((x_1_by_3-L)-3*sp)*3//4, mode='x', extent=[4*sp, y_1_by_3-5*sp//2], line_width=20)
        #line_xy(draw, y_div, mode='y', extent=[3*sp, x_1_by_3], line_width=10, )

    except: pass

    # 4.) add bearometer gauge

    if len(ticker) == 1:
        ticker = ticker[0]
        score = score[0]

    try: 
        b_idx = ['long-term', 'mid-term', 'near-term'].index(bfactor_str)
        if type(score) is list and len(score)>1:
            # check if temp_image_path dir if exist, if not create it
            if not os.path.exists(temp_image_path):
                os.makedirs(temp_image_path)
            path = os.path.join(temp_image_path, f"gauge_temp.png")
            process_ij((score[0], score[1], b_idx), ticker_lst=ticker, save_path=path)
            time.sleep(1)
        else:
            path = gauge_path.replace('_X2', '')
            path = path.replace('X', str(int(score)))
        path = path.replace('Y', str(b_idx))
        image_size_max = [(x_1_by_3+R)/2-2*sp, (y_1_by_3-T)-2*sp]
        xy = [int((x_1_by_3+R)/2) , y_0_by_3_mid]
        add_image(image, path, xy, image_size_max, x_justify='center', y_justify='center')
    except: pass

    # 5.) add news sorce logo
    try:
        assert news_source != ticker # no logo if news source is the company website
        resize_dict = {'Associated_Press': {'factor': 1.40},
                       'InvestorsObserver': {'factor': 0.90},
                       'Investorplace': {'factor': 1.35}, 
                       'Benzinga': {'factor': 0.75}, 
                       'CoinDesk': {'factor': 0.95},
                       'BioSpace': {'factor': 1.25}, 
                       'Al_Jazeera': {'factor': 0.85},
                       'FTC': {'factor': 1.45},
                       'DOT': {'factor': 1.35},
                       'Cointelegraph': {'factor': 1.40}, 
                       'Fierce Biotech': {'factor': 1.35},
                       'Fierce Pharma': {'factor': 1.35},
                       'CNBC': {'factor': 1.15}
        }
        try: resize_factor = resize_dict[news_source]['factor']
        except: resize_factor = 1

        path = logos_root + f"/{news_source}_bw.png"
        image_size_max = [2*sizex//10, 10*(B-yfooter)//11]
        image_size_max = [int(image_size_max[0]*resize_factor), int(image_size_max[1]*resize_factor)]
        xy = [R, (B+yfooter)//2]
        news_logo_width, _ = add_image(image, path, xy, image_size_max, y_justify='center', x_justify='right')
    except: news_logo_width = 0

    # 6.) add date
    date_am_pm = datetime.strptime(date, '%Y-%m-%d %H:%M')
    date_am_pm = date_am_pm.strftime('%Y-%m-%d %I:%M %p') 
    datetime_text = f"{date_am_pm} (ET)"
    add_text(draw, [R-news_logo_width-2*sp, (B+yfooter)//2 -sp//4], datetime_text, font_path_tnr, small_font_size, x_justify='right', y_justify='center')

    # 7.) save image
    if shape == "rectangle":
        save_path = save_image_root + "/summary_image_rec.jpg"
        image.save(save_path)
    else:
        save_path = save_image_root + "/summary_image.jpg"
        image.save(save_path)
    
    if show_plot: display(image)
    image.close()

#.................................plotting-protocols.................................

def check_post_url(post_text):
    '''
    Used to check for bad urls in X post text -- API bug
    '''
    try:
        post_url = 'https'+post_text.split('https')[1]
        post_url = post_url.replace(' ', '')
        bad_urls = []
        bad_urls.append('https://t.co/HTG1QwxL7A') # t.co formatted finance.yahoo.com

        bad_urls.append('https://t.co/Y0dmIc6uW1') # t.co formatted InsideEVs.com

        bad_urls.append('https://t.co/5WQL95dwd4') # t.co formatted uk.finance.yahoo.com
        bad_urls.append('https://t.co/I5fRVWrJV1') # t.co formatted uk.finance.yahoo.com world

        bad_urls.append('https://t.co/Du69fNZRa6') # t.co formatted ca.finance.yahoo.com

        bad_urls.append('https://t.co/yUUM4OSy7H') # t.co formatted seekingalpha.com
        bad_urls.append('https://t.co/VpHz48TzAV') # t.co formatted seekingalpha.com world

        bad_urls.append('https://t.co/GSQR09ttyP') # t.co formatted businesswire.com
        bad_urls.append('https://t.co/JDYLXE9U1q') # t.co formatted businesswire.com world 

        bad_urls.append('https://t.co/oRiQKeg9hX') # t.co formatted investorsobserver.com
        
        check_bool = post_url not in bad_urls and post_url.startswith('https://t.co/')
        return check_bool
    except:
        check_bool = False
        return check_bool

def post_update(article_idx, articles_df, posts_df, dev_mode=False, debug=False):
    """
    Function that posts an update to an existing thread with a news story + score meter image.
    """
    if dev_mode:
        account = 'world'
    else:
        account = 'finance'

    original_id, update_summary, articles_df = llm_summarize_news_update(article_idx, smart_llm, articles_df)

    if original_id is None:
        articles_df = log_outcome(article_idx, articles_df, "update original_id not found")
        return posts_df, articles_df
    try:
        # check if the original post is a story
        if original_id not in posts_df["article_id"].values:
            print("Original article of update was not posted.")
            articles_df = log_outcome(article_idx, articles_df, "update original_id not posted")
            return posts_df, articles_df
        else:
            # get the thread_id of the original post
            thread_id = posts_df.loc[posts_df["article_id"] == original_id, "thread_id"].values[0]
            last_post_id = posts_df.loc[posts_df["thread_id"] == thread_id].iloc[-1]["post_id"]
            post_text = f"Update: {update_summary}"
        
        if debug:
            print("Replying with update to post_id", last_post_id, "in thread", thread_id)
            print("Post text", post_text)

        reply_id, posts_df = X_post(text=post_text, posts_df=posts_df, reply_id=last_post_id, account=account, debug=debug)
        if reply_id is not None:
            posts_df.loc[posts_df["post_id"] == reply_id, "post_type"] = "update"
            posts_df.loc[posts_df["post_id"] == reply_id, "thread_id"] = thread_id
            posts_df.loc[posts_df["post_id"] == reply_id, "article_id"] = articles_df.loc[article_idx, "article_id"]

        articles_df = log_outcome(article_idx, articles_df, "social post")
        print("\033[93m" + "Update post is complete and posted." + "\033[0m")
    except Exception as e:
        print("Failed to post update to X.")
        if debug:
            traceback.print_exc()
    return posts_df, articles_df

def post_story(article_idx, articles_df=None, posts_df=None, dev_mode=False, debug=False, filter_bool=False):
    """
    Function for writing a brand new thread with a news story + score meter image.

    """
    if dev_mode:
        account = 'world'
    else:
        account = 'finance'
    d, llm_cost = get_article_post_data_dict(articles_df, article_idx=article_idx, filter_bool=filter_bool)
    if d['article_score'] !=0 and ~np.isnan(d['article_score']):
        image_paths = []
        try:
            make_summary_image(d['ticker'], d['image_text'], d['timeframe'], d['pub_date'], \
                            d['article_score'], d['text_keyphrases'], d['news_source'], \
                            d['other_ticker'], "rectangle", show_plot=debug)
            image_paths.append(save_image_root + "/summary_image_rec.jpg")
        except Exception as e: 
            print(f"An error occurred in the make_summary_image: {str(e)}")
            if debug:
                traceback.print_exc()
            
        assert len(image_paths)>0
        co_name = assets_df[assets_df['ticker'] == d['ticker']]['name'].to_list()[0]
        X_caption = make_X_caption(co_name, d['ticker'], d['article_score'], d['url'], d['news_source'], d['news_type']) #, article_score, url, news_source, news_type)
        
        if debug:
            print("Post caption", X_caption)
            display(Image.open(image_paths[0]))
    
        # posting functions
        try:
            post_id, posts_df = X_post(text=X_caption, posts_df=posts_df, image_paths=image_paths, account=account, debug=debug)
            thread_id = post_id # set thread_id to the story's post_id if a new thread is created
            if post_id is not None:
                posts_df.loc[posts_df["post_id"] == post_id, "post_type"] = "story"
                posts_df.loc[posts_df["post_id"] == post_id, "thread_id"] = thread_id
                posts_df.loc[posts_df["post_id"] == post_id, "article_id"] = d['article_id']
                posts_df.loc[posts_df["post_id"] == post_id, "image_text"] = d['image_text']

            posts_df = post_story_link(article_idx, thread_id, articles_df, posts_df, account, dev_mode, debug)

            time.sleep(1) # avoid rate limit? (not sure if necessary)
            posts_df = post_story_context(articles_df, article_idx, thread_id, posts_df, account, debug, llm_name=dumb_llm)

        except Exception as e: 
            print("Failed to post to X.")
            if debug:
                traceback.print_exc()

        if articles_df is not None:
            articles_df = log_outcome(article_idx, articles_df, "social post")
            articles_df.loc[article_idx, 'llm_cost'] += llm_cost
            print("\033[93m" + "Social post is complete and posted." + "\033[0m")
    return posts_df, articles_df

def post_story_link(article_idx, thread_id, articles_df, posts_df, account, dev_mode, debug):
    # add reply to post with source link:
    source_reply = f"\nSource: {articles_df.loc[article_idx, 'url']}"
    if debug:
        print("Source reply text:", source_reply)
    # get the most recent post in the thread
    if thread_id is not None:
        thread_reply_id = posts_df.loc[posts_df["thread_id"] == thread_id, "post_id"].iloc[-1]
        reply_id, posts_df = X_post(text=source_reply, posts_df=posts_df, reply_id=thread_reply_id, account=account, debug=debug)
        if reply_id is not None:
            posts_df.loc[posts_df["post_id"] == reply_id, "post_type"] = "link"
            posts_df.loc[posts_df["post_id"] == reply_id, "thread_id"] = thread_id
            posts_df.loc[posts_df["post_id"] == reply_id, "article_id"] = articles_df.loc[article_idx, "article_id"]
            source_tweet_text = posts_df.loc[posts_df["post_id"] == reply_id, 'post_text'].values[0]

        # check for url error in post (API bug), remove if broken
        if not check_post_url(source_tweet_text):
            print("Removing link post with bad URL.")
            posts_df = remove_X_post(reply_id, posts_df, account=account)
            
    return posts_df

def make_context_post_text(articles_df, article_idx, llm_name, score_threshold=40, debug=False):
    post_text = None
    llm_cost = 0
    summary = articles_df.iloc[article_idx]['summary']

    try:
        # get topics to expand on
        topics_dict, llm_cost = extract_topics_to_expand_on(articles_df, article_idx, llm_name)
        topics = topics_dict['topics']
        urls = topics_dict['urls']
        scores = topics_dict['familiarity_scores']
        specificity = topics_dict['specificity']
        if debug: print("Wiki_topics:", topics, "Wiki_scores:", scores)

        # sort by scores
        topics, urls, scores, specificity = zip(*sorted(zip(topics, urls, scores, specificity), key=lambda x: x[2]))
        
        for topic, url, score, spec in zip(topics, urls, scores, specificity):
            # only include topics that are specific with a low score
            if spec and score <= score_threshold:
                wiki_body, _ = get_wikipedia_data(url, save_root=None)
                if wiki_body is not None:

                    post_text, is_helpful, llm_cost_new = create_supplementary_note(topic, wiki_body, summary, dumb_llm)
                    llm_cost += llm_cost_new
                    if debug: print("Wiki_post_text:", post_text, "Wiki_is_helpful:", is_helpful)
                    post_text = reduce_text_to_target(post_text, llm_name, len_target = 38, max_recursions=3, mode='normal')
                    
            if post_text is not None and is_helpful: break
    except: pass
    
    articles_df.iloc[article_idx]['llm_cost'] += llm_cost  

    if post_text is not None:
        post_text = 'For context: ' + post_text + " " + str(url)
    return post_text

def post_story_context(articles_df, article_idx, thread_id, posts_df, account, debug, llm_name=dumb_llm):
    post_text = make_context_post_text(articles_df, article_idx, llm_name, debug=debug)
    if debug: print("Post text:", post_text)
    if post_text is not None: 
        if thread_id is not None:
            # get the most recent post in the thread
            thread_reply_id = posts_df.loc[posts_df["thread_id"] == thread_id, "post_id"].iloc[-1]
            reply_id, posts_df = X_post(text=post_text, posts_df=posts_df, reply_id=thread_reply_id, account=account, debug=debug)
            if reply_id is not None:
                posts_df.loc[posts_df["post_id"] == reply_id, "post_type"] = "wiki_context"
                posts_df.loc[posts_df["post_id"] == reply_id, "thread_id"] = thread_id
                posts_df.loc[posts_df["post_id"] == reply_id, "article_id"] = articles_df.loc[article_idx, "article_id"]

    return posts_df

def post_meta(articles_df, posts_df=None, dev_mode=False, debug=False, meta_type='week_in_review', llm_name=smart_llm):
    assert meta_type in ['week_in_review'], "Invalid meta_type"

    if dev_mode: account = 'world'
    else: account = 'finance'

    try:
        if meta_type == 'week_in_review':
            post_text, article_scores_dict = week_in_review_text(articles_df)            
            post_text_T, post_text_B = week_in_review_thread_texts(articles_df, article_scores_dict, llm_name)
            if debug: return post_text, post_text_T, post_text_B

            if not debug: post_id, posts_df = X_post(text=post_text, posts_df=posts_df, reply_id=None, account=account, debug=debug)
            if post_id is not None:
                posts_df.loc[posts_df["post_id"] == post_id, "post_type"] = "week_in_review"
                posts_df.loc[posts_df["post_id"] == post_id, "thread_id"] = post_id
                posts_df.loc[posts_df["post_id"] == post_id, "article_id"] = None

                if post_text_T and post_text_B:
                    reply_id, posts_df = X_post(text=post_text_T, posts_df=posts_df, reply_id=post_id, account=account, debug=debug)
                    if reply_id is not None:
                        time.sleep(1)
                        posts_df.loc[posts_df["post_id"] == reply_id, "post_type"] = "week_in_review_top"
                        posts_df.loc[posts_df["post_id"] == reply_id, "thread_id"] = post_id
                        posts_df.loc[posts_df["post_id"] == reply_id, "article_id"] = None

                        rreply_id, posts_df = X_post(text=post_text_B, posts_df=posts_df, reply_id=reply_id, account=account, debug=debug)
                        if rreply_id is not None:
                            time.sleep(1)
                            posts_df.loc[posts_df["post_id"] == rreply_id, "post_type"] = "week_in_review_bottom"
                            posts_df.loc[posts_df["post_id"] == rreply_id, "thread_id"] = post_id
                            posts_df.loc[posts_df["post_id"] == rreply_id, "article_id"] = None


        print("\033[93m" + f"Social post ({meta_type}) is complete and posted." + "\033[0m")
    
    except Exception as e: 
        print(e)
        print(f"failed to post ({meta_type}) to X")
    
    return posts_df

def post_scheduler(posts_df, articles_df, dev_mode=False, debug=False):
    date_today = datetime.now(pytz.timezone('America/New_York'))
    is_sunday = date_today.weekday() == 6
    is_noon = date_today.time() >= datetime.strptime('12:00', '%H:%M').time()
    
    # "week in review" (WIR) post 
    if is_sunday and is_noon:
        # check for 'week_in_review' made in past 2 days (to be safe)
        dates = posts_df[posts_df['post_type'] == 'week_in_review']['date'].dropna()
        dates = pd.to_datetime(dates).dt.tz_localize('America/New_York', ambiguous='NaT', nonexistent='shift_forward')  # Ensure dates are timezone-aware
        WIR_has_been_posted = any((date_today - dates).dt.days <= 2)
        
        if not WIR_has_been_posted:
            posts_df = post_meta(articles_df, posts_df=posts_df, dev_mode=dev_mode, debug=debug, meta_type='week_in_review')

    return posts_df