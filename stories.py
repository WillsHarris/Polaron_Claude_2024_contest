import pandas as pd
import os
import numpy as np
import json
import re
import sys
import cairosvg
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
from llm import create_post_text, extract_topics_to_expand_on, create_supplementary_note

def add_story(article_idx, smart_llm, stories_df, articles_df, assets_df, debug):

    article_id = articles_df.loc[article_idx, "article_id"]
    asset_name = assets_df[assets_df['ticker'] == articles_df.iloc[article_idx]['source_ticker']]['name'].values[0]
    summary = articles_df.loc[article_idx, "summary"]
    # create post text
    text = create_post_text(asset_name, summary, smart_llm)
    # create context text
    context, context_link = make_context_text(articles_df, article_idx, smart_llm)
    # get image url
    #image_url = get_image_url() fill in here wills

    if not debug:
        new_row = pd.DataFrame({"article_id": article_id,
                                "text": text,
                                "context": context,
                                "context_link": context_link,
                                "image_url": None,
                                }, 
                                index=[0])
        stories_df = pd.concat([stories_df, new_row], ignore_index=True)
    else:
        print("Not adding to stories in debug mode.")
    return stories_df

def make_context_text(articles_df, article_idx, llm_name, score_threshold=30, debug=False):
    post_text = None
    url = None
    summary = articles_df.iloc[article_idx]['summary']
    try:
        # get topics to expand on
        topics_dict = extract_topics_to_expand_on(articles_df, article_idx, llm_name)
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

                    post_text, is_helpful = create_supplementary_note(topic, wiki_body, summary, llm_name)
                    if debug: print("Wiki_post_text:", post_text, "Wiki_is_helpful:", is_helpful)
                    
            if post_text is not None and is_helpful: break
    except Exception as e:
        print(e)
    
    return post_text, url

def get_wikipedia_data(wiki_url, save_root=None):
    response = requests.get(wiki_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    body_text = None
    images_data = []

    try:
        body_ps = soup.find_all('p')
        body_lst = [p.text for p in body_ps]
        body_text = ' '.join(body_lst)
        # fix formatting
        body_text = str(body_text.replace('\n', ' ').lstrip())
        body_text = re.sub(r'\[.*?\]', '', body_text)
        assert len(body_text) > 50
    except Exception as e:
        body_text = None
    
    try:
        infobox = soup.findAll(class_='infobox')
        if not infobox: # alt infobox formatting 
            infobox = soup.findAll(class_='mw-default-size')
        infobox_images = infobox[0].find_all(href=True)
        for image_link in infobox_images:
            try:
                if save_root is not None:

                    image_data = {}
                    image_url = 'https://en.wikipedia.org' + image_link.get('href')
                    assert image_url.endswith(('.png', '.jpg', '.svg'))
                    image_data['image_url'] = image_url
                
                    response = requests.get(image_url, allow_redirects=True)
                    response.raise_for_status()

                    soup = BeautifulSoup(response.content, 'html.parser')

                    image_name = soup.find(class_='firstHeading mw-first-heading').text
                    image_name = image_name.replace('File:', '').replace(" ", "_")
                    image_data['image_name'] = image_name

                    assert 'png' in image_url or 'jpg' in image_url or 'svg' in image_url
                    
                    author_td = soup.find('td', text='Author')
                    if author_td:
                        author = author_td.find_next_sibling('td').text.replace('\n', '')
                        image_data['author'] = author

                    description = soup.find(class_='description mw-content-ltr en')
                    if description:
                        description_text = description.text.replace('English: ', ' ').lstrip()
                        image_data['description'] = description_text

                    license_tag = soup.find('a', href=lambda href: href and 'licenses' in href or 'public_domain' in href)
                    if license_tag:
                        image_data['license'] = license_tag.text

                    image_file_url_lst_og = [a.get('href') for a in soup.find_all('a', href=True)]
                    image_file_url_lst = [url for url in image_file_url_lst_og if 'upload' in url and image_name in url]
                    
                    if image_file_url_lst == []:
                        image_file_url_lst = [url for url in image_file_url_lst_og if 'upload' in url]
                        for url in image_file_url_lst:
                            if sum(element in url for element in image_name.split('.')[0].split("_"))>1:
                                image_file_url_lst = [url]
                                break

                    
                    if image_file_url_lst:
                        image_file_url = 'https:' + image_file_url_lst[0]
                        image_data['image_file_url'] = image_file_url
                        
                        response = requests.get(image_file_url, headers={'User-Agent': 'Mozilla/5.0'})
                        
                        # create save_root if it doesn't exist
                        if not os.path.exists(save_root): os.makedirs(save_root)
                        

                        if ".png" in image_file_url or ".jpg" in image_file_url:
                            img_data = BytesIO(response.content)
                            img = Image.open(img_data)
                            img.save(os.path.join(save_root, image_name))
                        elif ".svg" in image_file_url:
                            # Convert svg to png using cairosvg
                            svg_data = response.content
                            image_name = image_name.replace('.svg', '.png')
                            image_data['image_name'] = image_name
                            output_path = os.path.join(save_root, image_name)
                            cairosvg.svg2png(bytestring=svg_data, write_to=output_path, scale=10.0)
                
                images_data.append(image_data)

            except Exception as e: pass
    except Exception as e: pass

    return body_text, images_data