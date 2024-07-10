import json
import os
import re
import ast
import numpy as np

import tiktoken

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from openai import OpenAI
import pandas as pd
from util import log_outcome

# Get the directory that this script is in
dir_path = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(dir_path, "config.json"), 'r') as file:
    config = json.load(file)

smart_llm = config['smart_llm']
dumb_llm = config['dumb_llm']

load_dotenv() # load environment variables

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

#----------------------------------Token-Tracking ----------------------------------

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    try: 
        encoding = tiktoken.encoding_for_model(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens
    except:
        return 0

def get_embedding(text, model="text-embedding-3-large"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding

#----------------------------------LLM-----------------------------------------

def llm_score_summary(article_idx, max_score, llm_name, assets_df, articles_df):
    multi_stock_mode = False
    ticker = articles_df.loc[article_idx, 'source_ticker']
    asset_name = assets_df.loc[assets_df['ticker'] == ticker, 'name'].iloc[0]
    asset_names = None
    asset_tickers = None
    summary = articles_df.loc[article_idx, 'summary']
    other_tickers = articles_df.loc[article_idx, 'other_ticker']
    if other_tickers == 'None': other_tickers = None # temp fix
    
    # Check for other trakced tickers
    if not pd.isna(other_tickers):
        other_tickers_dict = eval(other_tickers) # list of tickers FIXME: (to become dict)
        other_tickers = list(other_tickers_dict.keys())
        other_tickers_tracked = []
        co_names_tracked = []
        for ot in other_tickers:
            if ot != ticker: # and ot in tickers_lst: 
                other_tickers_tracked.append(ot)
                try: # add company name 
                    co_names_tracked.append(assets_df[assets_df['ticker'] == ot]['name'].iloc[0])
                except: # use ticker if its not found
                    co_names_tracked.append(ot)

        # multi_stock_mode vars
        if len(other_tickers_tracked) > 0: 
            multi_stock_mode = True
            # add source company to begining of list
            other_tickers_tracked = [ticker] + other_tickers_tracked
            co_names_tracked = [asset_name] + co_names_tracked

            asset_names = str(co_names_tracked)
            asset_tickers = str(other_tickers_tracked)

    if multi_stock_mode is False:
        desc = (
            "Integer within range from -3 to 3 inclusive. -3: extremely bad news, "
            "very rare. -2: very bad news, rare. -1: bad news. 0: neutral or irrelevant news, esp. involving "
            "opinions, should be extremely common. 1: good news, 2: very good news, rare. 3: "
            "extremely good news, very rare."
            )

        prompt = (
            "Based on the following summary of a news article, determine how good or bad the news is for "
            "{asset_name} for stock: {ticker} on an exponential scale. Score 0 for anything (ie opinions) "
            "that is not strictly a reported event. Assume 0 until confident otherwise. If not 0 assume "
            "+/- 1's until confident in +/- 2's, etc."
            )
    
    if multi_stock_mode is True:
        desc = ( 
            "List of integers with values ranging from -3 to 3 inclusive. -3: extremely bad news, "
            "very rare. -2: very bad news, rare. -1: bad news. 0: neutral or irrelevant news, esp. involving "
            "opinions, should be extremely common. 1: good news, 2: very good news, rare. 3: "
            "extremely good news, very rare."
            )

        prompt = (
            "Based on the following summary of a news article, determine how good or bad the news is for "
            "{asset_names} for stocks: {asset_tickers} in that order on an exponential scale. Score 0 for anything (ie opinions) "
            "that is not strictly a reported event. Assume 0 until confident otherwise. If not 0 assume "
            "+/- 1's until confident in +/- 2's, etc."
            )
        
    # -------------

    response_schemas = [
        ResponseSchema(
            name="score",
            description=desc,
            type= "list" if multi_stock_mode else "int"
        ),
        ResponseSchema(
            name="reason",
            description="Brief explanation for the provided scores in two sentences or less.",
            type="str"
        )
    ]
    
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    template= prompt + """
    {format_instructions} \n
    Here is the news: {summary}
    """
    

    prompt = PromptTemplate(
        template=template,
        input_variables= [summary, asset_names, asset_tickers] if multi_stock_mode else [summary, asset_name, ticker],
        partial_variables={"format_instructions": format_instructions} 
    )


    chat = ChatAnthropic(model=llm_name, temperature=0)

    if multi_stock_mode is False:
        _input = prompt.format_prompt(summary=summary, asset_name=asset_name, ticker=ticker)
    elif multi_stock_mode is True:
        _input = prompt.format_prompt(summary=summary, asset_names=str(asset_names), asset_tickers=str(asset_tickers))

    output = chat.invoke(_input.to_messages()).content

    dict_out = output_parser.parse(output)

    # update the articles df:
    if multi_stock_mode is False:
        article_score = int(dict_out['score'])
    elif multi_stock_mode is True:
        article_scores = dict_out['score']
        article_score = article_scores[0]
        for i, t in enumerate(other_tickers_tracked):
            try:
                other_tickers_dict[t]['score'] = article_scores[i]
            except: pass
        
        articles_df.loc[article_idx, ['other_ticker']] = str(other_tickers_dict)
    
    reason = dict_out['reason']
    articles_df.loc[article_idx, ['article_score', 'reason']] = article_score, reason
    print("Article score:", article_score)
    if article_score == 0:
        print("Article score is 0, no action required.")
        articles_df = log_outcome(article_idx, articles_df, "article is 0")

    return article_score, articles_df

def fix_indirect_summary(summary, llm_name):
    # define the response schemas
    response_schemas = [ResponseSchema(
        name="summary",
        description="Str: reprased summary "
        )]
  
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    template='''
    Edit the following summary to read as the original source of the news. That is, \
    do not mention the article and avoid using phrases like "The text" or "The article".
    '''  

    template+='''
    {format_instructions}
    Here is the text: {summary}  
    '''

    prompt = PromptTemplate(
        template=template,
        input_variables=[summary],
        partial_variables={"format_instructions": format_instructions}
    )

    chat = ChatAnthropic(model=llm_name, temperature=0)

    _input = prompt.format_prompt(summary=summary)
    output = chat.invoke(_input.to_messages()).content
  
    try: dict_out = output_parser.parse(output)
    except:
       return
    new_summary = dict_out['summary']
    return new_summary

def llm_summarize_news_update(article_idx, llm_name, articles_df):

    def get_dups_dict(articles_df, article_idx):
        dups_dict = articles_df.iloc[article_idx]['duplicate_articles']
        dups_dict = dups_dict.replace('nan', 'None').replace('array', '').replace(',\n      dtype=object', '')
        dups_dict = ast.literal_eval(str(dups_dict).replace('nan', "None"))
        return dups_dict
    
    # get data on the news update
    news_update = articles_df.loc[article_idx]['summary']
    dups_dict = eval(articles_df.iloc[article_idx]['duplicate_articles'])
    assert dups_dict is not None
    original_id = dups_dict['match_id']
    match_type = dups_dict['match_type']
    assert match_type == 'update'

    # get data on the original news
    if original_id not in articles_df['article_id'].values:
        print("Original article not found in posts_df.")
        return None, None, articles_df
    
    original_idx = articles_df[articles_df['article_id'] == original_id].index[0]
    
    try: # use composite news if available
        dups_dict_original = get_dups_dict(articles_df, original_idx)
        news_original = dups_dict_original['composite_summary']
    except:
        news_original = articles_df.loc[original_idx]['summary']

    response_schemas = [
        ResponseSchema(
            name="update_summary",
            description="A one sentence summary that highlights exclusively the new information provided by 'news_update' not included in 'news_original'.",
            type = "string" 

        ),
        ResponseSchema(
            name="composite_summary",
            description="A brief summary that combines the information from both 'news_original' and 'news_update', applicable only if the update is relevant to the original news.",
            type = "string" 

        ),
        ResponseSchema(
            name="is_update_relevant",
            description="A boolean indicating whether the 'news_update' is relevant to 'news_original' and provides new information on the same topic.",
            type = "bool" 

        )
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    
    # build the prompt
    template="""
    This function generates a one sentence summary of the update provided by 'news_update' over 'news_original', and a brief composite summary of both texts if they are on the same topic.
    """ 
    
    template+="""
    {format_instructions}
    news_original = {news_original}
    news_update = {news_update}

    Do not include comments in the output JSON string.
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=['news_original', 'news_update'],
        partial_variables={"format_instructions": format_instructions} 
    )
    _input = prompt.format_prompt(news_original=news_original, news_update=news_update) 
    chat = ChatAnthropic(model=llm_name, temperature=0)
    output = chat.invoke(_input.to_messages()).content
    
    dict_out = output_parser.parse(output)
    
    # extract data from dict_out
    update_summary = dict_out['update_summary']
    composite_summary = dict_out['composite_summary']
    is_update_relevant = dict_out['is_update_relevant']

    
    if is_update_relevant:
        dups_dict['update_summary'] = update_summary
        dups_dict['composite_summary'] = composite_summary
        articles_df.loc[article_idx, 'duplicate_articles'] = str(dups_dict)
        return original_id, update_summary, articles_df
    else:
        print("The update is not relevant to the original news.")
        articles_df = log_outcome(article_idx, articles_df, "update not valid")
        return None, None, articles_df

def remove_stock_tickers(text):
    # Regex to match and remove the pattern "(Exchange: TICKER)"
    pattern1 = r'\((NYSE|NASDAQ): [A-Z]+\)'
    pattern2 = r'\((NYSE|NASDAQ):[A-Z]+\)'
    
    # clean text
    cleaned_text = text
    cleaned_text = re.sub(pattern1, '', cleaned_text)
    cleaned_text = re.sub(pattern2, '', cleaned_text)
    cleaned_text = cleaned_text.replace(' , ', ' ')
    cleaned_text = cleaned_text.replace('  ', ' ')
    return cleaned_text 

def llm_get_summary(article_idx, llm_name, assets_df, articles_df):
    ticker = articles_df.loc[article_idx, 'source_ticker']
    asset_name = assets_df.loc[assets_df['ticker'] == ticker, 'name'].iloc[0]
    text_input = articles_df.loc[article_idx, 'body']

    # define the response schemas
    response_schemas = [ResponseSchema(
        name="article_summary",
        description="Str: A brief summary, no more than two sentences, of the provided article.", 
        type="string"
        ),
        ResponseSchema(
        name="on_topic_bool",
        description="Bool: True if the article is mostly about the mentioned company. Otherwise, False", 
        type="bool"
        )]
  
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    template='''
    In your own words, write a brief summary, no more than two sentences, of the provided text.\
    Emphasize aspects of the text regarding company {asset_name} (ticker: {ticker}),  \
    in your summary. Write as if you are the publisher and DO NOT plagiarize the text.
    '''  

    template+='''
    {format_instructions}
    Here is the text: {text_input}  
    '''

    prompt = PromptTemplate(
        template=template,
        input_variables=[text_input, asset_name, ticker],
        partial_variables={"format_instructions": format_instructions}
    )

    chat = ChatAnthropic(model=llm_name, temperature=0)

    _input = prompt.format_prompt(text_input=text_input, asset_name=asset_name, ticker=ticker)
    output = chat.invoke(_input.to_messages()).content
  
    try: 
        dict_out = output_parser.parse(output)
    except:
        dict_out ={"article_summary": "".join(str(output).split('"article_summary":')[1].split('"on_topic_bool":')[0].split('"')[1:-1]), 
                    "on_topic_bool": str('true' in str(output).split('"on_topic_bool":')[1].lower())}
    # make on_topic_bool
    on_topic_bool = 'true' in str(dict_out["on_topic_bool"]).lower()
    
    # Fix summary for common issues ---
    summary = dict_out['article_summary']
    if "The text" in summary or "The article" in summary:
        summary = fix_indirect_summary(summary, llm_name=smart_llm)

    summary = remove_stock_tickers(summary)
    articles_df.loc[article_idx, 'summary'] = summary
    article_id = articles_df.loc[article_idx, 'article_id']
    if not on_topic_bool:
        print("\033[31m" + "Article rejected for being off-topic." + "\033[0m")
        articles_df = log_outcome(article_idx, articles_df, "off-topic")

    return summary, on_topic_bool, articles_df

def clean_other_ticker_lst(other_ticker_lst, source_ticker, assets_df):
    otl = other_ticker_lst
    co_name = assets_df[assets_df['ticker'] == source_ticker]['name'].values[0]

    # Handle string otl
    try: otl = eval(str(otl))
    except: return None

    # Handle NASDAQ: and NYSE: in list
    if type(otl) == list: 
        otl = [x.split(':')[-1] for x in otl]

    # Handle None
    if otl is None: return None
    if otl == ['None']: return None

    # Replace co_name with ticker -- TEMPOARY, find more robust solution
    tickers_dict = {'UAL': 'United Airlines', 
                    'QCOM': 'Qualcomm',
                    'EASA': 'Airbus',
                    'GOOGL': 'Alphabet',
                    'GOOGL': 'GOOG',
                    'ALK': 'Alaska Airlines',
                    'ALK': 'Alaska Air Group',
                    'LYFT': 'Lyft',
                    'LMT': 'Lockheed Martin',
                    'Blackrock': 'BLK',
                    'WFC': 'Wells Fargo',
                    'SSNLF': 'Samsung',
                    }

    for k, v in tickers_dict.items():
        if v in otl:
            otl[otl.index(v)] = k

    # Remove keywords from 'blacklist'
    remove_keywords_lst = ['fda', 'cdc', 'nasdaq', 'nyse', 'sec', 'cna', 'dow jones',\
                            's&p', 'none', 'sec', 'federal reserve', 'etf', 'ftc', 'nasa', \
                            'noaa', 'fcc', 'faa', 'epa', 'usda', 'fbi', 'cia', 'nsa', \
                            'dhs', 'doj', co_name.lower(), source_ticker.lower()]
    
    for e in remove_keywords_lst:
        otl = list(filter(lambda a: a.lower() != e, otl)) 

    # Remove co_name reworded 
    for i in range(len(otl)):
        co_name_elems = [e.lower() for e in co_name.split() if len(e) > 3]
        for e in co_name_elems: 
            if e in otl[i].lower():
                otl.remove(otl[i])
                break
            
    # Remove empty lists
    if len(otl) == 0: return None
    return otl

def llm_get_summary_metadata(article_idx, llm_name, assets_df, articles_df):
    summary = articles_df.loc[article_idx, 'summary']
    ticker = articles_df.loc[article_idx, 'source_ticker']
    # define the response schemas
    response_schemas = [
        ResponseSchema(
        name="tickers_list",
        description=f"'None' or a list of other company tickers (names if not publicly listed) \
        implicated in news. Exclude {ticker} from list.",
        type="list"
        ),
        ResponseSchema(
        name="news_type",
        description="Pick most appropriate category from \
        ['product', 'services', 'regulatory', 'legal', 'acquisition (of a company)', \
        'merger', 'partnership', 'government contract', 'labor', 'consumer', 'stock buyback', \
        'dividend', 'capital expenditure' , 'macroeconomic',  'management', \
        'earnings', 'financial', 'market analysis', 'opinion', 'other']",
        type="string"
        )]
  
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    template='''
    Identify tickers_list and news_type from the following news. \
    {format_instructions}
    Here is the news: {summary}  
    '''

    prompt = PromptTemplate(
        template=template,
        input_variables=[summary],
        partial_variables={"format_instructions": format_instructions}
    )

    chat = ChatAnthropic(model=llm_name, temperature=0)

    _input = prompt.format_prompt(summary=summary)
    output = chat.invoke(_input.to_messages()).content
    try:
        dict_out = output_parser.parse(output)
        tickers_list = dict_out['tickers_list']
        tickers_list = clean_other_ticker_lst(tickers_list, ticker, assets_df)
        news_type = dict_out['news_type']
        try: tickers_dict = str({str(t):{'score': None} for t in tickers_list})
        except: tickers_dict = None
    except:
        tickers_list = None
        news_type = None
        tickers_dict = None

    articles_df.loc[article_idx, 'other_ticker'] = tickers_dict
    articles_df.loc[article_idx, 'news_type'] = news_type
    if news_type in ['earnings', 'financial', 'market analysis', 'opinion', 'other']:
        articles_df = log_outcome(article_idx, articles_df, "news_type not relevant")
    return news_type, articles_df   

def create_post_text(asset_name, summary, llm_name):
    # define the response schemas
    response_schemas = [ResponseSchema(
        name="caption",
        description="A simple sentence for a social media post about a news story. Make the caption engaging and informative. Do not use hashtags.", 
        type="str"
    )]


    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    template='''
    Write a brief caption that highlights the main story of a news summary about {asset_name} in the style of a tweet. Exclude any part mentioning the asset's market performance. \
    {format_instructions}
    Here's the news summary: {summary}  
    '''

    prompt = PromptTemplate(
        template=template,
        input_variables=["asset_name", "summary"],
        partial_variables={"format_instructions": format_instructions}
        )
  
    chat = ChatAnthropic(model=llm_name, temperature=0)

    _input = prompt.format_prompt(asset_name=asset_name, summary=summary)
    output = chat.invoke(_input.to_messages()).content

    dict_out = output_parser.parse(output)
    caption_out = dict_out['caption']
    return caption_out


# ---------------------------------- For Posts ----------------------------------

def get_company_wiki_link(company_name, ticker, llm_name):

    response_schemas = [
        ResponseSchema(
            name="wiki_link",
            description="The Wikipedia link to the company's page, including the ticker if available.",
            type = "string" 

        )
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    
    # build the prompt
    template="""
    Provide a Wikipedia link to the company (ticker if available) provided.
    """ 
    
    template+="""
    {format_instructions}
    company_name = {company_name}
    ticker = {ticker}

    Do not include comments in the output JSON string.
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=['company_name', 'ticker'],
        partial_variables={"format_instructions": format_instructions} 
    )
    _input = prompt.format_prompt(company_name=company_name, ticker=ticker) 
    chat = ChatAnthropic(model=llm_name, temperature=0)
    output = chat.invoke(_input.to_messages()).content
    
    dict_out = output_parser.parse(output)
    
    # extract data from dict_out
    wiki_link = dict_out['wiki_link']

    return wiki_link

def extract_topics_to_expand_on(articles_df, article_idx, llm_name=smart_llm):
    news = articles_df.iloc[article_idx]['summary']
    assert type(news) == str
    response_schemas = [
        ResponseSchema(
            name="urls",
            description="List of URLs that privide more information about a given specific topic. Stick to wikipedia only",
            type = "list" 
        ),
        ResponseSchema(
            name="topics",
            description="List of topics corresponding to the URLs for further reading.",
            type = "list" 
        ),
        ResponseSchema(
            name="familiarity_scores",
            description="List of percentages (integers from 0 to 100, in increments of 10) representing how likely an avarage investor would know about each topic",
            type = "list" 
        ),
        ResponseSchema(
            name="specificity",
            description="List of json bools indicating if the topic is specific and mentioned in the news article (true) or general (false)",
            type = "list" 
        )
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    
    # build the prompt
    template="""
    Extract up to two urls regarding specific topics explicitly mensioned in the news that would help the reader understand the news better. 
    """ 
    
    template+="""
    {format_instructions}
    news = {news}

    Do not include comments in the output JSON string.
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=['news'],
        partial_variables={"format_instructions": format_instructions} 
    )
    _input = prompt.format_prompt(news=news) 
    chat = ChatAnthropic(model=llm_name, temperature=0)
    output = chat.invoke(_input.to_messages()).content
    
    try: # FIXME
        dict_out = output_parser.parse(output)
    except:
        dict_out = {}
    return dict_out

def create_supplementary_note(topic, wiki_body, summary, llm_name):
    news = summary

    response_schemas = [
        ResponseSchema(
            name="supplementary_note",
            description=f"A brief (one/two sentence) note explaining {topic} for an informed reader",
            type = "string" 

        ),
        ResponseSchema(
            name="is_helpful",
            description=f"A boolean indicating whether the note is about {topic} and does not repeat parts of the news -- (True).",
            type = "bool" 

        )
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    
    # build the prompt
    template="""
    Generate a brief (one/two sentence) note explaining the 'topic' using information from a 'wiki_body'. 
    If possible, give financial details or relevant context from wiki_body that most help assess the impact of the provided news on the underlying company.
    """ 

    template+="""
    {format_instructions}
    topic = {topic}
    news = {news}
    wiki_body = {wiki_body}

    Do not include comments in the output JSON string.
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=['topic', 'wiki_body', 'news'],
        partial_variables={"format_instructions": format_instructions} 
    )
    _input = prompt.format_prompt(topic=topic, wiki_body=wiki_body, news=news) 
    chat = ChatAnthropic(model=llm_name, temperature=0)
    output = chat.invoke(_input.to_messages()).content
    
    dict_out = output_parser.parse(output)
    
    # extract data from dict_out
    supplementary_note = dict_out['supplementary_note']
    is_helpful = dict_out['is_helpful']

    return supplementary_note, is_helpful