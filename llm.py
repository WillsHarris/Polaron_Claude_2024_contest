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
from util import *

# Get the directory that this script is in
dir_path = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(dir_path, "config.json"), 'r') as file:
    config = json.load(file)

smart_llm = config['smart_llm']
dumb_llm = config['dumb_llm']

load_dotenv() # load environment variables

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

#----------------------------------ChatGPT-----------------------------------------

def llm_score_summary(article_idx, max_score, llm_name, assets_df, articles_df, prompt_version='1.2.0'):
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

    # Select prompt version
    # --- v1.1.1 ---
    if prompt_version == '1.1.1':
        if multi_stock_mode is False:
            desc = ( # v1.1.1 score description
                "Integer within range from -3 to 3 inclusive. "
                "A lower (higher) score indicates recommendation to sell (buy) "
                "equity in the underlying asset. Respond with 0 if the news is unrelated "
                "or has no significant impact on {asset_name}."
                )
            
            prompt = ( # v1.1.1 prompt
                "Determine a score for the investment potential of asset {asset_name} "
                "(ticker symbols {ticker}) based on the provided news."
                )
        
        if multi_stock_mode is True:
            desc = ( #v1.1.1 multistock description
                "List of integers with values ranging from -3 to 3 inclusive. "
                "A lower (higher) score indicates recommendation to sell (buy) equity in the underlying asset. "
                "Order scores to match the provided asset list, {asset_names}. List elements should be 0 if the news "
                "is unrelated or has no significant impact on that asset."
                )
            
            prompt =( #v1.1.1 multistock prompt
                "Determine scores for the investment potential of assets {asset_names} "
                "(ticker symbols {asset_tickers}) based on the provided news."
                )
    elif prompt_version == '1.2.0':
        if multi_stock_mode is False:
            desc = (  # v1.2.0 score description
                "Integer within range from -3 to 3 inclusive. -3: extremely bad news, "
                "very rare. -2: very bad news, rare. -1: bad news. 0: neutral or irrelevant news, esp. involving "
                "opinions, should be extremely common. 1: good news, 2: very good news, rare. 3: "
                "extremely good news, very rare."
                )

            prompt = (  # v1.2.0 score prompt
                "Based on the following summary of a news article, determine how good or bad the news is for "
                "{asset_name} for stock: {ticker} on an exponential scale. Score 0 for anything (ie opinions) "
                "that is not strictly a reported event. Assume 0 until confident otherwise. If not 0 assume "
                "+/- 1's until confident in +/- 2's, etc."
                )
        
        if multi_stock_mode is True:
            desc = (  # v1.2.0 multistock description
                "List of integers with values ranging from -3 to 3 inclusive. -3: extremely bad news, "
                "very rare. -2: very bad news, rare. -1: bad news. 0: neutral or irrelevant news, esp. involving "
                "opinions, should be extremely common. 1: good news, 2: very good news, rare. 3: "
                "extremely good news, very rare."
                )

            prompt = (  # v1.2.0 multistock prompt
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

    if article_score == 0:
        print("Article score is 0, no trade required.")
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

def clean_summary_keyphrases(summary, summary_keyphrases):
    kps = []
    for kp in summary_keyphrases:
        kp_spit_L = kp.split("' ")
        if len(kp_spit_L) > 1:
            pre_sum = summary.split(kp)[0]
            if " '" in pre_sum:
                continue
        kp_split_R = kp.split(" '")
        if len(kp_split_R) > 1:
            post_sum = summary.split(kp)[-1]
            if "' " in post_sum:
                continue
        kps.append(kp)
    return kps

def llm_get_summary_keyphrases(summary, llm_name):
    # define the response schemas
    response_schemas = [ResponseSchema(
        name="phrases",
        description="A list of the key phrases and no addidtal text", 
        type="list"
    )]


    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    template='''
    Identify and list keywords or key phrases in the text that independently encapsulate the central message.
    limit the number of key phrases to 2. Dont key only part of a quote.\
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

    dict_out = output_parser.parse(output)
    phrases_out = dict_out['phrases']
    phrases_out = clean_summary_keyphrases(summary, phrases_out)

    return phrases_out

def create_social_text(asset_name, summary, llm_name):
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

#----------------------------------Aux LLMs-----------------------------------------

def get_product_aux(article_idx, llm_name, articles_df, storage_path):
    product_news = articles_df.iloc[article_idx]['summary']
    article_id = articles_df.iloc[article_idx]['article_id']

    response_schemas = [
        ResponseSchema(
            name="product_names",
            description="A list of product names mentioned in the product_news",
            type = "list" 
        ),
        ResponseSchema(
            name="product_statuses",
            description="A list of statuses for each product identified, indicating whether each one is 'unreleased' or 'existing'",
            type = "list" 
        ),
        ResponseSchema(
            name="information_availability",
            description="A boolean indicating whether the information about product names and statuses is available or applicable",
            type = "bool" 
        )
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    
    # build the prompt
    template="""
    Identify the name of the product or products in a list that are mentioned in the provided product_news and indicate if it is an unreleased product or an existing one.
    """ 
    
    template+="""
    {format_instructions}
    product_news = {product_news}

    Do not include comments in the output JSON string.
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=['product_news'],
        partial_variables={"format_instructions": format_instructions} 
    )
    _input = prompt.format_prompt(product_news=product_news) 
    chat = ChatAnthropic(model=llm_name, temperature=0)
    output = chat.invoke(_input.to_messages()).content
    
    dict_out = output_parser.parse(output)

    # update the aux_df from dict_out
    col_names = ['product_names', 'product_statuses', 'information_availability']
    for col_name in col_names:
        col_val = dict_out[col_name]
        if type(col_val) == str and col_name != 'information_availability':
            col_val = [col_val]
        if type(col_val) == list:
            col_val = str(col_val)
        add_to_aux('product', article_id, col_name, col_val, storage_path)
    
    return articles_df

def get_acquisition_aux(article_idx, llm_name, articles_df, storage_path):
    acquisition_news = articles_df.iloc[article_idx]['summary']
    response_schemas = [
        ResponseSchema(
            name="acquired_company",
            description="The name of the company that is being acquired.",
            type = "string" 
        ),
        ResponseSchema(
            name="acquiring_company",
            description="The name of the company that is acquiring the other company.",
            type = "string" 
        ),
        ResponseSchema(
            name="acquisition_status",
            description="The current status of the acquisition (completed or under consideration).",
            type = "string" 
        ),
        ResponseSchema(
            name="acquisition_price",
            description="The price at which the acquisition is taking place (in $B), if the information is available.",
            type = "float" 
        ),
        ResponseSchema(
            name="missing_information",
            description="A list of information that is not provided in the acquisition news.",
            type = "list" 
        )
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    
    # build the prompt
    template="""
    Analyze the provided acquisition news to identify the companies involved, the status of the acquisition, and the acquisition price if available.
    """ 
    
    template+="""
    {format_instructions}
    acquisition_news = {acquisition_news}

    Do not include comments in the output JSON string.
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=['acquisition_news'],
        partial_variables={"format_instructions": format_instructions} 
    )
    _input = prompt.format_prompt(acquisition_news=acquisition_news) 
    chat = ChatAnthropic(model=llm_name, temperature=0)
    output = chat.invoke(_input.to_messages()).content
    
    dict_out = output_parser.parse(output)

    # update the aux_df from dict_out
    article_id = articles_df.iloc[article_idx]['article_id']
    col_names = ['acquired_company', 'acquiring_company', 'acquisition_status', 'acquisition_price', 'missing_information']
    for col_name in col_names:
        col_val = dict_out[col_name]
        if type(col_val) == list:
            col_val = str(col_val)
        add_to_aux('acquisition', article_id, col_name, col_val, storage_path)

    return articles_df

def get_regulatory_aux(article_idx, llm_name, articles_df, storage_path):
    regulatory_news = articles_df.iloc[article_idx]['summary']
    response_schemas = [
        ResponseSchema(
            name="regulatory_body",
            description="The name of the regulatory body mentioned in the news (e.g., 'FDA', 'CDC')",
            type = "string" 
        ),
        ResponseSchema(
            name="country_of_regulatory_body",
            description="The country where the regulatory body operates",
            type = "string" 
        ),
        ResponseSchema(
            name="stage_of_regulation",
            description="The current stage of the regulation process (e.g., 'proposed', 'under way', 'complete')",
            type = "string" 
        ),
        ResponseSchema(
            name="impacted_entity",
            description="The specific product or service impacted by the regulation, or 'full corporation' if the entire company is affected",
            type = "string" 
        ),
        ResponseSchema(
            name="fine_amount",
            description="The amount of any fine (in $B) that is mentioned in the news, if available",
            type = "float" 
        ),
        ResponseSchema(
            name="unavailable_data",
            description="A list of the above prompt names that are not available from the regulatory news",
            type = "list" 
        )
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    
    # build the prompt
    template="""
    Analyze the provided regulatory news to extract key information about the regulatory body, country, stage of regulation, fine amount, and any unavailable data.
    """ 
    
    template+="""
    {format_instructions}
    regulatory_news = {regulatory_news}

    Do not include comments in the output JSON string.
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=['regulatory_news'],
        partial_variables={"format_instructions": format_instructions} 
    )
    _input = prompt.format_prompt(regulatory_news=regulatory_news) 
    chat = ChatAnthropic(model=llm_name, temperature=0)
    output = chat.invoke(_input.to_messages()).content
    
    dict_out = output_parser.parse(output)
    
    # update the aux_df from dict_out
    article_id = articles_df.iloc[article_idx]['article_id']
    col_names = ['regulatory_body', 'country_of_regulatory_body', 'stage_of_regulation', 'impacted_entity', 'fine_amount', 'unavailable_data']
    for col_name in col_names:
        col_val = dict_out[col_name]
        if type(col_val) == list:
            col_val = str(col_val)
        add_to_aux('regulatory', article_id, col_name, col_val, storage_path)

    return articles_df


# ---------------------------------- For Social Posts ----------------------------------

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
            description="List of percentages (integers from 0 to 100, in icroments of 20) representing how likely an avarage investor would know about each topic",
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
    If possible, give financial details or relivent context from wiki_body that most help assess the impact of the provided news on the underlying company.
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

def get_reflective_news_summary(news, ticker, llm_name):
    response_schemas = [
        ResponseSchema(
            name="summary",
            description="Brief summary of the news",
            type = "string" 

        )
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    
    # build the prompt
    template="""
    provide a very brief summary of the provided news as it related to the ticker. 
    Write assuming it the news has already happened and is not breaking. Use incomplete sentences to shorten length
    """ 
    
    template+="""
    {format_instructions}
    news = {news}
    news = {ticker}
    Do not include comments in the output JSON string.
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=['news', 'ticker'],
        partial_variables={"format_instructions": format_instructions} 
    )
    _input = prompt.format_prompt(news=news, ticker=ticker) 
    chat = ChatAnthropic(model=llm_name, temperature=0)
    output = chat.invoke(_input.to_messages()).content
    
    dict_out = output_parser.parse(output)
    
    # extract data from dict_out
    summary = dict_out['summary']
    
    return summary