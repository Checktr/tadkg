import asyncio
import os.path
import re
import sqlite3
from datetime import datetime
from multiprocessing import freeze_support

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, LongformerForSequenceClassification, \
    LongformerTokenizer, pipeline

import src.features.data_cleaning
from src.features import link_grabber
from src.models.predict.calculate_confidence_score import calculate_sentence_confidence_score, \
    calculate_abstracted_confidence_score

from src import config
from src.models.helper.dataset import Dataset

database_name = os.path.join(os.path.dirname(__file__), '..', '..', '..',
                             config['paths']['databases']['live_prediction'])

MAIN_RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '..', config['paths']['entities'])


def generate_tokenizer(model_name: str):
    """
    :param model_name: name of model to generate tokenizer for
    """
    if model_name == "bert-base-uncased":
        return BertTokenizer.from_pretrained(model_name)
    elif model_name == "allenai/longformer-base-4096":
        return LongformerTokenizer.from_pretrained(model_name)
    else:
        print(f"The model: {model_name} is not currently implemented.")
        exit(-1)


def generate_model(model_name: str, model_path: str):
    """
    :param model_name: name of model to generate model for
    :param model_path: path of model to use configurations of in generation
    """
    if model_name == "bert-base-uncased":
        return BertForSequenceClassification.from_pretrained(model_path, num_labels=2)
    elif model_name == "allenai/longformer-base-4096":
        return LongformerForSequenceClassification.from_pretrained(model_path, num_labels=2)
    else:
        print(f"The model: {model_name} is not currently implemented.")
        exit(-1)


def apply_data(filename=None, names=None):
    """
    Generate Database Tables and Run Asynchronous Predictions

    :param filename: csv file name
    :param names: list of entity names
    :return: None
    """
    conn = sqlite3.connect(database_name)

    sentence_table_name = 'Processed_Sentences'
    webpage_rank_table_name = 'Webpage_Confidence_Rankings'
    entity_rank_table_name = 'Entity_Confidence_Rankings'
    processed_websites_table_name = 'Processed_Websites'

    conn.execute(f'create table IF NOT EXISTS {sentence_table_name}'
                 f'('
                 f'text                  VARCHAR,'
                 f'isMergerOrAcquisition VARCHAR,'
                 f'isMerger              VARCHAR,'
                 f'isAcquisition         VARCHAR,'
                 f'website               VARCHAR,'
                 f'queriedEntity        VARCHAR,'
                 f'involvedEntity       VARCHAR,'
                 f'confidenceScore       VARCHAR'
                 f');')

    conn.execute(f'create table IF NOT EXISTS {webpage_rank_table_name}'
                 f'('
                 f'entity               VARCHAR,'
                 f'website               VARCHAR,'
                 f'confidenceScore       VARCHAR'
                 f');')

    conn.execute(f'create table IF NOT EXISTS {entity_rank_table_name}'
                 f'('
                 f'entity               VARCHAR PRIMARY KEY,'
                 f'confidenceScore       VARCHAR'
                 f');')

    conn.execute(f'create table IF NOT EXISTS {processed_websites_table_name}'
                 f'('
                 f'queriedEntity              VARCHAR,'
                 f'website              VARCHAR,'
                 f'date_processed        VARCHAR'
                 f');')

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("Script is using device: " + device)

    sentence_model_name = config['classifier']['isMergerOrAcquisition']['model_name']
    sentence_tokenizer = generate_tokenizer(sentence_model_name)
    page_model_name = config['classifier']['page_level']['model_name']
    page_tokenizer = generate_tokenizer(page_model_name)
    # Load trained models
    page_model_path = os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                   config['classifier']['page_level']['model_path'])
    page_model = generate_model(page_model_name, page_model_path)
    general_sentence_model_path = os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                               config['classifier']['isMergerOrAcquisition']['model_path'])
    general_sentence_model = generate_model(sentence_model_name, general_sentence_model_path)
    merger_sentence_model_path = os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                              config['classifier']['isMerger']['model_path'])
    merger_sentence_model = generate_model(sentence_model_name, merger_sentence_model_path)
    acquisition_sentence_model_path = os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                                   config['classifier']['isAcquisition']['model_path'])
    acquisition_sentence_model = generate_model(sentence_model_name, acquisition_sentence_model_path)
    ner = pipeline("ner", grouped_entities=True)

    search_keywords = config['collection']['templates']
    if filename is not None:
        data_iter = link_grabber.grab_csv_links(filename, search_keywords)
    else:
        df = pd.DataFrame({'Mfr': names})
        data_iter = link_grabber.grab_links(df, search_keywords)

    for link_data in data_iter:
        asyncio.run(
            generate_results_db(link_data["links"], link_data["entity"], sentence_table_name, webpage_rank_table_name,
                                entity_rank_table_name, processed_websites_table_name, conn, page_tokenizer,
                                page_model, sentence_tokenizer, general_sentence_model, merger_sentence_model,
                                acquisition_sentence_model, device, ner))


async def generate_results_db(url_list: list, entity_name: str, sentence_table_name: str, webpage_rank_table_name: str,
                              entity_rank_table_name: str, processed_webpages_table_name: str, connection,
                              page_tokenizer, page_model, sentence_tokenizer, general_sentence_model,
                              merger_sentence_model, acquisition_sentence_model, device: str, ner):
    """
    Scrapes data off of the internet given a set of urls, make predictions, and extract insights asynchronously

    :param url_list: list of urls
    :param entity_name: name of queried entity
    :param sentence_table_name: name of sentence table
    :param webpage_rank_table_name: name of webpage rank table
    :param entity_rank_table_name: name of entity rank table
    :param processed_webpages_table_name: name of processed webpages table
    :param connection: database connection
    :param page_tokenizer:page level tokenizer
    :param page_model: page level model
    :param sentence_tokenizer: sentence level tokenizer
    :param general_sentence_model: isMergerOrAcquisition sentence level model
    :param merger_sentence_model: isMerger sentence level model
    :param acquisition_sentence_model: isAcquisition sentence level model
    :param device: device to run models on: 'cpu' or 'conda:0'
    :param ner: named entity recognition model
    :return: None
    """
    global sanitized_data_df

    query = connection.execute(
        f"SELECT website From {processed_webpages_table_name} WHERE queriedEntity  = '{entity_name}'")
    cols = [column[0] for column in query.description]
    processed_url_list = list(pd.DataFrame.from_records(data=query.fetchall(), columns=cols)['website'])

    # remove all processed urls from the list of urls to process
    url_list = [x for x in url_list if x not in processed_url_list]

    # Add all new pages to pages which have been processed
    processed_webpages = pd.DataFrame({})
    processed_webpages['website'] = url_list
    processed_webpages['queriedEntity'] = entity_name
    processed_webpages['date_processed'] = datetime.today().strftime('%Y-%m-%d')

    processed_webpages.to_sql(processed_webpages_table_name, connection, if_exists="append", index=False)

    page_sanitized = src.features.data_cleaning.clean_from_urls(url_list, False, entity_name)
    page_text = list(page_sanitized["text"])

    if page_text:

        page_tokenized = page_tokenizer(page_text, padding=True, truncation=True, max_length=3072)

        # Create torch dataset
        page_dataset = Dataset(page_tokenized)
        page_model = page_model.to(device)
        page_trainer = Trainer(page_model)

        # Make prediction
        page_loader = DataLoader(page_dataset, batch_size=1, shuffle=False)
        raw_pred, _, _ = page_trainer.prediction_loop(page_loader, description="prediction")

        y_pred = np.argmax(raw_pred, axis=1)

        # Drop that column
        page_sanitized.drop('isMergerOrAcquisition', axis=1, inplace=True)
        page_sanitized['isMergerOrAcquisition'] = y_pred

        page_sanitized = page_sanitized.loc[page_sanitized['isMergerOrAcquisition'] == 1]

        sentences_sanitized = pd.DataFrame({})

        if len(page_sanitized) != 0:

            for index, row in page_sanitized.iterrows():
                raw_text_list = re.split("\. |(?:\n+ ?)+", row.get('text').replace("\t", "").replace("\r", ""), )
                stripped_text_list = [st.strip() for st in raw_text_list]
                df = pd.DataFrame(stripped_text_list, columns=["text"])
                df['website'] = row.get('website')
                df['entity'] = row.get('entity')
                sentences_sanitized = pd.concat([sentences_sanitized, df])

            sentence_text = list(sentences_sanitized["text"])

            sentence_tokenized = sentence_tokenizer(sentence_text, padding=True, truncation=True, max_length=512)

            # Create torch dataset
            sentence_dataset = Dataset(sentence_tokenized)
            general_sentence_model = general_sentence_model.to(device)
            sentence_trainer = Trainer(general_sentence_model)

            # Make prediction
            sentence_loader = DataLoader(sentence_dataset, batch_size=4, shuffle=False)
            raw_pred, _, _ = sentence_trainer.prediction_loop(sentence_loader, description="prediction")

            # Preprocess raw predictions
            y_pred = np.argmax(raw_pred, axis=1)

            sentences_sanitized['isMergerOrAcquisition'] = y_pred
            sentences_sanitized = sentences_sanitized.loc[sentences_sanitized['isMergerOrAcquisition'] == 1]
            sentence_text = list(sentences_sanitized["text"])
            sentences_sanitized['involvedEntity'] = None

            if sentence_text:

                entity_rec = ner(sentence_text)

                for x in range(len(sentence_text)):
                    entity_df = pd.DataFrame(entity_rec[x])
                    if not entity_df.equals(pd.DataFrame({})):
                        entity_df = entity_df.loc[entity_df['entity_group'] == 'ORG']
                        entity_df = entity_df.loc[entity_df['score'] > 0.95]
                        sentences_sanitized.loc[
                            sentences_sanitized['text'] == sentence_text[x], 'involvedEntity'] = pd.Series(
                            [list(entity_df.word.unique())] * len(sentences_sanitized))

                sentence_tokenized = sentence_tokenizer(sentence_text, padding=True, truncation=True, max_length=512)

                # Create torch dataset
                sentence_dataset = Dataset(sentence_tokenized)
                merger_sentence_model = merger_sentence_model.to(device)
                merger_sentence_trainer = Trainer(merger_sentence_model)

                # Make prediction
                merger_sentence_loader = DataLoader(sentence_dataset, batch_size=4, shuffle=False)
                raw_pred, _, _ = merger_sentence_trainer.prediction_loop(merger_sentence_loader,
                                                                         description="prediction")

                # Preprocess raw predictions
                y_pred = np.argmax(raw_pred, axis=1)

                sentences_sanitized['isMerger'] = y_pred

                acquisition_sentence_model = acquisition_sentence_model.to(device)
                acquisition_sentence_trainer = Trainer(acquisition_sentence_model)

                # Make prediction
                acquisition_sentence_loader = DataLoader(sentence_dataset, batch_size=4, shuffle=False)
                raw_pred, _, _ = acquisition_sentence_trainer.prediction_loop(acquisition_sentence_loader,
                                                                              description="prediction")

                # Preprocess raw predictions
                y_pred = np.argmax(raw_pred, axis=1)

                sentences_sanitized['isAcquisition'] = y_pred

                sentences_sanitized = sentences_sanitized.rename(columns={"entity": "queriedEntity"})

                sentences_sanitized.loc[sentences_sanitized['involvedEntity'].isnull(), ['involvedEntity']] = \
                    sentences_sanitized.loc[sentences_sanitized['involvedEntity'].isnull(), 'involvedEntity'].apply(
                        lambda z: ["{}"])

                sentences_sanitized['confidenceScore'] = sentences_sanitized.apply(
                    lambda z: calculate_sentence_confidence_score(z), axis=1)

                sentences_sanitized = pd.DataFrame(
                    {col: np.repeat(sentences_sanitized[col].values, sentences_sanitized['involvedEntity'].str.len())
                     for col in sentences_sanitized.columns.difference(['involvedEntity'])}).assign(
                    **{'involvedEntity': np.concatenate(sentences_sanitized['involvedEntity'].values)})[
                    sentences_sanitized.columns.tolist()]

                sentences_sanitized['involvedEntity'] = sentences_sanitized['involvedEntity'].replace('{}', None)

                sentences_sanitized.to_sql(sentence_table_name, connection, if_exists="append", index=False)

                # Generate webpage confidence rankings
                webpage_rankings = pd.DataFrame({})
                for entity in sentences_sanitized['queriedEntity'].unique():
                    subset_sentences = sentences_sanitized.loc[
                        sentences_sanitized['queriedEntity'] == entity
                        ]
                    for website in subset_sentences['website'].unique():
                        subset_sentences = sentences_sanitized.loc[
                            sentences_sanitized['website'] == website
                            ]

                        d = {
                            "confidenceScore": calculate_abstracted_confidence_score(subset_sentences),
                            "website": website,
                            "entity": entity
                        }

                        webpage_rankings = pd.concat(
                            (
                                webpage_rankings,
                                pd.DataFrame(
                                    [d],
                                    columns=d.keys()
                                )
                            ),
                            ignore_index=True,
                        )

                webpage_rankings.to_sql(webpage_rank_table_name, connection, if_exists="append", index=False)

                for entity in webpage_rankings['entity'].unique():
                    query = connection.execute(
                        f"SELECT * From {webpage_rank_table_name} WHERE entity= '{entity}'")
                    cols = [column[0] for column in query.description]
                    dataset = pd.DataFrame.from_records(data=query.fetchall(), columns=cols)
                    dataset['confidenceScore'] = pd.to_numeric(dataset['confidenceScore'])
                    score = calculate_abstracted_confidence_score(dataset)
                    connection.execute(f'INSERT INTO {entity_rank_table_name} (entity, confidenceScore) '
                                       f'VALUES(\'{entity}\', {score}) '
                                       f'ON CONFLICT(entity) DO UPDATE SET confidenceScore={score} ')


if __name__ == '__main__':
    freeze_support()
    apply_data(filename=MAIN_RAW_DATA_PATH)
