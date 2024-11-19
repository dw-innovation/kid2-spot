import pandas as pd
import requests
import json
from argparse import ArgumentParser
from tqdm import tqdm
from typing import Dict
from diskcache import Cache
from datageneration.utils import NON_ROMAN_LANG_GROUPS
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

WIKIDATA_API_WIKIPEDIA_SITE_LINKS = 'https://www.wikidata.org/w/api.php?action=wbgetentities&ids=WD_ID&props=sitelinks&format=json'
WIKIDATA_API_WD_REQUEST_ENDPOINT='https://en.wikipedia.org/w/api.php?action=query&prop=pageprops&ppprop=wikibase_item&redirects=1&titles=TITLE&format=json'
cache_wikidata_ids = Cache("wikidata-ids-cache")
cache_wikidata_non_roman_names = Cache("wikidata-non_roman_names-cache")

def requests_retry_session(
    retries=3,
    backoff_factor=0.3,
    status_forcelist=(500, 502, 504),
    session=None,
):
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


@cache_wikidata_ids.memoize()
def request_wd_id(wd_title):
    '''
    Given name of an area, it returns its wikidata id if it has.
    :param wd_title:
    :return:
    '''
    endpoint = WIKIDATA_API_WD_REQUEST_ENDPOINT.replace('TITLE', wd_title)
    response = requests_retry_session().get(endpoint)
    response.raise_for_status()
    wd_id = None
    data = None
    if response.status_code == 200:
        data= response.json()
    else:
        return wd_id

    pages = data.get("query", {}).get("pages", {})

    for page_id, page_info in pages.items():
        page_props = page_info.get("pageprops", {})
        wd_id = page_props.get("wikibase_item")

        if wd_id:
            return wd_id
    return wd_id

@cache_wikidata_non_roman_names.memoize()
def get_dict_of_non_roman_alternatives(wd_id: str) -> Dict:
    endpoint = WIKIDATA_API_WIKIPEDIA_SITE_LINKS.replace('WD_ID', wd_id)
    response = requests_retry_session().get(endpoint)
    response.raise_for_status()
    data = None
    results = {}
    if response.status_code == 200:
        data= response.json()
    else:
        return results

    entities = data.get("entities", {})
    if wd_id in entities:
        sitelinks = entities[wd_id].get("sitelinks", {})
        for key, value in sitelinks.items():
            if key.endswith('wiki'):
                lang = key.replace('wiki', '')
                alt_name = value['title']

                if lang in NON_ROMAN_LANGUAGES:
                    results[lang] = alt_name

    return results

area_names_non_roman_vocab = {}


def extract_non_roman_alternatives(area_name: str):
    '''
    extract wikidata
    :param area_name:
    :return:
    '''
    non_roman_versions = None
    area_wd_id = request_wd_id(area_name)
    if area_wd_id:
        non_roman_versions = get_dict_of_non_roman_alternatives(area_wd_id)
        if len(non_roman_versions) == 0:
            return non_roman_versions
    return non_roman_versions


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--geolocation_file', type=str, default='datageneration/data/countries+states+cities.json')
    parser.add_argument('--output_file', type=str, default='datageneration/data/area_non_roman_vocab.json')

    args = parser.parse_args()

    geolocation_file = args.geolocation_file
    output_file = args.output_file

    geolocation_data = pd.read_json(geolocation_file)
    vocabulary = {}
    for sample in tqdm(geolocation_data.to_dict(orient='records'), total=len(geolocation_data)):
        country_name = sample['name']
        if country_name not in vocabulary:
            non_roman_versions = extract_non_roman_alternatives(country_name)
            if non_roman_versions:
                vocabulary[country_name] = {'non_roman_versions': non_roman_versions}

        for state in sample['states']:
            state_name = state['name']

            if state_name not in vocabulary:
                non_roman_versions = extract_non_roman_alternatives(state_name)
                if non_roman_versions:
                    vocabulary[state_name] = {'non_roman_versions': non_roman_versions}

            for city in state['cities']:
                city_name = city['name']
                if city_name not in vocabulary:
                    non_roman_versions = extract_non_roman_alternatives(city_name)
                    if non_roman_versions:
                        vocabulary[city_name] = {'non_roman_versions': non_roman_versions}


    with open(output_file, 'w') as json_file:
        json.dump(vocabulary, json_file, ensure_ascii=False)