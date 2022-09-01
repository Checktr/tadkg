import random
import warnings
from pathlib import Path

import pandas as pd
import torch
from sqlalchemy import create_engine
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

from src import config

warnings.filterwarnings("ignore")

sentence_table_name = config['data']['tables']['sentence_level']


def random_state(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


random_state(1234)

model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)


def get_response(input_text, num_return_sequences, num_beams):
    batch = tokenizer([input_text], truncation=True, padding='longest', max_length=60, return_tensors="pt").to(
        torch_device)
    translated = model.generate(**batch, max_length=60, num_beams=num_beams, num_return_sequences=num_return_sequences,
                                temperature=1.5)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text


template_phrases = config['synthetic_data']['template_phrases']

filler = config['synthetic_data']['filler']

terminators = config['synthetic_data']['terminators']

company_name_file = open(config['synthetic_data']['entity_name_file'])
company_names = [line.rstrip() for line in company_name_file]
company_name_file.close()

all_phrases = set()


def process_paraphrase(paraphrase_in: str):
    if 'CMP1' not in paraphrase_in or 'CMP2' not in paraphrase_in:
        return
    cmp1 = random.choice(company_names)
    if random.randint(0, 3) == 0:
        cmp1 += ' ' + random.choice(terminators)
    cmp2 = cmp1
    while cmp2 == cmp1:
        cmp2 = random.choice(company_names)
    if random.randint(0, 3) == 0:
        cmp2 += ' ' + random.choice(terminators)
    processed_paraphrase = paraphrase_in.replace('CMP1', cmp1).replace('CMP2', cmp2)
    all_phrases.add((processed_paraphrase, cmp2))


def generate_phrases():
    for raw_phrase in template_phrases:
        for i in range(5):
            phrase = raw_phrase.replace('<filler>', random.choice(filler))
            phrase = phrase.replace('<decimal>', str(random.randint(0, 100) / 10))
            phrase = phrase.replace('<integer>', str(random.randint(0, 100)))

            # print("-" * 100)
            # print("Input_phrase: ", phrase)
            # print("-" * 100)
            responses = get_response(phrase, 10, 10)
            for paraphrase in responses:
                process_paraphrase(paraphrase)

    list_phrases = list(all_phrases)
    random.shuffle(list_phrases)
    return list_phrases


db_columns = [
    config['synthetic_sentence_schema']['attribute'],
    config['synthetic_sentence_schema']['target'],
    config['synthetic_sentence_schema']['entity']
]

for key in list(config['synthetic_sentence_schema']['defaults'].keys()):
    db_columns.append(key)

phrase_df = pd.DataFrame(
    {},
    columns=db_columns,
)

phrases = generate_phrases()
for phrase in phrases:

    phrase_dict = {
        config['synthetic_sentence_schema']['attribute']: phrase[0],
        config['synthetic_sentence_schema']['target']: 1,
        config['synthetic_sentence_schema']['entity']: phrase[1]
    }

    for key in list(config['synthetic_sentence_schema']['defaults'].keys()):
        value = config['synthetic_sentence_schema']['defaults'][key]
        if value is not None:
            phrase_dict.update({key: value})
        else:
            phrase_dict.update({key: float("NaN")})

    phrase_df = phrase_df.append(phrase_dict, ignore_index=True)

print(phrase_df)

engine = create_engine(
    f'sqlite:///{Path(Path(__file__).parent.parent.parent.parent, config["paths"]["databases"]["test_train"])}',
    echo=True
)
sqlite_connection = engine.connect()
phrase_df.to_sql(
    sentence_table_name, sqlite_connection, if_exists="append", index=False
)
