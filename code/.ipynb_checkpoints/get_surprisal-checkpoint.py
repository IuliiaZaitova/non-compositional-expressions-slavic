# Imports
from minicons import scorer
import pandas as pd
from unidecode import unidecode

# Constants
DIACRITIC_LANGUAGES = ['PL', 'CS']

# Models Initialization
model_bert_small = scorer.IncrementalLMScorer("blinoff/roberta-base-russian-v0", 'cpu')
model_gpt_small = scorer.IncrementalLMScorer("ai-forever/rugpt3small_based_on_gpt2", 'cpu')
model_bert_large = scorer.MaskedLMScorer("ai-forever/ruRoberta-large", 'cpu')
model_gpt_large = scorer.IncrementalLMScorer("ai-forever/rugpt3large_based_on_gpt2", 'cpu')

def load_experiment_data(filepath):
    """Load experiment data from Excel file."""
    xls = pd.ExcelFile(filepath)
    return {sheet_name: pd.read_excel(xls, sheet_name=sheet_name) for sheet_name in xls.sheet_names}

from unidecode import unidecode

def calculate_surprisal(sentence, model, diacritics=False):
    if diacritics:
        sentence = unidecode(sentence)
    input = sentence
    
    surprisal_dict = dict()
    
    try:
        surprisal = model.token_score(input, surprisal=True, base_two=False)
    except IndexError:
        print(f"Error with input: {input}")
        return {input: 0.0}  # or some other default
    
    word_number = 0
    prev_word = ''
    prev_surprisal = 0
    sentence_tokens = sentence.split()

    # Filter out unwanted tokens like '<s>' and '</s>'
    filtered_surprisal = [word for word in surprisal[0] if word[0] not in ['<s>', '</s>']]

    for word in filtered_surprisal:
        prev_word += word[0]
        prev_surprisal += word[1]
        
        # Check if word_number exceeds sentence_tokens length
        if word_number >= len(sentence_tokens):
            print(f"Warning: Tokenization mismatch for sentence '{sentence}'")
            break

        if prev_word == sentence_tokens[word_number]:
            surprisal_dict[prev_word] = prev_surprisal
            word_number += 1
            prev_word = ''
            prev_surprisal = 0

    return surprisal_dict


def get_surprisal_for_field(record, field, model, use_diacritics):
    """Compute the surprisal for a specific field and update the record."""
    if isinstance(record[field], str):
        diacritics = use_diacritics and (field in ["phrase l2", "sentence l2"])
        surprisal_key = f'surprisal_{field}'
        record[surprisal_key] = calculate_surprisal(record[field], model, diacritics)

def get_model_df(model, lang, experiment_data, model_name):
    """Generate a DataFrame based on model results for a given language."""
    
    data_records = []
    
    for _, row in experiment_data[f'{lang}-RU'].iterrows():
        record = {}
        
        # List of fields to check
        fields_to_check = ['phrase l2', 'sentence l2', 'phrase ru', 'sentence ru', 'literal translation', 'phrase ru with literal']
        use_diacritics = lang in DIACRITIC_LANGUAGES
        
        # Loop over each field to compute surprisal
        for field in fields_to_check:
            if field in row and isinstance(row[field], str):
                record[field] = row[field]
                get_surprisal_for_field(record, field, model, use_diacritics)

        data_records.append(record)
    
    df = pd.DataFrame(data_records)
    csv_filename = f"data/metrics/surprisal/{lang}_{model_name}_data.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Data saved to {csv_filename}")

    return df
    