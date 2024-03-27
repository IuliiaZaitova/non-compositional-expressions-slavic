import subprocess
import pandas as pd
from transformers import T5ForConditionalGeneration, AutoTokenizer
from collections import defaultdict, Counter
from scipy.spatial import distance
import pickle 

def get_phonetic_transcription(wordlist, lang, l2):
    wordlist = [w for w in wordlist if type(w) == str]    
    transcriptions = dict()
    # Load pre-trained model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained('charsiu/g2p_multilingual_byT5_tiny_16_layers_100')
    tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')

    # Language mapping
    lang_dict = {'UK': '<ukr>: ', 'BE': '<bel>: ', 'CS': '<cze>: ', 'PL': '<pol>: ', 'BG': '<bul>: ', 'RU': '<rus>: ', 'NL': '<dut>',
    'DE': '<ger>'}
    
    # Initialize a dictionary to store phonetic transcriptions
    phones_dict = dict()
    
    # Function to run gruut_ipa and capture output
    def run_gruut_ipa(command):
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        output, _ = process.communicate()
        return output.decode('utf-8').strip()
    

    lang_exprs = [lang_dict[lang] + str(x).lower() for x in wordlist]
    
    # Tokenize and generate transcriptions for the language expressions
    out_lang = tokenizer(lang_exprs, padding=True, add_special_tokens=False, return_tensors='pt')
    preds_lang = model.generate(**out_lang, num_beams=1, max_length=50)
    phones_lang = tokenizer.batch_decode(preds_lang.tolist(), skip_special_tokens=True)
    
    
    # Split transcriptions into phonemes
    phones_split = list()
    for t in range(len(phones_lang)):
        if lang in ['CS', 'PL']:
            command_lang = f'python3 -m gruut_ipa phonemes cs-cz {phones_lang[t]}'
        elif lang  == 'DE':
            command_lang = f'python3 -m gruut_ipa phonemes de-de {phones_lang[t]}'
        elif lang  == 'NL':
            command_lang = f'python3 -m gruut_ipa phonemes nl {phones_lang[t]}'
        else:
            command_lang = f'python3 -m gruut_ipa phonemes ru-ru {phones_lang[t]}'

            
        w_lang = run_gruut_ipa(command_lang)            
        phones_split.append((w_lang))
    for w in range(len(wordlist)):
        transcriptions[wordlist[w]] = phones_split[w]
    df = pd.DataFrame(list(transcriptions.items()), columns=['Word', 'Transcription'])
    
    # Save the DataFrame to a CSV file
    df.to_csv(f'data/results/{lang.lower()}_{l2}_transcriptions_responses', index=False)
    return transcriptions

def load_transcription_dict(file_path):
    try:
        # Load the CSV file into a DataFrame, skipping the first row
        transcription_df = pd.read_csv(file_path, header=None, names=['key', 'value'], skiprows=1)

        # Convert the DataFrame to a dictionary
        transcription_dict = dict(zip(transcription_df['key'], transcription_df['value']))
        
        return transcription_dict

    except Exception as e:
        print(f"Error loading transcription dictionary: {str(e)}")
        return None

def get_pwld(ipa_dict1, ipa_dict2):
    # Load a dict phone --> feature vector
    with open('data/phoible/feature_dict.p', "rb") as input_file:
        phone2vec = pickle.load(input_file)
    
    # Define a dictionary of phoneme replacements
    replacement_dict = {
        'd͡z': 'dʑ',
        't͡ɕː': 'tɕ',
        'õ': 'o',
        'd͡ʐ': 'd̺̥z̺̥',
        'd͡ʑ': 'dʑ',
        'd͡ʒ': 'd̥ʒ̥',
        'g': 'ɡ̙',
        'j͡a': 'j a',
        'm˦˥': '',
        't͡sː': 't̪̻s̪̻',
        't̻͡͡s': 't̪̻s̪̻',
        't͡s': 't̪̻s̪̻',
        't͡ɕ': 'tɕ',
        't͡ʂ': 'ʈ̻ʂ̻',
        't͡ʃ': 't̺ʃ̺',
        'ʈ͡͡ʂ': 'ʈ̻ʂ̻',
        'ʲ': 'j',
        'ʲː': 'jː',
        '⁽': '',
        '⁾': '',
        'ʐː': 'ʐ',
        't͡': 't',
        'ă': 'a', 
        'ɨ̯': 'ɨ',
        'nan': '',
        'ɨ˩˦': 'ɨ',
        'n˧˥': 'n',
        't͡z': 't z',
        'ʰ': 'h',
        'u˥': 'u',
        'ʧ': 't̺ʃ̺',
        'o˧˥': 'o',
        'ɨˀ': 'ɨ',
        'O': 'o',
        'a˦˥': 'a',
        'j˧': 'j',
        '⁾ː': '',
        'n˧˧': 'n'
    }
    
    dict_list = [ipa_dict1, ipa_dict2]
    
    for d in range(len(dict_list)):
        ipa_dict = dict_list[d]
        not_in_phoible = set()
        
        for w, ipa in ipa_dict.items():
            substituted_w_lang = []
            
            for ph in ipa.split():
                if ph not in phone2vec:
                    not_in_phoible.add(ph)
                    if ph in replacement_dict:
                        ph = replacement_dict[ph]
                        substituted_w_lang.append(ph)
                    else:
                        print(f'Oops, this ph is not present: {ph}')
                else:
                    substituted_w_lang.append(ph)
            
            substituted_w_lang = ' '.join(substituted_w_lang)
            
            # Update the entry with the substituted_w_lang
            ipa_dict[w] = substituted_w_lang
        
        dict_list[d] = ipa_dict
    
    # Create a set to store all possible phonemes
    all_phonemes = set()
    
    # Iterate through the dictionary values and extract all possible phonemes
    for item in list(dict_list[0].values()) + list(dict_list[1].values()):
        # Combine the phoneme lists and flatten them
        for ph in item.split():
            all_phonemes.add(ph)
    
    phone2dist = defaultdict(lambda: defaultdict(float))
    
    # Calculate phoneme distances using Hamming distance
    for p1 in set(all_phonemes):
        for p2 in set(all_phonemes):
            dist = distance.hamming(phone2vec[p1], phone2vec[p2])
            phone2dist[p1][p2] = dist
    
    def PWLD(s, t):
        s = s.split()
        t = t.split()
        
        # From Wikipedia article; Iterative with two matrix rows.
        if s == t:
            return 0.0
        elif len(s) == 0:
            return len(t) / len(s)
        elif len(t) == 0:
            return len(s) / len(t)
        
        len_s, len_t = len(s), len(t)
        v0 = [None] * (len_t + 1)
        v1 = [None] * (len_t + 1)
        
        for i in range(len(v0)):
            v0[i] = i
        
        for i in range(len_s):
            v1[0] = (i + 0.5) / len_s
            for j in range(len_t):
                cost = 0 if s[i] == t[j] else phone2dist[s[i]][t[j]]
                v1[j + 1] = min(v1[j] + 0.5 / len_s, v0[j + 1] + 0.5 / len_s, v0[j] + cost / len_s)
            
            for j in range(len(v0)):
                v0[j] = v1[j]
        
        return v1[len_t]
    
    pwld_dict = dict()
    
    # Calculate Pairwise Levenshtein Distances (PWLD)
    for item1, item2 in zip(list(dict_list[0].keys()), list(dict_list[1].keys())):
        pwld_dict[(item1, item2)] = PWLD(dict_list[0][item1], dict_list[1][item2])
    
    return pwld_dict