import pandas as pd
import matplotlib.pyplot as plt
import subprocess

# Run the orthographic_equivalencies.py script
subprocess.run(['python3', '/Users/yuliazaitova/Desktop/work/PhD/coling2023/LREC-2022-SynDist-Surprisal/Multilingual_Cloze_Tests/Distances/orthographic_equivalencies.py'])

# Run the utils.py script
subprocess.run(['python3', '/Users/yuliazaitova/Desktop/work/PhD/coling2023/LREC-2022-SynDist-Surprisal/Multilingual_Cloze_Tests/Distances/utils.py'])

def get_was_ortho():
        # Use pandas to read the Excel file and create a dictionary of DataFrames
    dfs = pd.read_excel('data/experiment_data.xlsx', sheet_name=None)
    
    # 'dfs' is now a dictionary where keys are the sheet names, and values are DataFrames
    # You can access each DataFrame by its corresponding sheet name
    
    # Print the sheet names
    
    dfs.pop('stats')
    # Access and print each DataFrame
    for sheet_name, df in dfs.items():
        dfs[sheet_name] = df[['phrase in L2', 'L1 MCU']]
        dfs[sheet_name].columns = sheet_name.split('-')
        dfs[sheet_name] = dfs[sheet_name].dropna()
    # List of language pairs
    language_pairs = ['CS-RU', 'BG-RU', 'UK-RU', 'BE-RU', 'PL-RU']
    results_dict = dict()
    # Define orth_dist and other necessary functions (e.g., levenshtein_distance, character_surprisals, character_entropy, etc.)
    
    for lang_pair in language_pairs:
        df = dfs[lang_pair]  # Access the DataFrame for the current language pair
        l2 = lang_pair.split('-')[0]
        # Compute Levenshtein distance and other metrics
        levensthein_foreign = levenshtein_distance(df, foreign=l2, native='RU', costs=orth_dist)
        levensthein_native = levenshtein_distance(df, foreign='RU', native=l2, costs=orth_dist)
    
        # Check assertions
        assert levensthein_foreign['LD'].all() == levensthein_native['LD'].all()
        assert levensthein_foreign['normalized LD'].all() == levensthein_native['normalized LD'].all()
    
        # Compute character surprisals
        probs_foreign, surprisals_foreign = character_surprisals(levensthein_foreign, foreign=l2, native='RU')
        probs_native, surprisals_native = character_surprisals(levensthein_native, foreign='RU', native=l2)
    
        # Compute character entropy
        char_entropy_foreign = character_entropy(surprisals_foreign, probs_foreign)
        char_entropy_native = character_entropy(surprisals_native, probs_native)
    
        # Compute full conditional entropy
        H_foreign_native = full_conditional_entropy(l2, 'RU', levensthein_foreign, surprisals_native, probs_native)
        H_native_foreign = full_conditional_entropy('RU', l2, levensthein_native, surprisals_foreign, probs_foreign)
    
        # Compute word adaptation surprisal
        was_foreign = word_adaptation_surprisal(levensthein_foreign, surprisals_foreign, probs_foreign)
        was_native = word_adaptation_surprisal(levensthein_native, surprisals_native, probs_native)
        results_dict[lang_pair] = [was_foreign, was_native]
        return results_dict