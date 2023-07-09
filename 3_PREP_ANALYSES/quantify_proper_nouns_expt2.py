import pandas as pd
import spacy

def identify_proper_nouns(csv_file):
    # Download the spaCy English model if not already downloaded
    try:
        import spacy
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        print("Downloading the spaCy English model...")
        import spacy.cli
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load('en_core_web_sm')

    df = pd.read_csv(csv_file)
    words = df['word_lower'].tolist()
    proper_noun_counts = []

    for word in words:
        doc = nlp(word)
        proper_nouns = [token.text for token in doc if token.pos_ == 'PROPN']
        proper_noun_counts.append(len(proper_nouns))

    df['PROPER_NOUN_COUNT'] = proper_noun_counts

    return df


csv_file_path = 'exp2_data_with_norms_reordered_20221029.csv'

updated_df = identify_proper_nouns(csv_file_path)

# Count the number of proper nouns (if any) in each word
count = 0
for i in range(len(updated_df)):
    if updated_df['PROPER_NOUN_COUNT'][i] > 0:
        count += 1

print(count) # 0.5% of words have at least one proper noun

updated_df['PROPER_NOUN_COUNT'].value_counts()