import re
import unicodedata
import langid
from num2words import num2words
import MeCab
import uroman as ur
from Levenshtein import ratio, distance

# Initialize models
mecab = MeCab.Tagger()
uroman = ur.Uroman()

# ========== Text Normalization Functions ==========
def num_to_words(m, lang):
    s = m.group(0)
    return num2words(s, lang=lang)

def normalize_uroman(text):
    text = text.lower()
    text = text.replace("'", "'")
    text = re.sub("([^a-z' ])", " ", text)
    text = re.sub(' +', ' ', text)
    return text.strip()

def normalize(text, langs: list[str] = ['en', 'ja', 'ko']):
    # TVer対応
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\(.+?\)', '', text)
    text = re.sub(r'<.+?>', '', text) # Voice Tag

    # Unicode正規化
    text = unicodedata.normalize('NFKC', text)

    langid.set_languages(langs)
    lang, _ = langid.classify(text)
    text = re.sub(r'([0-9]+)', lambda wrapper: num_to_words(wrapper, lang), text)

    if lang == 'ja':
        m = mecab.parse(text).strip()
        kana = ' '.join([line.split('\t')[1] for line in m.split('\n')[:-1]])
        roman = uroman.romanize_string(kana)
    else:
        roman = uroman.romanize_string(text)

    return normalize_uroman(roman)

def find_similarity_and_distance(text1, text2):
    # Levenshtein類似度を計算
    _text1 = ''.join(normalize(text1))
    _text2 = ''.join(normalize(text2))
    
    return ratio(_text1, _text2), distance(_text1, _text2)
    
if __name__ == "__main__":
    
    text_ref = "これは品評会でよくあるただの牛のけんかです。"
    texts = [
        "これは品評会でよくあるタナム虫の喧嘩です",
        "これは、貧品界でよくあるタナの牛の喧嘩です。",
        "それは、貧辺界でよくあるカナの牛の喧嘩です。",
        "これは、貧返界でよくあるタナの節の喧嘩です。",
        "それは貧品界でよくあるタナの牛の喧嘩です"
    ]
    
    print(f"[{text_ref}] [{normalize(text_ref)}]")
    
    for text in texts:
        r, d = find_similarity_and_distance(text_ref, text)
        print(f"r:{r:.3f} d:{d} [{text}] [{normalize(text)}]")