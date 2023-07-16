import re
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from langdetect import detect

nlp_en = spacy.load('en_core_web_sm')

# Carregar modelo em português
nlp_pt = spacy.load('pt_core_news_sm')

class TextPreprocessor:
    def __init__(self):
        self.tokenizer = word_tokenize
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = SnowballStemmer('english')
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        # Detectar o idioma do texto
        language = self.detectandoLinguagem(text)

        # Tokenização
        tokens = self.tokenizer(text, language=language)

        # Remoção de caracteres especiais
        tokens = [self.removendoCaracteres(token) for token in tokens]

        # Remoção de stopwords
        tokens = [token for token in tokens if token.lower() not in self.stop_words]

        # Lematização (apenas para inglês)
        if language == 'english':
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

        # Stemming (apenas para português)
        if language == 'portuguese':
            stemmer = SnowballStemmer('portuguese')
            tokens = [stemmer.stem(token) for token in tokens]

        tokens = [token for token in tokens if token]  # Remove tokens vazios

        # Extração de entidades
        entities = self.extract_entities(text, language)

        return {
            'tokens': tokens,
            'entities': entities
        }

    def detectandoLinguagem(self, text):
        language_code = detect(text)
        if language_code == 'pt':
            return 'portuguese'
        elif language_code == 'en':
            return 'english'
        else:
            return ''

    def removendoCaracteres(self, token):
        # Utilize uma expressão regular para remover caracteres especiais
        pattern = r'[^a-zA-Z0-9\s]'  # Mantém apenas letras, números e espaços em branco
        token = re.sub(pattern, '', token)
        return token

    def extract_entities(self, text, language):
        if language == 'english':
            doc = nlp_en(text)
        elif language == 'portuguese':
            doc = nlp_pt(text)
        else:
            return []

        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_
            })

        return entities

teste = TextPreprocessor()

print(teste.preprocess_text("testando imagem"))