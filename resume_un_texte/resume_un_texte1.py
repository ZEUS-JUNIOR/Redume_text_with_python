import nltk
import string
from heapq import nlargest

text = """L'objectif principal de l'utilisation de l'apprentissage automatique pour la synthèse de texte est de réduire le texte de référence à une version plus petite tout en conservant ses connaissances à côté de sa signification. Plusieurs descriptions textuelles résumées sont fournies, par exemple, expliquent le rapport en tant que texte généré à partir d'un ou plusieurs documents qui communiquent des connaissances pertinentes dans le premier texte, et qui ne représentent pas plus de la moitié du texte principal et généralement beaucoup plus limité que cela. J'espère que vous savez maintenant ce qu'est la synthèse de texte et pourquoi nous devons utiliser l'apprentissage automatique pour cela. Dans la section ci-dessous, je vais vous présenter un projet d'apprentissage automatique sur la synthèse de texte avec Python.
Nous n'avons pas besoin d'utiliser beaucoup d'apprentissage automatique ici. Nous pouvons facilement résumer du texte sans entraîner de modèle. Mais encore, nous devons utiliser un traitement du langage naturel, pour cela, j'utiliserai la bibliothèque NLTK en Python. Exécutons maintenant quelques étapes pour supprimer les ponctuations du texte, puis nous devons effectuer quelques étapes de traitement de texte, et à la fin, nous allons simplement tokeniser le texte et vous pourrez alors voir les résultats de la synthèse de texte avec python. 
"""
nltk.download('punkt')
if text.count(". ") > 20:
    length = int(round(text.count(".")/10, 0))
else:
    length = 1

nopuch = [char for char in text if char not in string.punctuation]
nopuch = "".join(nopuch)

processed_text = [word for word in nopuch.split() if word.lower() not in nltk.corpus.stopwords.words('french')]

word_freq = {}
for word in processed_text:
    if word not in word_freq:
        word_freq[word] = 1
    else:
        word_freq[word] = word_freq[word] + 1

max_freq = max(word_freq.values())
for word in word_freq.keys():
    word_freq[word] = (word_freq[word]/max_freq)

sent_list = nltk.sent_tokenize(text)
sent_score = {}
for sent in sent_list:
    for word in nltk.word_tokenize(sent.lower()):
        if word in word_freq.keys():
            if sent not in sent_score.keys():
                sent_score[sent] = word_freq[word]
            else:
                sent_score[sent] = sent_score[sent] + word_freq[word]

summary_sents = nlargest(length, sent_score, key=sent_score.get)
summary = " ".join(summary_sents)
print(summary)
