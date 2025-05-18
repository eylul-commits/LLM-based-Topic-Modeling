import os
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env

from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from gensim import corpora, models
from gensim.models import CoherenceModel
from bertopic import BERTopic
from sklearn.metrics import silhouette_score
import numpy as np
from top2vec import Top2Vec
import openai

def preprocess(text):
    text = re.sub(r'\W+', ' ', text.lower())
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in tokens if word not in stop_words and len(word) > 2])

def extract_gpt_topics(docs, num_topics=10):
    load_dotenv()  # Load environment variables from .env
    openai.api_key = os.getenv("OPENAI_API_KEY")
    prompt = (
        f"Given the following documents, extract {num_topics} distinct topics. "
        "For each topic, list 5 representative keywords. "
        "Return the topics as a numbered list, each with its keywords separated by commas.\n\n"
        + "\n\n".join(docs)
    )
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.3,
    )
    topics = []
    for line in response.choices[0].message.content.split('\n'):
        if '.' in line:
            parts = line.split('.', 1)[-1].strip().split(',')
            keywords = [w.strip().lower() for w in parts if w.strip()]
            if keywords:
                # Tokenize each keyword and join with spaces
                tokenized_keywords = []
                for keyword in keywords:
                    tokens = word_tokenize(keyword)
                    tokenized_keywords.extend(tokens)
                topics.append(tokenized_keywords)
    return topics[:num_topics]

def main():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('all')

    categories = [ 'comp.graphics',
    'comp.os.ms-windows.misc',
    'comp.sys.ibm.pc.hardware',
    'comp.sys.mac.hardware',
    'comp.windows.x',
    'misc.forsale',
    'rec.autos',
    'rec.motorcycles',
    'rec.sport.baseball',
    'rec.sport.hockey']


    # Load dataset
    newsgroups = fetch_20newsgroups(categories=categories, remove=('headers', 'footers', 'quotes'))
    texts = newsgroups.data[:5000]

    # Preprocessing
    cleaned_texts = [preprocess(doc) for doc in texts]
    tokenized_texts = [doc.split() for doc in cleaned_texts]

    # Dictionary & Corpus
    dictionary = corpora.Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

     # Train LDA model
    lda_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=10, passes=10)

    for idx, topic in lda_model.print_topics(-1):
         print(f"Topic {idx}: {topic}")

    # # LDA Coherence
    coherence_model_lda = CoherenceModel(model=lda_model, texts=tokenized_texts,
                                          dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print(f"LDA Coherence Score: {coherence_lda:.4f}")

    # BERTopic
    topic_model = BERTopic(language="english", calculate_probabilities=True)
    topics, _ = topic_model.fit_transform(cleaned_texts)

    print(topic_model.get_topic_info())
    topic_model.visualize_topics().show()

    # BERTopic Coherence
    topics_words = topic_model.get_topics()
    bertopic_topics = [
        [word for word, _ in topic_words]
        for topic_id, topic_words in topics_words.items()
        if topic_id != -1 and len(topic_words) > 0
    ]

    coherence_model_bertopic = CoherenceModel(topics=bertopic_topics, texts=tokenized_texts,
                                              dictionary=dictionary, coherence='c_v')
    coherence_bertopic = coherence_model_bertopic.get_coherence()
    print(f"BERTopic Coherence Score: {coherence_bertopic:.4f}")

    # Silhouette Score for BERTopic
    embeddings = topic_model.embedding_model.embed(texts)
    labels = topic_model.topics_

    valid_idx = [i for i, label in enumerate(labels) if label != -1]
    sil_score = silhouette_score(np.array(embeddings)[valid_idx], [labels[i] for i in valid_idx])
    print(f"Silhouette Score (BERTopic): {sil_score:.4f}")

    # Top2Vec
    top2vec_model = Top2Vec(cleaned_texts, speed="learn", workers=4)
    top2vec_words, word_scores, topic_nums = top2vec_model.get_topics()
    top2vec_topics = [words[:5] for words in top2vec_words[:10]]
    print("Top2Vec Topics:")
    for i, topic in enumerate(top2vec_topics):
        print(f"Topic {i}: {', '.join(topic)}")
    coherence_model_top2vec = CoherenceModel(topics=top2vec_topics, texts=tokenized_texts, dictionary=dictionary, coherence='c_v')
    coherence_top2vec = coherence_model_top2vec.get_coherence()
    print(f"Top2Vec Coherence Score: {coherence_top2vec:.4f}")
    # Silhouette for Top2Vec
    try:

        doc_vectors = top2vec_model.document_vectors
        doc_ids = list(range(len(cleaned_texts)))
        doc_topics = top2vec_model.get_documents_topics(doc_ids)[0]
        
        # Filter out documents with no topic (-1)
        valid_idx = [i for i, topic in enumerate(doc_topics) if topic != -1]
        if len(valid_idx) > 1:  # i need at least 2 samples for silhouette score
            valid_vectors = doc_vectors[valid_idx]
            valid_topics = [doc_topics[i] for i in valid_idx]
            sil_score_top2vec = silhouette_score(valid_vectors, valid_topics)
            print(f"Silhouette Score (Top2Vec): {sil_score_top2vec:.4f}")
        else:
            print("Silhouette Score (Top2Vec): Not enough valid samples for calculation")
    except Exception as e:
        print(f"Silhouette Score (Top2Vec): Error - {e}")
    # GPT-based Topic Extraction
    print("Extracting GPT-based topics...")
    gpt_topics = extract_gpt_topics(cleaned_texts[:50], num_topics=10)
    print("GPT Topics:")
    for i, topic in enumerate(gpt_topics):
        print(f"Topic {i}: {', '.join(topic)}")
    
    # Ensure all topics are in the dictionary
    filtered_gpt_topics = []
    for topic in gpt_topics:
        filtered_topic = [word for word in topic if word in dictionary.token2id]
        if filtered_topic:
            filtered_gpt_topics.append(filtered_topic)
    
    coherence_model_gpt = CoherenceModel(topics=filtered_gpt_topics, texts=tokenized_texts,
                                        dictionary=dictionary, coherence='c_v')
    coherence_gpt = coherence_model_gpt.get_coherence()
    print(f"GPT Coherence Score: {coherence_gpt:.4f}")

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # Good practice on Windows
    main()
