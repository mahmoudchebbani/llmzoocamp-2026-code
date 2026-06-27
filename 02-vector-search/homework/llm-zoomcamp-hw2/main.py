from embedder import Embedder
from git_helper import load_lessons
from minsearch import VectorSearch
import numpy as np
from ingest import build_index

def rrf(result_lists, k=60, num_results=5):
    scores = {}
    docs = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            key = (doc["filename"], doc["start"])
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            docs[key] = doc

    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ranked[:num_results]]

def main():
    print("llm-zoomcamp-hw2!")

    embedder = Embedder()
    embedding_1 = embedder.encode('How does approximate nearest neighbor search work?')
    chunks = load_lessons()
    batch = [chunk['content'] for chunk in chunks]

    # sqlite_search_vector_doc = [doc for doc in documents if doc['filename'] == '02-vector-search/lessons/07-sqlitesearch-vector.md']

    doc_embedding = embedder.encode_batch(batch)

    scores = doc_embedding.dot(embedding_1)

    idx = np.argmax(scores)
    print(chunks[idx]['filename'])

    vindex = VectorSearch(keyword_fields=['filename'])
    vindex.fit(doc_embedding, chunks)

    embedding_2 = embedder.encode('What metric do we use to evaluate a search engine?')
    search_result = vindex.search(embedding_2, num_results=5)
    print(search_result[0]['filename'])


    print("===========================================")
    index = build_index(chunks)
    query_3 = 'How do I store vectors in PostgreSQL?'
    text_search_result = index.search(
        query_3,
        num_results=5,
    )
    for result in text_search_result:
        print(result['filename'])
    print("------")
    embedding_3 = embedder.encode(query_3)
    vector_search_result = vindex.search(embedding_3, num_results=5)
    for result in vector_search_result:
        print(result['filename'])

    print("===========================================")
    query_4 = "How do I give the model access to tools?"
    text_search_result = index.search(
        query_4,
        num_results=5,
    )
    embedding_4 = embedder.encode(query_4)
    vector_search_result = vindex.search(embedding_4, num_results=5)
    results = rrf([vector_search_result, text_search_result])
    print(results[0]['filename'])


if __name__ == "__main__":
    main()
