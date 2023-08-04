import pandas as pd
import logging
logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)
from haystack.utils import  print_answers
from haystack.document_stores import ElasticsearchDocumentStore ,InMemoryDocumentStore
from haystack.pipelines import ExtractiveQAPipeline,GenerativeQAPipeline, Pipeline , FAQPipeline
from haystack.nodes import FARMReader,BM25Retriever ,Seq2SeqGenerator , TfidfRetriever
from sklearn.feature_extraction.text import TfidfVectorizer

QA = pd.read_csv('Dataset/QA.csv')

#Convert data to dictionary
QAdicts = QA.to_dict('records')

Extractivefinal_dicts=[]

for each in QAdicts:
            tmp = {}
            tmp['content'] = each.pop('text')
            tmp['body_title'] = each.pop('title')
            Extractivefinal_dicts.append(tmp)

Extractivedocument_store = InMemoryDocumentStore(use_bm25=True)
#reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")
#Extractiveretriever= TfidfRetriever(document_store=Extractivedocument_store)
Extractivedocument_store.delete_documents()
Extractivedocument_store.write_documents(Extractivefinal_dicts)
#Extractivepipe = ExtractiveQAPipeline( reader=reader,retriever=Extractiveretriever)


class ExtactiveModel:
    # create a TfidfVectorizer to compute the TF-IDF weights
    @staticmethod
    def handleQuestions(question):
        vectorizer = TfidfVectorizer(min_df=2)
        # retrieve all documents from the document store
        documents = Extractivedocument_store.get_all_documents()

        # concatenate the titles and contents into two separate lists
        titles = [doc.meta["body_title"] for doc in documents]
        contents = [doc.content for doc in documents]
        title_and_content = [title + " " + content for title, content in zip(titles, contents)]

        # fit the vectorizer on the concatenated strings
        vectorizer.fit(title_and_content)

        # compute the TF-IDF weights for the title and content separately
        title_weights = vectorizer.transform(titles)
        content_weights = vectorizer.transform(contents)

        # compute the cosine similarity between the query and the title and content weights
        query_weight = vectorizer.transform([question])
        title_scores = (0.9 * (query_weight @ title_weights.T)).toarray().ravel()
        content_scores = (0.1 * (query_weight @ content_weights.T)).toarray().ravel()

        # combine the title and content scores into a single score
        combined_scores = title_scores + content_scores

        # set the score attribute of each Document object to the corresponding score in combined_scores
        for i, doc in enumerate(documents):
            doc.score = combined_scores[i]

        # sort the documents by score in descending order and return the top-scoring documents
        documents.sort(key=lambda x: x.score, reverse=True)
        return (documents[0].content)

