import pandas as pd
import logging
logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)
from haystack.utils import  print_answers
from haystack.document_stores import ElasticsearchDocumentStore ,InMemoryDocumentStore
from haystack.pipelines import ExtractiveQAPipeline,GenerativeQAPipeline, Pipeline , FAQPipeline
from haystack.nodes import FARMReader,BM25Retriever ,Seq2SeqGenerator , TfidfRetriever
from sklearn.feature_extraction.text import TfidfVectorizer


Udacity = pd.read_csv('Dataset/Udacity Courses.csv')

#Convert data to dictionary
Udacitydicts = Udacity.to_dict('records')

FincalUdacity_dicts=[]

for each in Udacitydicts:
            tmp = {}
            tmp['content'] = each.pop('URL')
            tmp['body_title'] = each.pop('Title')
            tmp['Level']  =  each.pop('Level')
            tmp['Rating']  =  each.pop('Rating')
            FincalUdacity_dicts.append(tmp)

Udacitydocument_store = InMemoryDocumentStore(use_bm25=True)
#reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")
#Udacityretriever= TfidfRetriever(document_store=Udacitydocument_store)
Udacitydocument_store.delete_documents()
Udacitydocument_store.write_documents(FincalUdacity_dicts)
#Udacitypipe = ExtractiveQAPipeline( reader=reader,retriever=Udacityretriever)

class ExtractiveCourseModel:
    @staticmethod
    def handleQuestions(question,title_weight=1, content_weight=0):        
        # create a TfidfVectorizer to compute the TF-IDF weights
        vectorizer = TfidfVectorizer(min_df=2)
         # retrieve all documents from the document store
        documents = Udacitydocument_store.get_all_documents()

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
        title_scores = (title_weight * (query_weight @ title_weights.T)).toarray().ravel()
        content_scores = (content_weight * (query_weight @ content_weights.T)).toarray().ravel()

        # combine the title and content scores into a single score
        combined_scores = title_scores + content_scores

        # set the score attribute of each Document object to the corresponding score in combined_scores
        for i, doc in enumerate(documents):
            doc.score = combined_scores[i]

        # sort the documents by score in descending order and return the top-scoring documents
        documents.sort(key=lambda x: x.score, reverse=True)
        new_document = []
        for doc in documents:
            if doc.score >= 0.5:
              new_document.append(doc)
        documents = new_document
        new_document_set = [None, None, None]
        Final_document_set = []
        for doc in documents:
            level = doc.meta['Level']
            if level == 'beginner' and new_document_set[0] is None:
                new_document_set[0] = doc
            elif level == 'intermediate' and new_document_set[1] is None:
                new_document_set[1] = doc
            elif level == 'advanced' and new_document_set[2] is None:
                new_document_set[2] = doc

        if all(doc is None for doc in new_document_set):
            Final_document_set.append('Sorry there are no courses for this course now!')
        else:
            Final_document_set = [doc for doc in new_document_set if doc is not None]
        return Final_document_set
