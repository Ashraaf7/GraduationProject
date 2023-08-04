import pandas as pd
import logging
logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)
from haystack.utils import  print_answers
from haystack.document_stores import ElasticsearchDocumentStore ,InMemoryDocumentStore
from haystack.pipelines import ExtractiveQAPipeline,GenerativeQAPipeline, Pipeline , FAQPipeline
from haystack.nodes import FARMReader,BM25Retriever ,Seq2SeqGenerator , TfidfRetriever
from sklearn.feature_extraction.text import TfidfVectorizer
from subprocess import Popen, PIPE, STDOUT , run

Mediumdata =pd.read_csv('Dataset/PreMediumData.csv')
Mediumdata2 =pd.read_csv('Dataset/PreMediumData2.csv')
Mediumdata3 =pd.read_csv('Dataset/PreMediumData3.csv')
Geeksdata =pd.read_csv('Dataset/PreGeeksData.csv')
#Convert data to dictionary
Mediumdicts = Mediumdata.to_dict('records')
Mediumdicts2 = Mediumdata2.to_dict('records')
Mediumdicts3 = Mediumdata3.to_dict('records')
Geeksdicts = Geeksdata.to_dict('records')
final_dicts=[]
for each in Mediumdicts:
            tmp = {}
            tmp['content'] = each.pop('text')
            tmp['body_title'] = each.pop('title')
            final_dicts.append(tmp)
for each in Mediumdicts2:
            tmp = {}
            tmp['content'] = each.pop('text')
            tmp['body_title'] = each.pop('title')
            final_dicts.append(tmp)
for each in Mediumdicts3:
            tmp = {}
            tmp['content'] = each.pop('text')
            tmp['body_title'] = each.pop('title')
            final_dicts.append(tmp)

for each in Geeksdicts:
            tmp = {}
            tmp['content'] = each.pop('content')
            tmp['body_title'] = each.pop('title')
            final_dicts.append(tmp)
        
document_store = InMemoryDocumentStore(use_bm25=True)

generator = Seq2SeqGenerator(model_name_or_path="vblagoje/bart_lfqa")#take time
retriever = BM25Retriever (document_store=document_store)   
document_store.delete_documents()
document_store.write_documents(final_dicts) 

pipe = GenerativeQAPipeline(generator=generator, retriever=retriever)  


class GenerativeModel:
    @staticmethod
    def handleQuestions(question):
        answer = pipe.run(question, params={"Retriever": {"top_k": 5}, "Generator": {"top_k": 1}})#take time       
        answer = answer["answers"][0]
        return answer.answer