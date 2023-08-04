from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import UserUtteranceReverted
from GenerativeModel import GenerativeModel 
from ExtactiveModel import ExtactiveModel
from ExtractiveCourseModel import ExtractiveCourseModel
from subprocess import Popen, PIPE, STDOUT , run
import time

class ActionHayStack(Action):

    def name(self) -> Text:
        return "action_answer"

    async def run(self, dispatcher: CollectingDispatcher,tracker: Tracker,domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        question = tracker.latest_message.get('text')
        
        try:
                print(question + 'is sent')
                answer = GenerativeModel.handleQuestions(question)
                dispatcher.utter_message(answer)
        
        except:
            dispatcher.utter_message('Sorry , i don\'t understand can you ask me again ?')
            
        return [UserUtteranceReverted()]




class ActionHayStackExtractive(Action):

    def name(self) -> Text:
        return "action_roadmap"

    async def run(self, dispatcher: CollectingDispatcher,tracker: Tracker,domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        question = tracker.latest_message.get('text')
        
        try:
                print(question + 'is sent')
                answer = ExtactiveModel.handleQuestions(question)
                dispatcher.utter_message(answer)       
        except:
            dispatcher.utter_message('Sorry , i don\'t understand can you ask me again ?')
            
        return [UserUtteranceReverted()]
    

class ActionHayStackUdacityExtractive(Action):

    def name(self) -> Text:
        return "action_course"

    async def run(self, dispatcher: CollectingDispatcher,tracker: Tracker,domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        question = tracker.latest_message.get('text')
        
        try:
                print(question + 'is sent')
                answer = ExtractiveCourseModel.handleQuestions(question)
                for doc in answer:
                    dispatcher.utter_message(text=f"{doc.meta['Level']}: {doc.content}")    
        except:
            dispatcher.utter_message('Sorry , i don\'t understand can you ask me again ?')
            
        return [UserUtteranceReverted()]