from functions_check import Functions
from functions_check import lang_detection
class Buttons:
    def __init__(self,functions:Functions):
        self.functions = functions
    
    def render(self):
        if self.functions.layout.selected_main == "Home":
            self.functions.home()
        if self.functions.layout.selected_main == "Sentiment Analysis":
            self.functions.sentiment_analysis()
        if self.functions.layout.selected_main == "Emotion Analysis":
            self.functions.emotion_analysis()
        if self.functions.layout.selected_main == "Text Summarization":
            self.functions.text_summarization()
        if self.functions.layout.selected_main == "Language Detection":
            self.functions.language_detection()
        if self.functions.layout.selected_main == "Keyword Extraction":
            self.functions.keyword_extraction()
        if self.functions.layout.selected_main == "Word Frequency":
            self.functions.word_frequency()
        if self.functions.layout.selected_main == "Sentence Length":
            self.functions.sentence_length()
        if self.functions.layout.selected_main == "POS Distibution":
            self.functions.pos_distribution()
        if self.functions.layout.selected_main == "WordCloud Generation":
            self.functions.wordcloud_generation()
        if self.functions.layout.selected_main == "Formality Checker":
            self.functions.formality_checker()

