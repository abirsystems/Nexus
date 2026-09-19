# Importing Libraries and files
from layout import Layout
from language_map import LANGUAGE_MAP,REVERSE_LANGUAGE_MAP
import streamlit as st
from models.language_detector import LanguageDetector
import time
from string import punctuation
from collections import Counter
from heapq import nlargest
import pandas as pd
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from keybert import KeyBERT
from textblob import TextBlob
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification

# Cache the model so it's loaded only once.
# model_name is now a Hugging Face Hub repo id, so `from_pretrained` downloads
# (and transformers caches) the weights instead of reading from a local folder.
@st.cache_resource(show_spinner=False)
def get_distilbert_pipeline(model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

@st.cache_resource(show_spinner=False)
def get_goemotions_pipeline(model_name: str = "SamLowe/roberta-base-go_emotions"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    # top_k is requested per-call (with top_k=None) instead of via the deprecated
    # return_all_scores=True construction flag, whose output shape isn't consistent
    # across transformers versions.
    return pipeline("text-classification", model=model, tokenizer=tokenizer)

# Splits text into chunks that fit inside a model's max sequence length, instead of
# truncating and throwing away everything past the first ~512 tokens. Splits on the
# model's own tokenizer so chunk boundaries always line up with real token counts.
# max_tokens defaults to 510 to leave room for the 2 special tokens ([CLS]/[SEP],
# <s>/</s>, etc.) the pipeline adds back in on top of each chunk.
def chunk_text_by_tokens(tokenizer, text, max_tokens=510):
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return [(text, 0)]
    chunks = []
    for i in range(0, len(token_ids), max_tokens):
        chunk_ids = token_ids[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append((chunk_text, len(chunk_ids)))
    return chunks

# A text-classification call made with top_k=None can come back as a flat
# list of dicts (one per label) or, for some pipeline/transformers versions,
# a list wrapping that list. This flattens either shape into one list of dicts.
def normalize_classification_result(result):
    if isinstance(result, dict):
        return [result]
    if isinstance(result, list) and result and isinstance(result[0], list):
        return result[0]
    return result

# Label mapping (SST-2 fine-tuned DistilBERT is binary: POSITIVE / NEGATIVE)
label_map = {
    "POSITIVE": "Positive",
    "NEGATIVE": "Negative"
}

formal_words = set([
    "please", "thank you", "sincerely", "best regards", "kindly", "excuse me", "dear", "respectfully", 
    "honored", "appreciate", "cordially", "regards", "with respect", "for your consideration", "would you mind", 
    "grateful", "acknowledge", "thankful", "truly", "understanding", "yours truly", "faithfully", "congratulations", 
    "compliments", "with appreciation", "much obliged", "could you", "may I", "please be advised", "permit me", 
    "in compliance", "at your earliest convenience", "I look forward to", "please let me know", "to whom it may concern", 
    "please find attached", "appreciatively", "with all due respect", "it would be appreciated", "it is my honor", 
    "I would like to", "I would appreciate", "best wishes", "for your information", "kind regards", "thanks in advance", 
    "yours sincerely", "in regards to", "with gratitude", "in appreciation", "with best wishes", "could you kindly", 
    "looking forward to", "with the utmost respect", "please accept", "with warm regards", "with kind regards", 
    "at your service", "please confirm", "my pleasure", "respectfully yours", "I humbly", "we are grateful", 
    "please inform", "as per your request", "in your service", "as previously mentioned", "I trust", 
    "with respect and appreciation", "please advise", "your prompt response is appreciated", "I would be happy to", 
    "thank you very much", "I hope this finds you well", "I am writing to", "thank you for your time", 
    "it is my privilege", "your attention to this matter", "looking forward", "it would be my pleasure", "have a great day", 
    "gratefully", "best regards", "thank you for your cooperation", "with sincere regards", "it would be an honor", 
    "please be reminded", "in good faith", "with consideration", "your feedback is appreciated", "I would be delighted", 
    "please note", "with thanks", "with appreciation", "to be continued", "please take note", "I hope to hear from you", 
    "in conclusion", "yours faithfully"
])

informal_words = set([
    "hey", "yo", "sup", "wanna", "gimme", "okay", "lol", "yo", "bro", "dude", "chill", "buddy", "what's up", "nah", 
    "bff", "lmao", "omg", "lolz", "rofl", "idk", "ttyl", "brb", "catch you later", "peace", "fam", "no worries", 
    "whatever", "cya", "k", "pls", "gotcha", "no prob", "tbh", "lmfao", "holla", "hype", "cray", "l8r", "yeah", 
    "yass", "babe", "yo dude", "dawg", "wut", "wth", "chillax", "bffl", "glhf", "fml", "smh", "wtf", "yo man", 
    "soo", "lolol", "lmao", "broke", "yolo", "lit", "savage", "hit me up", "stay woke", "good vibes", "let's go", 
    "hey yo", "wanna go", "brother", "homie", "sure thing", "let's catch up", "let me know", "ain't nobody", "hit me", 
    "chillin", "what's good", "grub", "swag", "bored", "no cap", "vibe", "gurl", "hey there", "come thru", "y'all", 
    "fomo", "yo girl", "that's fire", "send it", "catch ya", "mad", "lucky", "peace out", "squad", "g'day", "yo fam", 
    "slay", "dank", "flex", "slay queen", "cringe", "bruh", "rly", "imma", "who's down", "bored af", "lowkey", 
    "highkey", "for real", "sheesh", "bet", "on fleek", "g'day mate", "dank memes", "smash", "hit me up later", "nah fam", 
    "goals", "peace bro", "weird flex", "lit af", "sick", "that's dope", "howdy", "gimme a sec"
])

def check_formality(text):
    blob = TextBlob(text)
    words = set(w.lower() for w in blob.words)

    formal_score = sum(1 for word in words if word in formal_words)
    informal_score = sum(1 for word in words if word in informal_words)

    if formal_score > informal_score:
        return "Formal"
    elif informal_score > formal_score:
        return "Informal"
    else:
        polarity = blob.sentiment.polarity
        if polarity > 0.1:
            return "Informal"
        elif polarity < -0.1:
            return "Formal"
        else:
            return "Neutral"

# Loading the spacy model (cached so it only loads once, not on every script rerun)
@st.cache_resource(show_spinner=False)
def get_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = get_spacy_model()

# Extractive Summarization Function
def Extractive_Summarizer(text,length):
    # Processing the text
    doc = nlp(text)
    # Creating a list of tokens, excluding stop words and punctuation
    tokens = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct and token.text != "\n"]
    # Calculating Normalized Word Frequencies
    word_freq = Counter(tokens)
    if not word_freq:
        return text  # nothing to score (e.g. text is only stopwords/punctuation)
    max_freq = max(word_freq.values())
    for word in word_freq.keys():
        word_freq[word] = word_freq[word]/max_freq
    # Creating Sentence Tokens (kept as spaCy spans, not plain strings, so we can
    # score using the same tokenization as word_freq and can restore original order)
    sentences = list(doc.sents)
    # Scoring Sentences: use spaCy's own tokens per sentence (so punctuation-attached
    # words like "amazing." still match word_freq's "amazing"), and average by the
    # number of scoring words in the sentence so long sentences don't win purely on length
    sent_score = {}
    for i, sent in enumerate(sentences):
        sent_words = [tok.text.lower() for tok in sent if not tok.is_stop and not tok.is_punct and tok.text != "\n"]
        if not sent_words:
            continue
        score = sum(word_freq.get(w, 0) for w in sent_words) / len(sent_words)
        sent_score[i] = score
    if not sent_score:
        return text
    # Selecting the top n sentences, then restoring their original order so the
    # summary reads coherently instead of jumping around by score
    length = min(length, len(sent_score))
    top_indices = nlargest(length, sent_score, key=sent_score.get)
    top_indices.sort()
    summary = " ".join(sentences[i].text.strip() for i in top_indices)
    return summary

# Cache the T5 model/tokenizer so they're downloaded/loaded once, not on every summarization call.
# model_name is a Hugging Face Hub repo id; the tokenizer_config "extra_special_tokens"
# list-vs-dict bug only ever showed up in locally re-saved configs, so pulling fresh
# from the Hub sidesteps it and the patching step is no longer needed.
@st.cache_resource(show_spinner=False)
def get_t5_summarizer(model_name: str = "t5-small"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

# Abstractive Summarization Function
def _t5_generate_summary(tokenizer, model, text, max_length=120, min_length=40):
    # T5 was fine-tuned on the exact literal prefix "summarize: " (no space before the
    # colon) — using "summarize : " weakens the model's recognition of the task cue.
    inputs = tokenizer("summarize: " + text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=min_length,
        num_beams=4,
        do_sample=False,
        early_stopping=True,
        length_penalty=2.0,          # favors concise, well-formed summaries over rambling ones
        no_repeat_ngram_size=3,      # standard fix for repeated phrases in T5/BART summaries
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def Abstractive_Summarizer(text):
    # Load tokenizer and model (cached after first call)
    tokenizer, model = get_t5_summarizer()

    # Split into token-bounded chunks (same helper used for sentiment/emotion) so long
    # documents are fully covered instead of only summarizing their first ~512 tokens.
    # 500 tokens leaves room for the "summarize: " prefix's own tokens.
    chunks = chunk_text_by_tokens(tokenizer, text, max_tokens=500)
    chunk_texts = [c for c, weight in chunks if weight > 0]
    if not chunk_texts:
        return ""

    if len(chunk_texts) == 1:
        return _t5_generate_summary(tokenizer, model, chunk_texts[0])

    # Map: summarize each chunk on its own
    chunk_summaries = [_t5_generate_summary(tokenizer, model, chunk) for chunk in chunk_texts]

    # Reduce: summarize the combined chunk summaries into one coherent final summary.
    # If that combined text is itself still too long for one pass, chunk-and-summarize
    # it again until it fits.
    combined = " ".join(chunk_summaries)
    while len(tokenizer.encode(combined, add_special_tokens=False)) > 500:
        combined_chunks = [c for c, weight in chunk_text_by_tokens(tokenizer, combined, max_tokens=500) if weight > 0]
        combined = " ".join(_t5_generate_summary(tokenizer, model, c) for c in combined_chunks)

    return _t5_generate_summary(tokenizer, model, combined, max_length=150, min_length=40)

# Cache the detector so the .tflite model is loaded from disk only once,
# not on every single detection call (this was the main source of slowness/hangs)
@st.cache_resource(show_spinner=False)
def get_language_detector(score_threshold: float = 0.5):
    return LanguageDetector(score_threshold=score_threshold)

# language detection function
def lang_detection(text):
    detector = get_language_detector()
    results = detector.detect(text)
    if not results:  # empty list
        lang = "unknown"
    else:
        lang, score = results[0]  # get the top result
        # Use the same threshold the detector was already built with above,
        # instead of re-checking against a different, inconsistent value here
        if score <= 0.5:
            lang = "unknown"
    lang_name = LANGUAGE_MAP.get(lang, "Unknown Language")
    return lang_name

# Cache the KeyBERT model so it isn't reloaded from scratch on every call
@st.cache_resource(show_spinner=False)
def get_keybert_model():
    return KeyBERT()

class Functions:
    def __init__(self,layout:Layout):
        self.layout = layout
        self.uploaded_text = self.layout.uploaded_text
        self.manual_text = self.layout.manual_text
        self.combined_text = ""
        
        # Saving in combined_text
        if self.manual_text:
            self.combined_text += self.manual_text + " "
        elif self.uploaded_text:
            self.combined_text += self.uploaded_text

    def home(self):
        st.title("Unlock the Power of Language")
        st.write(
            """
            Welcome to **Nexus NLP Playground**, your interactive hub for experimenting, learning, 
            and innovating with Natural Language Processing. Whether you’re a student, researcher, 
            or developer, this platform is designed to make NLP concepts accessible, practical, and fun.
            """
        )

    def sentiment_analysis(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        st.header("Sentiment Analysis", anchor=None,text_alignment="center")
        if self.combined_text:
            # st.subheader("Result")

            # Load cached pipeline
            sentiment_pipeline = get_distilbert_pipeline(model_name)

            # Split into token-bounded chunks so long texts/uploaded files are
            # fully analyzed instead of just their first ~512 tokens, then combine
            # each chunk's result into one overall score, weighted by chunk length.
            chunks = chunk_text_by_tokens(sentiment_pipeline.tokenizer, self.combined_text)
            positive_weight = 0.0
            total_weight = 0
            for chunk_text, weight in chunks:
                if weight == 0:
                    continue
                chunk_result = sentiment_pipeline(chunk_text, truncation=True, max_length=512)[0]
                # Convert each chunk's result to "probability the chunk is positive"
                # so opposite-label chunks can be combined on the same scale.
                positive_prob = chunk_result["score"] if chunk_result["label"] == "POSITIVE" else 1 - chunk_result["score"]
                positive_weight += positive_prob * weight
                total_weight += weight

            avg_positive_prob = positive_weight / total_weight if total_weight else 0.5
            if avg_positive_prob >= 0.5:
                label = label_map["POSITIVE"]
                score = avg_positive_prob
            else:
                label = label_map["NEGATIVE"]
                score = 1 - avg_positive_prob

            # Display nicely
            st.markdown(
                f"""
                <div style='border:2px solid #333; padding:15px; border-radius:8px; margin-bottom:20px;'>
                    <strong style="font-size:30px;">Sentiment: {label}</strong><br>
                    <span style="font-size:20px;">Confidence: {score:.4f}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.warning("Please enter text in the text area or upload a file for Analysis.")

    def emotion_analysis(self, model_name: str = "SamLowe/roberta-base-go_emotions"):
        st.header("Emotion Analysis", anchor=None, text_alignment="center")

        if self.combined_text:
            emotion_pipeline = get_goemotions_pipeline(model_name)

            # Same chunking approach as sentiment analysis: split into token-bounded
            # chunks, then average each emotion label's score across chunks (weighted
            # by chunk length) instead of only scoring the first ~512 tokens.
            chunks = chunk_text_by_tokens(emotion_pipeline.tokenizer, self.combined_text)
            label_totals = {}
            total_weight = 0
            for chunk_text, weight in chunks:
                if weight == 0:
                    continue
                chunk_result = emotion_pipeline(chunk_text, truncation=True, max_length=512, top_k=None)
                chunk_result = normalize_classification_result(chunk_result)
                for item in chunk_result:
                    label_totals[item["label"]] = label_totals.get(item["label"], 0.0) + item["score"] * weight
                total_weight += weight

            if total_weight:
                label_averages = {lbl: total / total_weight for lbl, total in label_totals.items()}
            else:
                label_averages = {}

            label = max(label_averages, key=label_averages.get)
            confidence = label_averages[label]

            # Display top emotion cleanly
            st.markdown(
                f"""
                <div style='border:2px solid #333; padding:15px; border-radius:8px;'>
                    <strong style="font-size:30px;">Top Emotion: {label}</strong><br>
                    <span style="font-size:20px;">Confidence: {confidence:.4f}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.warning("Please enter text or upload a file for analysis.")

    def text_summarization(self):
        st.header("Text Summarization", text_alignment="center")
        if self.combined_text:
            # Extractive Summary
            st.subheader("Extractive Summarization")
            length = st.slider("Summary Length (Number of Sentences)", min_value=1, max_value=10, value=3)
            summary_extractive = Extractive_Summarizer(self.combined_text, length)
            st.markdown(
                f"""
                <div style='border:2px solid #333; padding:15px; border-radius:8px; margin-bottom:20px;'>
                    <strong style="font-size:16px;">{summary_extractive}</strong>
                </div>
                """,
                unsafe_allow_html=True
            )
            # Abstractive Summary
            st.subheader("Abstractive Summarization")
            summary_abstractive = Abstractive_Summarizer(self.combined_text)
            st.markdown(
                f"""
                <div style='border:2px solid #333; padding:15px; border-radius:8px;'>
                    <strong style="font-size:16px;">{summary_abstractive}</strong>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.warning("Please enter text in the text area or upload a file for Analysis.")

    def language_detection(self):
        st.header("Language Detection", text_alignment="center")
        if self.combined_text:
            lang_name = lang_detection(self.combined_text)
            if lang_name == "Unknown Language":
                st.warning("Unable to detect the language of the input text. Please ensure the text is in a supported language.")
            else:
                st.subheader("Detected Language : ")
                st.markdown(
                        f"""
                        <div style='border:2px solid #333; padding:15px; border-radius:8px;'>
                            <strong style="font-size:24px;">{lang_name}</strong>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        else:
           st.warning("Please enter text in the text area or upload a file for Analysis.")

    def keyword_extraction(self):
        st.header("Keyword Extraction",text_alignment="center")
        if self.combined_text:
            kw_model = get_keybert_model()
            keywords_with_scores = kw_model.extract_keywords(
                self.combined_text,
                keyphrase_ngram_range=(1, 2),
                stop_words='english',
                top_n=10
            )
            keywords_only = [kw for kw, score in keywords_with_scores]
            # Join keywords vertically using <br>
            keywords_text = "<br>".join(keywords_only)
            # Display inside one white box
            st.markdown(
                f"""
                <div style="
                    background-color:white;
                    width:70%;                /* reduce width (try 50–80%) */
                    padding:8px 12px;         /* smaller padding */
                    margin-top:10px;
                    border-radius:8px;
                    border:1px solid #ddd;
                    font-size:20px;
                    font-weight:bold;
                    color:black;
                    text-align:center;
                    overflow-y:auto;          /* scroll if content exceeds height */
                    margin-left:auto;         /* center horizontally */
                    margin-right:auto;
                ">
                {keywords_text}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.warning("Please enter text in the text area or upload a file for Analysis.")

    def word_frequency(self):
        st.header("Word Frequency",text_alignment="center")
        if self.combined_text:
            color_map1 = plt.colormaps()
            color_map1.insert(0,"select a colormap")
            color1 = st.selectbox("Select Color Map",options=color_map1,key="word_freq_colormap")
            tokens = nltk.word_tokenize(self.combined_text.lower())
            words = [w for w in tokens if w.isalpha() and w not in nltk.corpus.stopwords.words('english')]
            freq = Counter(words).most_common(20)
            labels, counts = zip(*freq)
            if color1 != "select a colormap":
                fig,ax = plt.subplots()
                sns.barplot(y=list(labels),x=list(counts),hue=list(labels),palette=color1,ax=ax)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
                ax.set_xlabel("Frequency")
                ax.set_ylabel("Words")
                ax.set_title("Top 20 Word Frequencies")
                st.pyplot(fig=fig)
        else:
           st.warning("Please enter text in the text area or upload a file for Analysis.")

    def sentence_length(self):
        st.header("Sentence Length",text_alignment="center")
        if self.combined_text:
            color_map2 = plt.colormaps()
            color_map2.insert(0,"select a colormap")
            color2 = st.selectbox("Select Color Map",options=color_map2,key="sent_length_colormap") 
            sentences = nltk.sent_tokenize(self.combined_text)
            lengths = [len(s.split()) for s in sentences]
            group = ["short" if l < 15 else "long" for l in lengths]
            df = pd.DataFrame({"lengths": lengths, "group": group})
            if color2 != "select a colormap":
                fig,ax = plt.subplots()
                sns.histplot(data=df,x="lengths",hue="group", bins=20, kde=True, palette=color2, ax=ax)
                # Labels and title
                ax.set_xlabel("Sentence Length (words)")
                ax.set_ylabel("Frequency")
                ax.set_title("Distribution of Sentence Lengths")
                st.pyplot(fig=fig)
        else:
           st.warning("Please enter text in the text area or upload a file for Analysis.")

    def pos_distribution(self):
        st.header("POS Distribution",text_alignment="center")
        if self.combined_text:
            color_map3 = plt.colormaps()
            color_map3.insert(0,"select a colormap")
            color3 = st.selectbox("Select Color Map",options=color_map3,key="pos_dist_colormap")
            doc = nlp(self.combined_text)
            pos_counts = Counter([token.pos_ for token in doc])
            # Map POS codes to explanations
            pos_labels = {pos: spacy.explain(pos) for pos in pos_counts.keys()}
            # Replace codes with explanations
            labels = [pos_labels[pos] if pos_labels[pos] else pos for pos in pos_counts.keys()]
            counts = list(pos_counts.values()) 
            if color3 != "select a colormap":
                fig,ax = plt.subplots()
                sns.barplot(y=list(labels),x=list(counts),palette=color3,ax=ax,orient="h")
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
                ax.set_xlabel("Frequency")
                ax.set_ylabel("Parts Of Speech")
                ax.set_title("Parts Of Speech Distribution")
                st.pyplot(fig=fig)
        else:
           st.warning("Please enter text in the text area or upload a file for Analysis.")

    def wordcloud_generation(self):
        st.header("WordCloud Generation",text_alignment="center")
        if self.combined_text:
            color_map4 = plt.colormaps()
            color_map4.insert(0,"select a colormap")
            color4 = st.selectbox("Select Color Map",options=color_map4,key="word_cloud_colormap") 
            if color4 != "select a colormap":
                fig,ax = plt.subplots()
                wordcloud = WordCloud(
                    width=800,
                    height=400,
                    background_color='white',
                    colormap=color4,
                    max_words=200
                ).generate(self.combined_text)
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)
        else:
           st.warning("Please enter text in the text area or upload a file for Analysis.")

    def formality_checker(self):
        st.header("Formality Checker", anchor=None,text_alignment="center")

        if self.combined_text:
            # Run rule-based check
            blob = TextBlob(self.combined_text)
            words = set(w.lower() for w in blob.words)

            formal_score = sum(1 for word in words if word in formal_words)
            informal_score = sum(1 for word in words if word in informal_words)

            # Decide label
            if formal_score > informal_score:
                label = "Formal"
                confidence = formal_score / (formal_score + informal_score + 1e-6)
            elif informal_score > formal_score:
                label = "Informal"
                confidence = informal_score / (formal_score + informal_score + 1e-6)
            else:
                polarity = blob.sentiment.polarity
                if polarity > 0.1:
                    label = "Informal"
                elif polarity < -0.1:
                    label = "Formal"
                else:
                    label = "Neutral"
                confidence = 0.5  # fallback confidence

            # Display result in styled box
            st.markdown(
                f"""
                <div style='border:2px solid #333; padding:15px; border-radius:8px;'>
                    <strong style="font-size:30px;">Formality: {label}</strong><br>
                    <span style="font-size:20px;">Confidence: {confidence:.4f}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.warning("Please enter text or upload a file for analysis.")