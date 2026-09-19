import base64
import streamlit as st
from streamlit_option_menu import option_menu
import pdfplumber
import docx

class Layout:
    def __init__(self):
        self.title = "Nexus"
        self.sidebar_title = "Analysis Options"
        self.background_image = "bg.png"
        self.text_area = None
        self.manual_text = None
        self.uploaded_file = None
        self.selected_main = None
        self.uploaded_text = []

    def set_background(self):
        encoded_image = base64.b64encode(open(self.background_image, "rb").read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{encoded_image}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                height : 100vh;
                width : 100vw;
                margin: 0;
                padding: 0;
                overflow-y : auto;     /* enable vertical scrolling */
                overflow-x : hidden;   /* hide horizontal overflow */
            }}

            /* Hide Streamlit's default top padding */
            [data-testid="stHeader"] {{
                background: transparent;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    
    def sidebar_options(self):
        encoded_image = base64.b64encode(open(self.background_image, "rb").read()).decode()
        st.sidebar.title(self.sidebar_title)
        st.markdown(
            f"""
            <style>
            /* Sidebar Container */
            [data-testid="stSidebar"] {{
                background-image : url("data:image/png;base64,{encoded_image}");
            }}
            /* Sidebar heading styling */
            [data-testid="stSidebar"] h1, 
            [data-testid="stSidebar"] h2 {{
                font-size: 40px;       /* bigger font */
                font-weight: 800;      /* bold */
                color: white;        /* cyan glow effect */
                text-align: center;    /* center align */
                margin-top: -65px;     /* shift upwards */
                overflow-y : auto;     /* enable vertical scrolling */
                overflow-x : hidden;   /* hide horizontal overflow */
            }}
            </style>
            """,unsafe_allow_html=True
        )
        with st.sidebar: 
            self.selected_main = option_menu(
                menu_title = "Tool Type",
                options = ["Home","Sentiment Analysis","Emotion Analysis","Text Summarization","Language Detection","Keyword Extraction","Word Frequency","Sentence Length","POS Distibution","WordCloud Generation","Formality Checker"],
                styles={
                    "container": {
                        "padding": "5px",
                        "background-color": "#6D999D",
                        "font-color": "white",     # heading font color
                        "font-weight": "bold",    # heading bold
                        "font-size": "10px",      # heading size
                    },  
                    "icon": {"color": "white", "font-size": "20px"},                  # icon color
                    "nav-link": {
                        "font-size": "15px",
                        "text-align": "left",
                        "margin": "5px",
                        "color": "white",                                            # default font color
                        "background-color": "transparent",
                    },
                    "nav-link-selected": {
                        "background-color": "#77DEE8",                              # blue background when clicked
                        "color": "white",                                           # white font when clicked
                    },
                },
            )
        
    def main_area(self):
        st.markdown(f"<h1 style='color: white; font-size: 90px; font-weight: 900; text-align: center;margin-top: -130px'>{self.title}</h1>", unsafe_allow_html=True)
        self.text_area = st.text_area("Text For Playground Options","", height=100)
        self.manual_text = self.text_area
        # Custom CSS for textarea
        st.markdown(
            """
            <style>
            textarea {
                background-color: white !important;
                color: black !important;
                caret-color: red !important;   /* Change cursor color */
                font-size: 15px !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        self.uploaded_file = st.file_uploader("Upload Your File For Playground Options", type=["txt", "docx", "pdf"])

        # Process Uploaded file
        if self.uploaded_file is not None:
            # Reading Text File
            if self.uploaded_file.type == "text/plain":
                self.uploaded_text = str(self.uploaded_file.read(), "utf-8")

            # Reading PDF File
            elif self.uploaded_file.type == "application/pdf":
                with pdfplumber.open(self.uploaded_file) as pdf:
                    self.uploaded_text = ""
                    for page in pdf.pages:
                        self.uploaded_text += page.extract_text() + "\n"

            # Reading DOCX File
            elif self.uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                doc = docx.Document(self.uploaded_file)
                self.uploaded_text = "\n".join([para.text for para in doc.paragraphs])