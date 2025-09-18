# Elite-Body-AI-Chatbot

# Objective:
 To Develop a basic web-based chatbot that responds using AI  and dynamically updates a document section on the website based on user queries or AI suggestions.


Elite-Body-AI-Chatbot is a Streamlit web app that answers questions about Elite Body Home services using website scraping, semantic search, and local AI models — no paid API keys required.

## Features

- Scrapes https://elitebodyhome.com/ content dynamically
- Uses SentenceTransformers + FAISS for semantic search of website content
- Answers user queries with Hugging Face local question-answering model
- Streamlit UI with chat history and dynamic service document section
- Fully open-source, runs locally with free models, no API keys needed

## Setup

### Prerequisites

- Python 3.8+
- Git (optional)
- Internet connection (for scraping)

### Installation



 # 1. Managing dependencies
 Create requirements.txt with your python libs: This file lets users install needed packages easily.

 # 2. Deployment: Running Locally on Any Laptop
 Prerequisites for the user:
 Python 3.8+ installed (recommended Python 3.10 or 3.11)

 Git installed (optional, but recommended)

 # A. Step-by-step instructions
 Clone/download the repo:
 git clone https://github.com/<Mrud11>/elitebody-chatbot.git
 cd elitebody-chatbot

 # B. Create and activate a Python virtual environment (recommended):
 python -m venv venv
 venv\Scripts\activate

 # C. Install dependencies:

  bash
 pip install --upgrade pip
 pip install -r requirements.txt


### Installation

 # D. Run the Streamlit app:

 bash
 streamlit run app.py

## License

MIT License

 
