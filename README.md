# MERCURE Application 👾

## 📋 Description

MERCURE is a Django application enriched with AI tooling (LangChain, LangGraph, PyTorch, Transformers) for ingesting, analyzing, and classifying news articles. The default setup targets AML/CFT (LCB-FT) press monitoring with a Retrieval-Augmented Generation (RAG) workflow, yet you can port it to any domain by updating the keyword list inside `keywords_app` (UI or `keywords.sqlite3`).

## Features

- **Factiva article import & management:** Upload Excel exports and store them safely
- **Automated classification:** LangChain/LangGraph + OpenAI pipeline to detect relevant signals
- **Advanced LangGraph workflow:** Multi-stage classification pipeline with confidence scoring and ambiguity validation
- **Keyword-driven analytics:** Filter articles with custom expressions
- **Excel export:** Generate ready-to-share reports
- **Full web interface:** Manage workflows through Django templates
- **REST API documentation:** Auto-generated via Swagger/OpenAPI

## Visualizations

All screenshots live in the `Visualizations/` directory so you can reuse or replace them as needed:

![Main menu](Visualizations/Main%20menu.png)
![Data management](Visualizations/Data%20management%20and%20visualization.png)
![Import data](Visualizations/Page%20to%20import%20Data.png)
![Choose classification data](Visualizations/Page%20to%20choose%20the%20data%20for%20classification.png)
![Classified articles](Visualizations/Page%20to%20observe%20classified%20articles.png)

## Tech Stack

- **Backend:** Django 4.2.20
- **AI/ML:** PyTorch, Transformers, Sentence-Transformers
- **LLM tooling:** LangChain, LangGraph, OpenAI GPT
- **Databases:** Multiple SQLite files
- **Frontend:** Django templates + Bootstrap
- **API:** Django REST Framework
- **Documentation:** DRF Spectacular (Swagger)

## Prerequisites

- **Python 3.9** (required for PyTorch compatibility)
- **Git**
- **OpenAI API key** (mandatory for the AI features)

### Bring your own data

The repository does **not** contain sample Factiva exports because the content is proprietary and licensed. Import your own Excel exports (matching the expected columns that you can modify) before running the classification workflows.

## Installation

### 1. Clone the repository

```bash
git clone <repo-url>
cd exa-python-django-starter-kit
```

### 2. Create a Python 3.9 virtual environment

```bash
# Make sure Python 3.9 is installed
python3.9 --version

# Create the virtual environment
python3.9 -m venv venv39

# Activate the environment
source venv39/bin/activate   # macOS/Linux
# or
venv39\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
cd app-src
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file inside `app-src`:

```bash
# Django
DEBUG=True
DJANGO_SECRET_KEY=replace-with-your-secret-key
ALLOWED_HOSTS=localhost,127.0.0.1,0.0.0.0

# OpenAI (required for AI workflows)
OPENAI_API_KEY=replace-with-your-openai-key

# Project settings
ENV_PATH=.env
PYPROJECT_TOML_PATH=pyproject.toml
```

### 5. Verify the installation

```bash
DJANGO_SECRET_KEY="django-insecure-temp-key" \
ALLOWED_HOSTS="localhost,127.0.0.1" \
DEBUG=True \
OPENAI_API_KEY="temp-key" \
python manage.py check
```

## Run the application

### Development mode

```bash
source venv39/bin/activate
cd app-src
DJANGO_SECRET_KEY="django-insecure-temp-key" \
ALLOWED_HOSTS="localhost,127.0.0.1,0.0.0.0" \
DEBUG=True \
OPENAI_API_KEY="your-key" \
python manage.py runserver 
```



## Databases

Change the database as you want, initially there was three SQLite databases:

- **articles.sqlite3:** Classified articles
- **factiva.sqlite3:** All Factiva articles content available
- **keywords.sqlite3:** Keyword dictionary

## Advanced configuration

### Retrieval-Augmented Generation (RAG)

The RAG workflow (embeddings, FAISS retrieval, LangChain `RetrievalQA`, OpenAI calls) lives in `app-src/factiva_app/views.py`, primarily across `process_article`, `predict_all_articles_parallel`, and the `reporting_articles` view. Pair this code with `Visualizations/RAG Algorithm Architecture.png` for a quick architecture refresher.

![RAG Architecture](Visualizations/RAG%20Algorithm%20Architecture.png)

### LangGraph Classification Workflow

The project includes an advanced **LangGraph-based classification pipeline** (`app-src/factiva_app/langgraph_classifier.py`) that provides enhanced precision for LCB-FT article classification. This multi-stage workflow offers:

- **Multi-stage processing:** Document preparation → Initial classification → Confidence scoring → Ambiguity validation → Final decision
- **Confidence scoring:** Each keyword classification receives a confidence score (0.0-1.0) to identify ambiguous cases
- **Ambiguity validation:** Cases with medium confidence (0.4-0.7) are re-validated with detailed analysis
- **State management:** Shared state across workflow steps tracks processing history, tokens used, and costs
- **Improved precision:** Better detection of relevant LCB-FT articles through multi-pass validation

The LangGraph workflow is **automatically used** by the `process_article` function in `views.py`, which calls `process_article_with_langgraph` from `langgraph_classifier.py`. The workflow returns additional metadata including:
- `confidence_scores`: Confidence level for each keyword
- `reasoning`: Explanation of the final classification decision
- `processing_steps`: Audit trail of workflow execution

This enhanced classification is seamlessly integrated into the existing `reporting_articles` workflow and requires no additional configuration.

### Embedding model

MERCURE relies on a RoBERTa embedding model. Adjust the path inside `factiva_app/views.py`:

```python
embedding = HuggingFaceEmbeddings(
    model_name="/path/to/your/roberta",
    model_kwargs={
        "device": "cpu",
        "trust_remote_code": True,
    },
)
```

### Proxy

If you need to route outbound traffic through a proxy, uncomment and configure the snippet inside `factiva_app/views.py`:

```python
PROXY_source = "http://127.0.0.1:3128"
http_client = Client(proxy=PROXY_source, verify=False)
```

## Usage

### 1. Upload Factiva articles

1. Navigate to http://localhost:8000/factiva/upload/
2. Upload an Excel file containing:
   - `title`
   - `source_name`
   - `publisher_name`
   - `source_code`
   - `publication_date`
   - `content`

### 2. Manage keywords

1. Go to http://localhost:8000/keywords/
2. Add or update the expressions that drive the classifier

### 3. Classify articles

1. Visit http://localhost:8000/factiva/reporting/
2. Pick the date range, sources, and relevance filters
3. Launch the automated classification workflow

### 4. Export results

- Download the Excel summary
- Review classified articles stored in the database

## Security

- Rotate `DJANGO_SECRET_KEY` before production
- Serve the app behind HTTPS
- Update `ALLOWED_HOSTS` with the correct domains
- Treat your OpenAI API key as a secret (env vars, vault, etc.)


## Author

Enge NOUADJE (ndje-enge)