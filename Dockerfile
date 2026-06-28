FROM python:3.11-slim

ENV PIP_DEFAULT_TIMEOUT=300 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .

# Install in layers so retries don't restart the entire download set
RUN pip install --upgrade pip setuptools wheel && \
    pip install --retries 10 --timeout 300 numpy==1.26.4 pandas==2.2.3 && \
    pip install --retries 10 --timeout 300 scikit-learn==1.4.2 joblib==1.4.2 && \
    pip install --retries 10 --timeout 300 plotly==5.24.1 streamlit==1.39.0 && \
    pip install --retries 10 --timeout 300 langgraph==0.2.45 langchain-core==0.3.21 langchain-ollama==0.2.0 python-dotenv==1.0.1

COPY . .

RUN python -m scripts.generate_faq_csv && python -m src.build_index

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]