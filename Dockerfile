# Dockerfile
# ----------
# Ye image poore Amazon Seller Multi-Agent app ko containerize karti hai
# taake ye kisi bhi cloud (Azure, AWS, ya kahin bhi Docker chale) par
# same tarah se chal sake.

FROM python:3.11-slim

WORKDIR /app

# System dependency jo healthcheck ke liye chahiye
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Pehle sirf requirements copy karein — Docker layer caching se rebuild fast hoga
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Baaki poora app code copy karein
COPY . .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true"]
