# Fake News Detection System (NLP + Docker)

End-to-end machine learning project for detecting fake news articles using
a transformer-based sentence embedding model (E5 + SetFit).

The system includes:
- A Flask REST API for inference
- A Shiny web interface
- A pretrained model hosted on Hugging Face
- Fully dockerised setup for easy local execution

---

## 🚀 Quick Start (

### Requirements
- Docker
- Docker Compose

Frontend UI: http://localhost:8000

Backend API: http://localhost:5000

### Run locally
```bash
git clone https://github.com/yourname/Fake-News-Finder.git
cd Fake-News-Finder
docker compose up --build

