# Fake News Detection (NLP · Docker)

End-to-end fake news detection system using sentence embeddings (**E5 + SetFit**).
Given a news article (title + text), the system predicts **REAL** or **FAKE**.

The system includes:
- A Flask REST API for inference
- A Shiny web interface
- A pretrained model hosted on Hugging Face (downloaded at runtime)
- Fully dockerised setup for easy local execution

---

## Dataset

Trained on the Kaggle *Fake News Detection Dataset*  
https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets/data

- Used for training only
- Not redistributed in this repository

### Requirements
- Docker
- Docker Compose

Frontend UI: http://localhost:8000

Backend API: http://localhost:5000

### Entire system runs locally via Docker Compose:
```bash
git clone https://github.com/askevis/Fake-News-Finder.git
cd Fake-News-Finder/Fake-News-Docker
docker-compose up --build



