# sentimental-analysis
Here's a `README.md` file for your project:

```markdown
# IMDb Sentiment Classifier

A Python-based sentiment analysis pipeline using BERT embeddings and PyTorch to classify movie reviews from the IMDb dataset as positive or negative. The model is fine-tuned and deployed using FastAPI for real-time predictions, and automated with GitHub Actions for CI/CD.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [API](#api)
- [Deployment](#deployment)
- [Model Training](#model-training)
- [Technologies Used](#technologies-used)

## Project Overview
This project leverages the IMDb dataset for sentiment classification using a pre-trained BERT model. The pipeline is designed to handle text preprocessing, model training, and real-time API-based predictions. The application is deployed using FastAPI and is integrated with a CI/CD pipeline via GitHub Actions.

## Installation

### Clone the repository
```bash
git clone https://github.com/mahirbarot/imdb-sentiment-classifier.git
cd imdb-sentiment-classifier
```

### Install dependencies
```bash
pip install torch transformers datasets scikit-learn fastapi uvicorn
```

## Usage

### Train the Model
You can train the model locally using the provided dataset:
```bash
python sentiment_analysis_project.py
```
This will download the IMDb dataset, fine-tune the BERT model, and save the model as `sentiment_model.pt`.

### Run the API
To start the FastAPI server locally:
```bash
uvicorn sentiment_analysis_project:app --reload
```

Once the server is running, you can make POST requests to the `/predict/` endpoint to classify the sentiment of a movie review.

### Example API Request:
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict/' \
  -H 'Content-Type: application/json' \
  -d '{"text": "The movie was fantastic!"}'
```

### Response:
```json
{
  "text": "The movie was fantastic!",
  "sentiment": "Positive"
}
```

## Deployment

The project is set up to be deployed on Heroku or any cloud platform. GitHub Actions handle the CI/CD pipeline to automate deployment on push to the `main` branch.

### To deploy on Heroku:

1. Create an app on Heroku.
2. Add your repository to the Heroku app.
3. Push your code to deploy:
   ```bash
   git push heroku main
   ```

## Model Training

To train the model on your own, modify the sample size and batch size as needed in the `run_training` function of `sentiment_analysis_project.py`. You can tune the model by adjusting the number of epochs or hyperparameters.

## Technologies Used

- **Python**: Main programming language
- **PyTorch**: For model training and fine-tuning
- **Hugging Face Transformers**: For BERT-based embedding extraction
- **scikit-learn**: For logistic regression classifier
- **FastAPI**: For API deployment
- **GitHub Actions**: For CI/CD pipeline automation
- **Heroku**: For deployment

## License

This project is licensed under the MIT License.
```


