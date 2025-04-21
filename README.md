# 🚀 AWS SAM Deployment – Neural Network Model (Summary)

This project provides a **concise walkthrough** to deploy a simple LSTM-based neural network model using the **AWS Serverless Application Model (SAM)**. The model is exposed via an **API Gateway**, and runs inference inside a **Lambda function** using a Dockerized container.

## 🧠 What’s Inside

This folder includes:
- A trained `TextClassifier` PyTorch model
- Lambda handler (`app.py`) to serve predictions
- Dockerfile to build the serverless deployment package
- `requirements.txt` with Python dependencies
- Instructions for building and deploying via SAM

## 📝 Full Documentation

For the **complete setup**, including:
- Prerequisites
- Folder structure
- Code explanation (model, handler, encoding)
- Dockerfile and deployment flow
- `sam deploy --guided` configuration steps

👉 Please refer to the original README at:

🔗 [https://github.com/tannisthamaiti/AWS_deployment/tree/main/sam-test](https://github.com/tannisthamaiti/AWS_deployment/tree/main/sam-test)

## 📬 Contact & Contributions

This is part of a learning series from **@liveaiAIClub**. PRs and forks are welcome if you're experimenting with serverless ML!
