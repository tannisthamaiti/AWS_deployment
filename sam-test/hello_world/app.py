import json
import torch
import torch.nn as nn
import os



class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)  # Binary classification

    def forward(self, x):
        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        _, (hidden, _) = self.lstm(x)  # (batch, hidden_dim)
        out = self.fc(hidden[-1])  # Output layer
        return out
embed_dim = 8
num_filters = 16
filter_sizes = 2  # Different filter sizes for CNN
num_classes = 2
vocab_size =10
hidden_dim = 16

text_data = [
    ("hello world", 0),
    ("hi there", 0),
    ("deep learning", 1),
    ("neural network", 1),
    ("machine learning", 1),
    ("hello again", 0),
]

# Creating a simple vocabulary
vocab = sorted(set(word for sentence, _ in text_data for word in sentence.split()))
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
vocab_size = len(vocab)


# Function to encode text data
def encode_sentence(sentence):
    return [word_to_idx[word] for word in sentence.split()]

# model=model.load_state_dict(torch.load("model/text_cnn_weights.pth"))

def lambda_handler(event, context):
    body = json.loads(event["body"])

        

    file_path = "model/text_classifier_weights.pth"  # Change this to your file path
    #model = TextClassifier(vocab_size, embed_dim, hidden_dim)
    cuda_available = torch.cuda.is_available()
    #file_result=model.load_state_dict(torch.load(file_path))
    device = torch.device("cpu")
    
    try:
        #model = torch.load(file_path,weights_only=False, map_location=torch.device('cpu'))
        model = TextClassifier(vocab_size, embed_dim, hidden_dim)  # Instantiate the model
        model.load_state_dict(torch.load(file_path, map_location=torch.device("cpu")))
        model.eval()

        sentence = body['sentence']

 

    # Preprocess and encode the sentence
        encoded_sentence = torch.tensor(encode_sentence(sentence), dtype=torch.long)
      # Add batch dimension
        input_tensor = encoded_sentence.clone().detach()
        output = model(input_tensor)
        predicted_label = torch.argmax(output).item()
        return {"statusCode": 200, "body": f"The prediction is class {predicted_label}!"}
        
    except Exception as e:
        
        return {
            'statusCode': 500,
            'body': f"Runtime error: {str(e)}"
        }

  

    