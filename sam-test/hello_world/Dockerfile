# Pull the base image with python 3.12 as a runtime for your Lambda
FROM public.ecr.aws/lambda/python:3.12

# Copy the earlier created requirements.txt file to the container
COPY requirements.txt ./

# Install the python requirements from requirements.txt
RUN python3.12 -m pip install -r requirements.txt
RUN mkdir model
# Copy the earlier created app.py file to the container
COPY app.py ./
COPY text_classifier_weights.pth ./model

# Set the CMD to your handler
CMD ["app.lambda_handler"]