# Simple Gemini Chat Application

A simple chat application built with Flask and Google's Gemini API.

## Features

- Simple and clean user interface
- Real-time chat with Gemini AI
- Markdown support for code blocks and formatting
- Conversation memory within a session

## Prerequisites

- Python 3.7 or higher
- Flask
- Google Generative AI Python SDK

## Installation

1. Clone this repository or download the files

2. Install the required packages:
```
pip install flask google-generativeai
```

3. Set up your Gemini API key:
   - The API key is already configured in the app.py file
   - If you want to use your own API key, replace the value of `GEMINI_API_KEY` in app.py

## Running the Application

1. Navigate to the project directory:
```
cd path/to/gemini_chat_app
```

2. Run the Flask application:
```
python app.py
```

3. Open your web browser and go to:
```
http://127.0.0.1:5000/
```

## Usage

- Type your message in the input field at the bottom of the chat window
- Press Enter or click the Send button to send your message
- The AI will respond with its answer
- The conversation history is maintained during your session

## Customization

- You can modify the UI by editing the CSS in the static/style.css file
- You can change the Gemini model by updating the model name in app.py

## Troubleshooting

### Model Not Found Errors

If you encounter model-related errors like "404 models/gemini-pro is not found", try the following:

1. Check available models by visiting:
```
http://127.0.0.1:5000/models
```

2. Update the model name in app.py to use one of the available models listed

3. The application includes fallback mechanisms that will try different model versions if the primary model fails

Common model names to try:
- gemini-1.5-flash
- gemini-1.5-flash-latest
- gemini-1.5-flash-8b

### API Quota Exceeded

If you see a "429 Quota Exceeded" error, it means you've reached the free usage limit for the Gemini API. To resolve this:

1. **Wait and try again later**: Google refreshes the quota periodically
2. **Get your own API key**:
   - Visit [Google AI Studio](https://ai.google.dev/)
   - Create an account and get a free API key
   - Replace the API key in app.py with your own key

3. **Use a different model**: Some models have higher quota limits than others. Try using:
   - gemini-1.5-flash
   - gemini-1.5-flash-8b

## License

This project is open source and available under the MIT License.

## Acknowledgements

- Google for providing the Gemini API
- Flask for the web framework
- Bootstrap for the UI components