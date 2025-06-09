# LearningBuddy - 29 May Update

Learning buddy to teach medical residents how to treat patients with MDD

# Setting up Environment and installing required libraries

# Create virtual environment

python -m venv myenv
myenv\Scripts\activate

# Install packages

pip install streamlit langgraph langchain langchain-core langchain-openai pydantic openai elevenlabs markdown2

# Set Eleven Labs API Key as an Environment Variable

set ELEVEN_API_KEY="your_elevenLabs_api_key_here"

# Run app using the following command:

streamlit run streamlit_app.py
