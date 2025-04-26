# DigitalPulse Social Media Analyzer

An event-driven system for social media opinion analysis that groups, classifies, and generates conclusions about topics.

## Overview

DigitalPulse Social Media Analyzer is a Streamlit-based application that helps analyze social media opinions. The application:

1. Allows users to add comments/opinions
2. Analyzes topics to find related opinions using semantic similarity
3. Classifies opinions using Gemini AI into categories (Claim, Counterclaim, Rebuttal, Evidence)
4. Generates conclusions based on the analyzed opinions

## Features

- **Add Comments**: Users can add new comments/opinions to the dataset
- **Topic Analysis**: Users can either enter a new topic or select an existing one for analysis
- **Semantic Similarity**: Uses sentence embeddings to find opinions related to a topic
- **Opinion Classification**: Classifies opinions using Google's Gemini AI
- **Conclusion Generation**: Generates a concise conclusion summarizing the overall sentiment

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/digitalpulse-social-media-analyzer.git
cd digitalpulse-social-media-analyzer
```

2. If you're setting up a new repository:
```bash
git init
git remote remove origin  # In case there's an existing remote
git remote add origin https://github.com/smtakn44/Social-Media-Analysis-App.git
git branch -M main
```

3. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

4. Install required dependencies:
```bash
pip install -r requirements.txt
```

5. Run the application:
```bash
streamlit run main.py
```

## Usage

1. Get a Gemini API key from Google AI Studio
2. Enter your API key in the sidebar
3. Use the "Add Comment" tab to add opinions to the dataset
4. Use the "Analyze Topic" tab to analyze topics and generate conclusions

## Project Structure

```
├── data/                # CSV files for topics, opinions, and conclusions
├── models/              # Directory for storing model files
├── src/                 # Source code
│   ├── data_processor.py    # Data processing utilities
│   ├── embedding.py         # Text embedding and similarity functions
│   └── gemini_api.py        # Gemini API integration
├── main.py              # Main Streamlit application
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation
```

## Dependencies

- streamlit
- pandas
- scikit-learn
- sentence-transformers
- google-generativeai

## License

not yet