import google.generativeai as genai
import json

class GeminiAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
    
    def classify_opinion(self, opinion_text):
        """Classify an opinion as Claim, Counterclaim, Rebuttal, or Evidence"""
        prompt = f"""Classify the following text into one of these categories:
- Claim: A statement that supports a position
- Counterclaim: A statement that counters another claim or presents an opposing reason
- Rebuttal: A statement that counters a counterclaim
- Evidence: Ideas or examples that support claims, counterclaims, or rebuttals

Text to classify: "{opinion_text}"

Return ONLY the classification name (Claim, Counterclaim, Rebuttal, or Evidence) without any explanations.
"""
        response = self.model.generate_content(prompt)
        classification = response.text.strip()
        
        # Ensure the response is one of the valid categories
        valid_categories = ["Claim", "Counterclaim", "Rebuttal", "Evidence"]
        if classification not in valid_categories:
            # Try to extract from longer response
            for category in valid_categories:
                if category in classification:
                    classification = category
                    break
            else:
                classification = "Claim"  # Default
        
        return classification
    
    def generate_conclusion(self, topic_text, opinions_with_types):
        """Generate a conclusion based on topic and classified opinions"""
        prompt = f"""I'm analyzing a topic and related opinions from social media. Based on the topic and the various opinions, generate a concise conclusion that summarizes the overall sentiment and key points.

Topic: {topic_text}

Opinions:
"""
        for i, (opinion, opinion_type) in enumerate(opinions_with_types):
            prompt += f"{opinion_type}: {opinion}\n"
        
        prompt += """
Please create a concise conclusion that:
1. Summarizes the overall sentiment or consensus
2. Acknowledges different perspectives if they exist
3. Highlights the most significant points
4. Is written in a neutral, analytical tone
5. Is approximately 2-3 sentences long

Format your response as just the conclusion without any additional explanations.
"""
        
        response = self.model.generate_content(prompt)
        conclusion = response.text.strip()
        
        return conclusion