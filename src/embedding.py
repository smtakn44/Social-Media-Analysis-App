from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

class EmbeddingProcessor:
    def __init__(self, model_name="all-MiniLM-L6-v2", cache_dir="models"):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = None
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
    
    def load_model(self):
        """Load the sentence transformer model"""
        if self.model is None:
            print(f"Loading model {self.model_name}...")
            self.model = SentenceTransformer(self.model_name, cache_folder=self.cache_dir)
            print("Model loaded.")
        return self.model
    
    def encode_texts(self, texts):
        """Encode a list of texts into embeddings"""
        model = self.load_model()
        embeddings = model.encode(texts, show_progress_bar=True)
        print(f"{len(embeddings)} examples, each with {embeddings.shape[1]} dimensions ready.")
        return embeddings
    
    def get_related_opinions(self, topic_idx, topic_embeddings, opinion_embeddings, topics_text, opinions_text, threshold=0.85):
        """Find opinions related to a topic based on cosine similarity"""
        # Get the topic embedding
        topic_embedding = topic_embeddings[topic_idx]
        
        # Calculate cosine similarity between topic and all opinions
        sims = cosine_similarity([topic_embedding], opinion_embeddings)[0]
        
        # Find indices of opinions that exceed the threshold
        relevant_idxs = [i for i, s in enumerate(sims) if s > threshold]
        
        # Get the texts of relevant opinions
        found_opinions = [opinions_text[i] for i in relevant_idxs]
        
        # Get the similarity scores
        similarity_scores = [sims[i] for i in relevant_idxs]
        
        # Get the topic text
        topic_text = topics_text[topic_idx]
        
        return topic_text, found_opinions, similarity_scores, relevant_idxs
    
    def find_related_opinions(self, topic_text, opinions_texts, threshold=0.85):
        """Find opinions related to a given topic text"""
        # Encode the topic and opinions
        topic_embedding = self.encode_texts([topic_text])[0]
        opinion_embeddings = self.encode_texts(opinions_texts)
        
        # Calculate similarities
        sims = cosine_similarity([topic_embedding], opinion_embeddings)[0]
        
        # Find relevant opinions
        relevant_idxs = [i for i, s in enumerate(sims) if s > threshold]
        found_opinions = [opinions_texts[i] for i in relevant_idxs]
        similarity_scores = [sims[i] for i in relevant_idxs]
        
        return found_opinions, similarity_scores, relevant_idxs