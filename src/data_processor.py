import pandas as pd
import os

class DataProcessor:
    def __init__(self, data_path="data"):
        self.data_path = data_path
        self.topics_path = os.path.join(data_path, "topics.csv")
        self.opinions_path = os.path.join(data_path, "opinions.csv")
        self.conclusion_path = os.path.join(data_path, "conclusions.csv")
        
        # Create files if they don't exist
        self._create_files_if_not_exist()
        
        # Load data
        self.topics = self._load_data(self.topics_path)
        self.opinions = self._load_data(self.opinions_path)
        self.conclusions = self._load_data(self.conclusion_path)
    
    def _create_files_if_not_exist(self):
        # Define columns for each CSV file
        columns = ["id", "topic_id", "text", "type", "effectiveness"]
        
        # Create topics.csv if it doesn't exist
        if not os.path.exists(self.topics_path):
            df = pd.DataFrame(columns=columns)
            df.to_csv(self.topics_path, index=False)
        
        # Create opinions.csv if it doesn't exist
        if not os.path.exists(self.opinions_path):
            df = pd.DataFrame(columns=columns)
            df.to_csv(self.opinions_path, index=False)
        
        # Create conclusion.csv if it doesn't exist
        if not os.path.exists(self.conclusion_path):
            df = pd.DataFrame(columns=columns)
            df.to_csv(self.conclusion_path, index=False)
    
    def _load_data(self, file_path):
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return pd.DataFrame(columns=["id", "topic_id", "text", "type", "effectiveness"])
    
    def save_data(self):
        """Save all dataframes to their respective CSV files"""
        self.topics.to_csv(self.topics_path, index=False)
        self.opinions.to_csv(self.opinions_path, index=False)
        self.conclusions.to_csv(self.conclusion_path, index=False)
    
    def add_opinion(self, text, topic_id=None, opinion_type=None, effectiveness=None):
        """Add a new opinion to the opinions dataframe"""
        import uuid
        
        new_opinion = {
            "id": uuid.uuid4().hex[:12],
            "topic_id": topic_id,
            "text": text,
            "type": opinion_type,
            "effectiveness": effectiveness
        }
        
        self.opinions = pd.concat([self.opinions, pd.DataFrame([new_opinion])], ignore_index=True)
        self.save_data()
        return new_opinion["id"]
    
    def add_topic(self, text, topic_type="Position", effectiveness="Adequate"):
        """Add a new topic to the topics dataframe"""
        import uuid
        
        new_topic = {
            "id": uuid.uuid4().hex[:12],
            "topic_id": uuid.uuid4().hex[:12].upper(),
            "text": text,
            "type": topic_type,
            "effectiveness": effectiveness
        }
        
        self.topics = pd.concat([self.topics, pd.DataFrame([new_topic])], ignore_index=True)
        self.save_data()
        return new_topic["topic_id"]
    
    def add_conclusion(self, topic_id, text, conclusion_type="Concluding Statement", effectiveness="Adequate"):
        """Add a new conclusion to the conclusions dataframe"""
        import uuid
        
        new_conclusion = {
            "id": uuid.uuid4().hex[:12],
            "topic_id": topic_id,
            "text": text,
            "type": conclusion_type,
            "effectiveness": effectiveness
        }
        
        self.conclusions = pd.concat([self.conclusions, pd.DataFrame([new_conclusion])], ignore_index=True)
        self.save_data()
        return new_conclusion["id"]
    
    def get_topic_by_id(self, topic_id):
        """Get a topic by its topic_id"""
        return self.topics[self.topics["topic_id"] == topic_id]
    
    def get_opinions_by_topic_id(self, topic_id):
        """Get all opinions for a specific topic"""
        return self.opinions[self.opinions["topic_id"] == topic_id]
    
    def get_conclusion_by_topic_id(self, topic_id):
        """Get conclusion for a specific topic"""
        return self.conclusions[self.conclusions["topic_id"] == topic_id]
    
    def update_opinion_metadata(self, opinion_id, topic_id, opinion_type, effectiveness="Adequate"):
        """Update metadata for an opinion"""
        opinion_idx = self.opinions[self.opinions["id"] == opinion_id].index
        if len(opinion_idx) > 0:
            self.opinions.loc[opinion_idx[0], "topic_id"] = topic_id
            self.opinions.loc[opinion_idx[0], "type"] = opinion_type
            self.opinions.loc[opinion_idx[0], "effectiveness"] = effectiveness
            self.save_data()
            return True
        return False