import streamlit as st
import pandas as pd
import os
import time
from src.data_processor import DataProcessor
from src.embedding import EmbeddingProcessor
from src.gemini_api import GeminiAPI

# Set page configuration
st.set_page_config(
    page_title="DigitalPulse Social Media Analysis",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if "data_processor" not in st.session_state:
    st.session_state.data_processor = DataProcessor(data_path="data")

if "embedding_processor" not in st.session_state:
    st.session_state.embedding_processor = EmbeddingProcessor(cache_dir="models")

# Sidebar for API key
with st.sidebar:
    st.title("DigitalPulse")
    st.subheader("Social Media Analysis")
    
    # Input for Gemini API key
    api_key = st.text_input("Enter Gemini API Key:", type="password")
    
    if api_key:
        st.session_state.gemini_api = GeminiAPI(api_key)
        st.success("API key set!")
    else:
        st.warning("Please enter a Gemini API key to use classification and conclusion generation.")
    
    # Show data statistics
    st.subheader("Dataset Statistics")
    st.write(f"Topics: {len(st.session_state.data_processor.topics)}")
    st.write(f"Opinions: {len(st.session_state.data_processor.opinions)}")
    st.write(f"Conclusions: {len(st.session_state.data_processor.conclusions)}")

# Main content
st.title("DigitalPulse Social Media Analyzer")

# Choose between adding a comment or analyzing a topic
tab1, tab2 = st.tabs(["Add Comment", "Analyze Topic"])

with tab1:
    st.header("Add New Comment/Opinion")
    
    new_comment = st.text_area("Enter your comment/opinion:", height=150)
    
    if st.button("Submit Comment"):
        if new_comment:
            with st.spinner("Processing comment..."):
                # Add the comment to the dataset
                opinion_id = st.session_state.data_processor.add_opinion(
                    text=new_comment,
                    topic_id=None,
                    opinion_type=None,
                    effectiveness=None
                )
                st.success(f"Comment added successfully with ID: {opinion_id}")
        else:
            st.error("Please enter a comment.")

with tab2:
    st.header("Analyze Topic")
    
    # Option to enter a topic or select existing one
    analysis_option = st.radio("Select option:", ["Enter a topic", "Select topic"])
    
    if analysis_option == "Enter a topic":
        topic_text = st.text_area("Enter topic text:", height=150)
        
        if st.button("Analyze Topic"):
            if topic_text and 'gemini_api' in st.session_state:
                with st.spinner("Analyzing topic..."):
                    # Find related opinions without adding to topics.csv
                    processor = st.session_state.embedding_processor
                    data_processor = st.session_state.data_processor
                    
                    # Get all opinions
                    opinions_texts = data_processor.opinions["text"].tolist()
                    opinion_ids = data_processor.opinions["id"].tolist()
                    
                    if opinions_texts:
                        # Find related opinions
                        found_opinions, similarity_scores, relevant_idxs = processor.find_related_opinions(
                            topic_text, opinions_texts, threshold=0.85
                        )
                        
                        # Sort the opinions by similarity score (highest first)
                        sorted_indices = sorted(range(len(similarity_scores)), key=lambda i: similarity_scores[i], reverse=True)
                        
                        # Take only the top 7 opinions (or fewer if less than 7 are found)
                        top_indices = sorted_indices[:min(7, len(sorted_indices))]
                        
                        # Get the top opinions and their indices
                        top_opinions = [found_opinions[i] for i in top_indices]
                        top_relevant_idxs = [relevant_idxs[i] for i in top_indices]
                        top_scores = [similarity_scores[i] for i in top_indices]
                        
                        st.subheader("Topic:")
                        st.write(topic_text)
                        
                        if top_opinions:
                            related_opinions_with_types = []
                            
                            st.subheader("Related Opinions:")
                            
                            for i, (opinion, score, idx) in enumerate(zip(top_opinions, top_scores, top_relevant_idxs)):
                                # Classify the opinion
                                opinion_type = st.session_state.gemini_api.classify_opinion(opinion)
                                
                                # Display in the required format
                                st.write(f"Related Opinion {i+1} ({opinion_type})- {opinion}")
                                
                                related_opinions_with_types.append((opinion, opinion_type))
                                
                                # Add a delay between API calls to avoid rate limits
                                time.sleep(1)
                            
                            # Generate conclusion
                            st.subheader("Conclusion")
                            conclusion = st.session_state.gemini_api.generate_conclusion(
                                topic_text, related_opinions_with_types
                            )
                            
                            st.write(f"Conclusion- {conclusion}")
                        else:
                            st.info("No related opinions found. Add more opinions to the dataset.")
                    else:
                        st.info("No opinions in the dataset. Add some opinions first.")
            elif not 'gemini_api' in st.session_state:
                st.error("Please add a Gemini API key in the sidebar.")
            else:
                st.error("Please enter a topic.")
    
    else:  # Select topic
        # Get unique topic IDs and texts
        topics = st.session_state.data_processor.topics
        
        if not topics.empty:
            topic_options = [(row["topic_id"], f"{row['topic_id']}: {row['text'][:50]}...") 
                            for _, row in topics.iterrows()]
            
            # Create a dictionary for the selectbox
            topic_dict = {text: id for id, text in topic_options}
            
            selected_topic_text = st.selectbox(
                "Select a topic:",
                options=list(topic_dict.keys())
            )
            
            selected_topic_id = topic_dict[selected_topic_text]
            
            if st.button("Analyze Topic"):
                with st.spinner("Analyzing topic..."):
                    # Get topic details
                    topic_row = st.session_state.data_processor.get_topic_by_id(selected_topic_id)
                    
                    if not topic_row.empty:
                        topic_text = topic_row["text"].values[0]
                        
                        st.subheader("Topic:")
                        st.write(topic_text)
                        
                        # Get all opinions to analyze with embeddings
                        processor = st.session_state.embedding_processor
                        data_processor = st.session_state.data_processor
                        
                        # Get all opinions
                        opinions_texts = data_processor.opinions["text"].tolist()
                        opinion_ids = data_processor.opinions["id"].tolist()
                        
                        if opinions_texts:
                            # Find related opinions
                            found_opinions, similarity_scores, relevant_idxs = processor.find_related_opinions(
                                topic_text, opinions_texts, threshold=0.85
                            )
                            
                            # Sort the opinions by similarity score (highest first)
                            sorted_indices = sorted(range(len(similarity_scores)), key=lambda i: similarity_scores[i], reverse=True)
                            
                            # Take only the top 7 opinions (or fewer if less than 7 are found)
                            top_indices = sorted_indices[:min(7, len(sorted_indices))]
                            
                            # Get the top opinions and their indices
                            top_opinions = [found_opinions[i] for i in top_indices]
                            top_relevant_idxs = [relevant_idxs[i] for i in top_indices]
                            top_scores = [similarity_scores[i] for i in top_indices]
                            
                            if top_opinions:
                                related_opinions_with_types = []
                                
                                st.subheader("Related Opinions:")
                                
                                for i, (opinion, score, idx) in enumerate(zip(top_opinions, top_scores, top_relevant_idxs)):
                                    # Classify the opinion
                                    opinion_type = st.session_state.gemini_api.classify_opinion(opinion)
                                    
                                    # Display in the required format
                                    st.write(f"Related Opinion {i+1} ({opinion_type})- {opinion}")
                                    
                                    related_opinions_with_types.append((opinion, opinion_type))
                                    
                                    # Add a delay between API calls to avoid rate limits
                                    time.sleep(1)
                                
                                # Generate new conclusion but don't save it
                                st.subheader("New Generated Conclusion")
                                new_conclusion = st.session_state.gemini_api.generate_conclusion(
                                    topic_text, related_opinions_with_types
                                )
                                
                                st.write(f"Conclusion- {new_conclusion}")
                                
                                # Display existing conclusion from CSV if it exists
                                existing_conclusion = data_processor.get_conclusion_by_topic_id(selected_topic_id)
                                if not existing_conclusion.empty:
                                    st.subheader("Existing Conclusion from CSV")
                                    st.write(f"Conclusion- {existing_conclusion['text'].values[0]}")
                                else:
                                    st.info("No existing conclusion found in CSV for this topic.")
                            else:
                                st.info("No related opinions found. Add more opinions to the dataset.")
                        else:
                            st.info("No opinions in the dataset. Add some opinions first.")
                    else:
                        st.error("Topic not found.")
        else:
            st.info("No topics in the dataset. Create a new topic first.")

# Footer
st.markdown("---")
st.caption("DigitalPulse Social Media Analysis Tool - Event Driven System")