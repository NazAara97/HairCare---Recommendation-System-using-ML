import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
from fuzzywuzzy import fuzz
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import traceback

# Ensure necessary NLTK data is downloaded
try:
    nltk.download('stopwords')
    nltk.download('wordnet')
except Exception as e:
    print(f"Error downloading NLTK data: {e}")

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """Clean the input text by lowercasing, removing special characters, numbers, and stopwords"""
    try:
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetic characters
        words = text.split()
        cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        return ' '.join(cleaned_words)
    except Exception as e:
        print(f"Error in cleaning text: {e}")
        return ""

# Define the path for datasets
base_path = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
dataset = os.path.join(base_path, 'dataset.csv')
dataset1 = os.path.join(base_path, 'dataset1.csv')

# Model path
model_path = r'c:\Users\DELL\Desktop\Final project test'

# Check for file existence
if not os.path.exists(dataset):
    print(f"Dataset file not found at: {dataset}")
    exit(1)
if not os.path.exists(dataset1):
    print(f"Dataset1 file not found at: {dataset1}")
    exit(1)
if not os.path.isdir(model_path):
    print(f"Model directory not found at: {model_path}")
    exit(1)

# Load the combined model (containing both LSA transformer and TF-IDF vectorizer)
combined_model_path = os.path.join(model_path, 'combined_model.pkl')
combined_model_path1 = os.path.join(model_path, 'combined_model1.pkl')
model_pk1_path = os.path.join(model_path, 'model.pk1')

try:
    combined_model = joblib.load(combined_model_path)
    lsa1 = combined_model['lsa_transformer']
    vectorizer1 = combined_model['tfidf_vectorizer']
    print("Combined model loaded successfully.")
except FileNotFoundError as e:
    print(f"File not found: {e}")
    exit(1)
except Exception as e:
    print(f"Error during model loading: {e}")
    exit(1)

try:
    combined_model1 = joblib.load(combined_model_path1)
    lsa2 = combined_model1['lsa_transformer1']
    vectorizer2 = combined_model1['tfidf_vectorizer1']
    print("Combined model1 loaded successfully.")
except FileNotFoundError as e:
    print(f"File not found: {e}")
    exit(1)
except Exception as e:
    print(f"Error during model loading: {e}")
    exit(1)

# Load model.pk1
try:
    model_pk1 = joblib.load(model_pk1_path)
    print("Model pk1 loaded successfully.")
except FileNotFoundError as e:
    print(f"File not found: {e}")
    exit(1)
except Exception as e:
    print(f"Error loading model pk1: {e}")
    exit(1)

# Preprocess function to handle conditions columns for the remedies dataset
def preprocess_conditions(df):
    condition_columns = ['condition_type1', 'condition_type2', 'condition_type3', 'condition_type4', 'condition_type5', 'condition_type6']
    df_conditions = df[condition_columns].fillna('')  # Fill NaN with empty strings
    return df_conditions.apply(lambda x: ' '.join(x), axis=1)

# Define recommendation function for chatbot with LSA
def recommend_chatbot(user_query, dataset1, vectorizer1, lsa1):
    try:
        cleaned_query = clean_text(user_query)
        if not cleaned_query:
            print("Cleaned query is empty.")
            return ["Query too vague or invalid. Please try again."]
        
        print("Cleaned Query:", cleaned_query)
        query_vec = vectorizer1.transform([cleaned_query])
        print("Query Vector Shape:", query_vec.shape)
        query_lsa = lsa1.transform(query_vec)
        print("Query LSA Shape:", query_lsa.shape)
        
        dataset_vectors = lsa1.transform(vectorizer1.transform(dataset1['User Query']))
        print("Dataset Vectors Shape:", dataset_vectors.shape)
        
        similarities = cosine_similarity(query_lsa, dataset_vectors)
        print("Similarities:", similarities.flatten())
        similar_indices = similarities.flatten().argsort()[::-1]
        
        recommended_responses = dataset1.iloc[similar_indices[:10]]['Combined Response']
        print("Recommended Responses:", recommended_responses)
        return recommended_responses.tolist()
    except Exception as e:
        print(f"Error during recommendation: {e}")
        print(traceback.format_exc())  # More detailed error reporting
        return ["An error occurred during recommendation."]

# Define doctor recommendation function for chatbot with LSA
def recommend_chatbot_doctor(doctor_name, dataset1, vectorizer2, lsa2):
    try:
        cleaned_query = clean_text(doctor_name)
        if not cleaned_query:
            print("Cleaned doctor name is empty.")
            return ["Doctor name is too vague or invalid. Please try again."]
        
        print("Cleaned Doctor Name:", cleaned_query)
        doctor_name_vec = vectorizer2.transform([cleaned_query])
        doctor_name_lsa = lsa2.transform(doctor_name_vec)
        
        dataset_vectors = lsa2.transform(vectorizer2.transform(dataset1['Doctor Name with Qualification']))
        similarities = cosine_similarity(doctor_name_lsa, dataset_vectors)
        similar_indices = similarities.flatten().argsort()[::-1]
        
        recommended_responses1 = dataset1.iloc[similar_indices[:10]]['Doctor Response']
        print("Recommended Doctor Responses:", recommended_responses1)
        return recommended_responses1.tolist()
    except Exception as e:
        print(f"Error during recommendation: {e}")
        print(traceback.format_exc())  # More detailed error reporting
        return ["An error occurred during recommendation."]

# Model class definition
class Model:
    def __init__(self, dataset, dataset1, model_path):
        
            self.dataset = pd.read_csv(dataset)
            self.dataset1 = pd.read_csv(dataset1).fillna('')

            # Preprocess the conditions column
            self.dataset.loc[:, 'conditions'] = preprocess_conditions(self.dataset)

            # Initialize the TF-IDF vectorizer for conditions
            self.vectorizer = TfidfVectorizer()

            # Calculate the TF-IDF matrix for the conditions in the dataset
            self.tfidf_matrix = self.vectorizer.fit_transform(self.dataset['conditions'])
            print("TF-IDF Matrix for conditions calculated successfully.")

            # Prepare combined response and doctor response columns
            self.dataset1['Combined Response'] = (
                self.dataset1['Chatbot Response'] + 
                ' <br><strong>Doctor Name:</strong> ' + 
                self.dataset1['Doctor Name with Qualification'] +
                '<br><br><strong>Process:</strong> ' + 
                self.dataset1['Process']
            )
            self.dataset1['Doctor Response'] = (
                self.dataset1['Doctor Name with Qualification'] + 
                ' <br><strong>Response:</strong> ' + 
                self.dataset1['Chatbot Response'] + 
                '<br><br><strong>Process:</strong> ' + 
                self.dataset1['Process']
            )

            # Initialize the LSA transformers and vectorizers
            self.vectorizer1 = vectorizer1
            self.lsa1 = lsa1
            self.vectorizer2 = vectorizer2
            self.lsa2 = lsa2
            
    def recommend_remedies(self, remedy_type, conditions, hair_type=None, age_group=None, environment_condition=None):
    # Find remedies matching remedy_type and other optional filters
        filtered_dataset = self.dataset.loc[self.dataset['remedy_type'] == remedy_type]

        # Filter by hair type, age group, and environment condition if provided
        if hair_type:
            filtered_dataset = filtered_dataset.loc[filtered_dataset['hair_type'] == hair_type]
        if age_group:
            filtered_dataset = filtered_dataset.loc[filtered_dataset['age_group'] == age_group]
        if environment_condition and 'environment_condition' in self.dataset.columns:
            filtered_dataset = filtered_dataset.loc[filtered_dataset['environment_condition'] == environment_condition]

        # Prepare conditions matching (exact and similar)
        exact_conditions_array = []
        for condition in conditions:
            matchers = filtered_dataset.loc[filtered_dataset['conditions'].str.contains(condition)]
            if not matchers.empty:
                filtered_dataset = matchers
                exact_conditions_array.append(condition)
            else:
                similar_conditions_Array = []
                for joined_condition in self.dataset['conditions']:
                    if fuzz.token_set_ratio(condition, joined_condition) > 90:
                        similar_conditions_Array.append(joined_condition)
                exact_conditions_array.append(' or '.join(similar_conditions_Array))

        # Preprocess the filtered dataset's conditions for TF-IDF
        filtered_dataset.loc[:, 'conditions'] = preprocess_conditions(filtered_dataset)

        # Calculate the TF-IDF matrix for the filtered dataset's conditions
        tfidf_matrix = self.vectorizer.transform(filtered_dataset['conditions'])

        # Concatenate user input conditions
        user_conditions = ' and '.join(exact_conditions_array)

        # Transform the user conditions into a TF-IDF matrix
        user_tfidf = self.vectorizer.transform([user_conditions])

        # Calculate the cosine similarity between user conditions and dataset conditions
        cosine_similarity_score = cosine_similarity(user_tfidf, tfidf_matrix)

        # Sort based on similarity score
        sorted_indexes = cosine_similarity_score.argsort()[0][::-1]
        sorted_dataset = filtered_dataset.iloc[sorted_indexes]

        # Get remedy names and indexes from the sorted dataset
        list_of_remedy_names = sorted_dataset['remedy_name'].tolist()
        list_of_remedy_indexes = sorted_dataset['index'].tolist()

        return list_of_remedy_indexes, list_of_remedy_names



    def recommend(self, remedy_type, conditions, hair_type=None, age_group=None, environment_condition=None):
        # Call the recommend_remedies method with additional filters
        list_of_remedy_names, list_of_remedy_indexes = self.recommend_remedies(remedy_type, conditions, hair_type, age_group, environment_condition)
        print(f"Recommended {remedy_type} remedies for {conditions}, hair_type: {hair_type}, age_group: {age_group}, environment_condition: {environment_condition}:")
        for name in list_of_remedy_names:
            print(name)

        return list_of_remedy_names, list_of_remedy_indexes


    def recommend_chatbot(self, user_query):
        return recommend_chatbot(user_query, self.dataset1, self.vectorizer1, self.lsa1)
        
    def recommend_chatbot_doctor(self, doctor_name):
        return recommend_chatbot_doctor(doctor_name, self.dataset1, self.vectorizer2, self.lsa2)

    def model_export(self, filepath):
        try:
            with open(filepath, 'wb') as f:
                joblib.dump(self, f)
            print("Model exported successfully.")
        except Exception as e:
            print(f"Error during model export: {e}")
            print(traceback.format_exc())  # More detailed error reporting

# Initialize model
model = Model(dataset, dataset1, model_path)

# Test user query
user_query = "What is the remedy for dry hair?"
print("Chatbot Recommendations:", model.recommend_chatbot(user_query))

# Test doctor query
doctor_name = "Dr. Raj Patel"
print("Doctor Recommendations:", model.recommend_chatbot_doctor(doctor_name))

# Get recommendations for specific remedy_type, conditions, hair_type, age_group, and environment_condition
remedy_names, remedy_indexes = model.recommend('shampoo', ['cleanse'], hair_type='normal', age_group='below 60', environment_condition='cold')