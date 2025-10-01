from IPython import get_ipython
from IPython.display import display
# %%
# Install required packages
!pip install fuzzywuzzy
!pip install sklearn
!pip install pandas

from IPython.display import display, FileLink
import pandas as pd
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity
import joblib  

# Load dataset
dataset = pd.read_csv('dataset.csv')

# Preprocess the conditions
def preprocess(dataset):
    condition_columns = ['condition_type1', 'condition_type2', 'condition_type3', 'condition_type4', 'condition_type5', 'condition_type6']
    df_conditions = dataset[condition_columns].fillna('')
    # Ensure the output is a Series with the correct index
    # Changed from apply to transform to return a Series
    return df_conditions.apply(lambda x: ' '.join(x.astype(str)), axis=1)
class Model:
    def __init__(self, dataset_path):
        # Load the dataset
        self.dataset = pd.read_csv(dataset_path)
        self.dataset['conditions'] = preprocess(self.dataset)

        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=1000)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.dataset['conditions'])
    
    def recommend_remedies(self, remedy_type, conditions, hair_type=None, age_group=None, environment_condition=None):
        # Filter dataset by remedy type
        filtered_dataset = self.dataset[self.dataset['remedy_type'] == remedy_type]

        # Filter by optional parameters
        if hair_type:
            filtered_dataset = filtered_dataset[filtered_dataset['hair_type'] == hair_type]
        if age_group:
            filtered_dataset = filtered_dataset[filtered_dataset['age_group'] == age_group]
        if environment_condition:
            filtered_dataset = filtered_dataset[filtered_dataset['environment_condition'] == environment_condition]

        # Check if the filtered dataset is empty
        if filtered_dataset.empty:
            print("No remedies found for the given criteria.")  # Or raise an exception
            return [], []  # Return empty lists if no remedies are found

        exact_conditions_array = []
        for condition in conditions:
            matchers = filtered_dataset[filtered_dataset['conditions'].str.contains(condition)]
            if not matchers.empty:
                filtered_dataset = matchers
                exact_conditions_array.append(condition)
            else:
                similar_conditions_array = [
                    joined_condition for joined_condition in self.dataset['conditions']
                    if fuzz.token_set_ratio(condition, joined_condition) > 80
                ]
                exact_conditions_array.append(' or '.join(similar_conditions_array))

        filtered_dataset['conditions'] = preprocess(filtered_dataset)

        # TF-IDF transformation
        tfidf_matrix = self.vectorizer.transform(filtered_dataset['conditions'])

        # Compute similarity
        user_conditions = ' and '.join(exact_conditions_array)
        user_tfidf = self.vectorizer.transform([user_conditions])
        cosine_similarity_score = cosine_similarity(user_tfidf, tfidf_matrix)

        # Sort results
        sorted_indexes = cosine_similarity_score.argsort()[0][::-1]
        sorted_dataset = filtered_dataset.iloc[sorted_indexes]

        return sorted_dataset['index'].tolist(), sorted_dataset['remedy_name'].tolist()

        

    def recommend(self, remedy_type, conditions, hair_type=None, age_group=None, environment_condition=None):
        remedy_indexes, remedy_names = self.recommend_remedies(remedy_type, conditions, hair_type, age_group, environment_condition)
        print(f"Recommended {remedy_type} remedies for {conditions}, hair_type: {hair_type}, age_group: {age_group}, environment_condition: {environment_condition}:")
        for name in remedy_names:
            print(name)
        return remedy_names, remedy_indexes

    def train_model(self):
        X = self.dataset['conditions']
        y = self.dataset['remedy_name']

        # Drop NaNs
        self.dataset.dropna(subset=['conditions', 'remedy_name'], inplace=True)
        X, y = self.dataset['conditions'], self.dataset['remedy_name']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # TF-IDF transformation
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)

        # Train model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_tfidf, y_train)

        # Predictions & Evaluation
        y_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)

        print(f"Logistic Regression Accuracy: {accuracy * 100:.2f}%")
        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall: {recall * 100:.2f}%")

        return model

    def evaluate_model(self, model):
        X = self.dataset['conditions']
        y = self.dataset['remedy_name']
        X_tfidf = self.vectorizer.transform(X)

        cross_val_accuracy = cross_val_score(model, X_tfidf, y, cv=5, scoring='accuracy').mean()
        cross_val_precision = cross_val_score(model, X_tfidf, y, cv=5, scoring='precision_weighted', error_score='raise').mean()
        cross_val_recall = cross_val_score(model, X_tfidf, y, cv=5, scoring='recall_weighted', error_score='raise').mean()

        print(f"Cross-Validation Accuracy: {cross_val_accuracy * 100:.2f}%")
        print(f"Cross-Validation Precision: {cross_val_precision * 100:.2f}%")
        print(f"Cross-Validation Recall: {cross_val_recall * 100:.2f}%")

# Create model instance and train
model = Model('dataset.csv')
trained_model = model.train_model()

# Evaluate model
model.evaluate_model(trained_model)

# Save trained model
joblib.dump(trained_model, 'model.pk1')
print("Model saved as model.pk1")

# Generate a download link for model
FileLink(r'model.pk1')

# Recommendation example
remedy_names, remedy_indexes = model.recommend('treatment', ['frizz'], hair_type='oily', age_group='above 60', environment_condition='cold')

# **Test Set Evaluation**

test_set = {

    1: {
        'remedy_type': 'shampoo',
        'conditions': ['hair loss', 'dandruff'],
        'hair_type': 'normal',
        'age_group': 'below 60',
        'environment_condition': 'normal',
        'actual_indices': [2,3]
    },
    2: {
        'remedy_type': 'serum',
        'conditions': ['thinning', 'growth'],
        'hair_type': 'dry',
        'age_group': 'below 60',
        'environment_condition': 'normal',
        'actual_indices': [951]
    },
    3: {
        'remedy_type': 'mask',
        'conditions': ['growth', 'thinning'],
        'hair_type': 'dry',
        'age_group': 'below 60',
        'environment_condition': 'cold',
        'actual_indices': [736, 738]
    },
    4: {
        'remedy_type': 'oil',
        'conditions': ['dandruff'],
        'hair_type': 'oily',
        'age_group': 'above 60',
        'environment_condition': 'cold',
        'actual_indices': [588,589]
    },
    5: {
        'remedy_type': 'conditioner',
        'conditions': ['moisturize', 'cleanse'],
        'hair_type': 'normal',
        'age_group': 'below 60',
        'environment_condition': 'normal',
        'actual_indices': [1211, 1212]
    },
    6: {
        'remedy_type': 'serum',
        'conditions': ['thinning', 'dandruff'],
        'hair_type': 'dry',
        'age_group': 'above 60',
        'environment_condition': 'normal',
        'actual_indices': [1062,1058]
    },
    7: {
        'remedy_type': 'shampoo',
        'conditions': ['thinng'],
        'hair_type': 'dry',
        'age_group': 'above 60',
        'environment_condition': 'hot',
        'actual_indices': [290,291]
    },
    8: {
        'remedy_type': 'oil',
        'conditions': ['hairloss', 'thinning'],
        'hair_type': 'dry',
        'age_group': 'above 60',
        'environment_condition': 'normal',
        'actual_indices': [458, 466]
    },
    9: {
        'remedy_type': 'mask',
        'conditions': ['hairloss', 'thinning'],
        'hair_type': 'normal',
        'age_group': 'above 60',
        'environment_condition': 'hot',
        'actual_indices': [819, 820]
    },
    10: {
        'remedy_type': 'conditioner',
        'conditions': ['growth'],
        'hair_type': 'normal',
        'age_group': 'below 60',
        'environment_condition': 'hot',
        'actual_indices': [1217,1218]
    },
    11: {
        'remedy_type': 'serum',
        'conditions': ['hair loss'],
        'hair_type': 'dry',
        'age_group': 'above 60',
        'environment_condition': 'hot',
        'actual_indices': [1067,1068]
    },
    12: {
        'remedy_type': 'shampoo',
        'conditions': ['frizz', 'dandruff'],
        'hair_type': 'dry',
        'age_group': 'above 60',
        'environment_condition': 'hot',
        'actual_indices': [280,284]
    },
    13: {
        'remedy_type': 'oil',
        'conditions': ['moisturize'],
        'hair_type': 'oily',
        'age_group': 'above 60',
        'environment_condition': 'cold',
        'actual_indices': [589,590]
    },
    14: {
        'remedy_type': 'mask',
        'conditions': ['thinning'],
        'hair_type': 'dry',
        'age_group': 'above 60',
        'environment_condition': 'cold',
        'actual_indices': [890,891]
    },
}

correct_user_count, incorrect_user_count = 0, 0
all_actual, all_predicted = [], []

for user, data in test_set.items():
    actual_indices = set(data['actual_indices'])
    predicted_indices, _ = model.recommend_remedies(data['remedy_type'], data['conditions'], data['hair_type'], data['age_group'], data['environment_condition'])
    predicted_indices = set(predicted_indices)

    # Limit predicted indices to the relevant remedy type for a fair comparison
    filtered_actual_indices = [idx for idx in actual_indices if model.dataset.iloc[idx]['remedy_type'] == data['remedy_type']]
    filtered_predicted_indices = [idx for idx in predicted_indices if model.dataset.iloc[idx]['remedy_type'] == data['remedy_type']]
    
    # Ensure both lists have elements and are of the SAME LENGTH before extending
    num_elements = min(len(filtered_actual_indices), len(filtered_predicted_indices))  # Get the minimum length
    
    # Use only the first 'num_elements' elements from each list
    all_actual.extend(filtered_actual_indices[:num_elements])  
    all_predicted.extend(filtered_predicted_indices[:num_elements])  

    if all([idx in predicted_indices for idx in actual_indices]):
        correct_user_count += 1
    else:
        incorrect_user_count += 1

# Check if all_actual and all_predicted have elements before calculating metrics
if all_actual and all_predicted:  # Check if both lists are not empty
    accuracy = correct_user_count / (correct_user_count + incorrect_user_count)
    precision = precision_score(all_actual, all_predicted, average='weighted', zero_division=0)
    recall = recall_score(all_actual, all_predicted, average='weighted', zero_division=0)

    print(f"Test Set Accuracy: {accuracy * 100:.2f}%")
    print(f"Test Set Precision: {precision * 100:.2f}%")
    print(f"Test Set Recall: {recall * 100:.2f}%")
else:
    print("Test set evaluation skipped due to empty actual or predicted lists.") 