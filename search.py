import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import json
from sentence_transformers import SentenceTransformer, util
import time

nltk.download('punkt')
nltk.download('stopwords')

# # pre-trained Sentence-BERT model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# def preprocess(text):
#     """
#     Preprocess the input text by tokenizing, converting to lowercase,
#     removing stop words and non-alphanumeric tokens.
#     """
#     # Tokenize and convert to lowercase
#     tokens = word_tokenize(text.lower())
#     # Define stop words
#     stop_words = set(stopwords.words('english'))
#     # Filter tokens
#     filtered = [word for word in tokens if word.isalnum() and word not in stop_words]
    
#     # Remove generic words
#     toremove = ['mg', 'ml', 'oral', 'tablet', 'tablets', 'capsule', 'capsules', 'solution', 
#                 'suspension', 'injection', 'injections', 'inhalation', 'inhaler', 'inhalers', 
#                 'drug', 'drugs', 'medication', 'medications', 'medicine', 'medicines', 'treatment', 
#                 'treatments', 'therapy', 'therapies', 'dose', 'doses', 'dosage', 'dosages', 
#                 'administration']
    
#     filtered = [word for word in filtered if word not in toremove]
    
#     return ' '.join(filtered)

# def search_csv(file_path, user_query, similarity_threshold=0.75):
#     """
#     Perform semantic search using cosine similarity in the CSV file for rows matching the user query.

#     Parameters:
#     - file_path: Path to the CSV file.
#     - user_query: The search query input by the user.
#     - similarity_threshold: Minimum similarity score required to include a result.

#     Returns:
#     - A JSON-formatted string with matching rows and their details.
#     """
#     # Load the CSV file
#     try:
#         df = pd.read_csv(file_path)
#     except Exception as e:
#         return json.dumps({"error": str(e)})

#     # Preprocess and encode the user query
#     processed_query = preprocess(user_query)
#     query_embedding = model.encode(processed_query, convert_to_tensor=True)

#     matching_rows = []

#     # Iterate over DataFrame rows
#     for index, row in df.iterrows():
#         # Concatenate relevant string-type columns for semantic searching
#         row_text = ' '.join([
#             str(row.get('Organization_Name', '')),
#             str(row.get('Product_Name', '')),
#             str(row.get('Ingredients', '')),
#             str(row.get('Territory_Code', '')),
#             str(row.get('Routes_of_Administration', '')),
#             str(row.get('Warnings', '')),
#             str(row.get('Clinical_Pharmacology', '')),
#             str(row.get('Indications_and_Usage', ''))
#         ])
#         # Preprocess and encode the row text
#         processed_row_text = preprocess(row_text)
#         row_embedding = model.encode(processed_row_text, convert_to_tensor=True)

#         # Calculate cosine similarity
#         similarity = util.pytorch_cos_sim(query_embedding, row_embedding).item()

#         # Check if similarity is above the threshold
#         if similarity >= similarity_threshold:
#             row_dict = {
#                 "unique_key": row.get('unique_key', ''),
#                 "Organization_Name": row.get('Organization_Name', ''),
#                 "Product_Name": row.get('Product_Name', ''),
#                 "Ingredients": row.get('Ingredients', ''),
#                 "Territory_Code": row.get('Territory_Code', ''),
#                 "Routes_of_Administration": row.get('Routes_of_Administration', ''),
#                 "Warnings": row.get('Warnings', ''),
#                 "Clinical_Pharmacology": row.get('Clinical_Pharmacology', None),
#                 "Indications_and_Usage": row.get('Indications_and_Usage', None),
#                 "index": index,
#                 "similarity": similarity
#             }
#             matching_rows.append(row_dict)

#     return json.dumps(matching_rows, indent=2)

# if __name__ == "__main__":
#     csv_files = [
#         r'C:\Users\nirmiti.deshmukh\ML_Tagging\Output_Files\dm_spl_release_human_rx_part2.csv',
#         r'C:\Users\nirmiti.deshmukh\ML_Tagging\Output_Files\dm_spl_release_human_rx_part3.csv',
#         r'C:\Users\nirmiti.deshmukh\ML_Tagging\Output_Files\dm_spl_release_human_rx_part4.csv',
#         r'C:\Users\nirmiti.deshmukh\ML_Tagging\Output_Files\dm_spl_release_human_rx_part5.csv'
#     ]
    
#     user_query = input("Enter your search query (e.g., 'lactose monohydrate medicines'): ")

#     all_results = []

#     for file in csv_files:
#         result = search_csv(file, user_query)
#         all_results.append(json.loads(result))

#     with open('search_results.json', 'w') as f:
#         json.dump(all_results, f, indent=2)

#     print(f"Search completed. Results saved to 'search_results.json'.")


#     #semantic search mein change krna h rather than keyword 



# Load pre-trained Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

def preprocess(text):
    """
    Preprocess the input text by tokenizing, converting to lowercase,
    removing stop words and non-alphanumeric tokens.
    """
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered = [word for word in tokens if word.isalnum() and word not in stop_words]
    
    # Remove generic words that are not helpful for matching
    toremove = ['mg', 'ml', 'oral', 'tablet', 'capsule', 'solution', 
                'suspension', 'injection', 'drug', 'medicine', 'treatment', 
                'therapy', 'dose', 'administration']
    filtered = [word for word in filtered if word not in toremove]
    
    return ' '.join(filtered)

def search_csv(file_path, user_query, top_k=5, similarity_threshold=0.75):
    """
    Perform semantic search in the CSV file for rows matching the user query.

    Parameters:
    - file_path: Path to the CSV file.
    - user_query: The search query input by the user.
    - top_k: Number of top matching results to consider.
    - similarity_threshold: Minimum similarity score required to include a result.

    Returns:
    - A JSON-formatted string with matching rows and their details.
    """
    start_time = time.time()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting search...")

    # Preprocess and encode the user query
    processed_query = preprocess(user_query)
    query_embedding = model.encode(processed_query, convert_to_tensor=True)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Query encoded.")

    matching_rows = []

    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] CSV file loaded.")

        # Iterate over DataFrame rows to encode and find similarities
        for index, row in df.iterrows():
            row_text = ' '.join([
                str(row.get('Public title', '')),
                str(row.get('Condition', '')),
                str(row.get('Developmental phase', '')),
                str(row.get('Type of intervention', '')),
                str(row.get('Organization', '')),
                str(row.get('Primary outcomes', ''))
            ])
            processed_row_text = preprocess(row_text)
            row_embedding = model.encode(processed_row_text, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(query_embedding, row_embedding).item()

            if similarity >= similarity_threshold:
                row_dict = {
                    "unique_key": row.get('unique_key', ''),
                    "Public_title": row.get('Public title', ''),
                    "Condition": row.get('Condition', ''),
                    "Developmental_phase": row.get('Developmental phase', ''),
                    "Type_of_intervention": row.get('Type of intervention', ''),
                    "Organization": row.get('Organization', ''),
                    "Primary_outcomes": row.get('Primary outcomes', ''),
                    "index": index,
                    "similarity": similarity
                }
                matching_rows.append(row_dict)

    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error occurred: {str(e)}")
        return json.dumps({"error": str(e)})

    end_time = time.time()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Finished processing. Time taken: {end_time - start_time:.2f} seconds.")

    # Sort and select top results
    matching_rows = sorted(matching_rows, key=lambda x: x['similarity'], reverse=True)[:top_k]

    # Convert the list of dictionaries to JSON format
    return json.dumps(matching_rows, indent=2)

if __name__ == "__main__":
    # Define paths of processed CSV files
    csv_files = [
        r'C:\Users\nirmiti.deshmukh\ML_Tagging\japan_data\japan_tagged\Japan_Clinical_Trial_Dataset 1_tagged.csv'
    ]
    
    user_query = input("Enter your search query (e.g., 'cancer treatment using monoclonal antibodies'): ")

    all_results = []
    total_start_time = time.time()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting full process...")

    # Perform search in each CSV file
    for file in csv_files:
        result = search_csv(file, user_query, similarity_threshold=0.75)
        all_results.append(json.loads(result))

    total_end_time = time.time()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Search completed. Total time taken: {total_end_time - total_start_time:.2f} seconds.")

    # Save results to a JSON file
    with open('search_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Results saved to 'search_results.json'.")
