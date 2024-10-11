import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# def process_csv_file(file_path, output_dir):
#     df = pd.read_csv(file_path)

#     vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

#     # Ingredients tagging
#     ingredient_tfidf = vectorizer.fit_transform(df['Ingredients'].fillna(''))
#     kmeans_ingredients = KMeans(n_clusters=5, random_state=42)
#     df['Ingredient_Tags'] = kmeans_ingredients.fit_predict(ingredient_tfidf)

#      # Handle Indications_and_Usage tagging
#     indications_tfidf = vectorizer.fit_transform(df['Indications_and_Usage'].fillna(''))
#     kmeans_indications = KMeans(n_clusters=5, random_state=42)
#     df['Indications_Tags'] = kmeans_indications.fit_predict(indications_tfidf)
    
#     # Product_Name tagging
#     product_name_tfidf = vectorizer.fit_transform(df['Product_Name'].fillna(''))
#     kmeans_product_name = KMeans(n_clusters=5, random_state=42)
#     df['Product_Name_Tags'] = kmeans_product_name.fit_predict(product_name_tfidf)

#     output_file_path = os.path.join(output_dir, os.path.basename(file_path).replace('.csv', '_tagged.csv'))
#     df.to_csv(output_file_path, index=False)
#     print(f"Updated CSV file saved at: {output_file_path}")

# # CSV files to process
# csv_files = [
#     r'C:\Users\nirmiti.deshmukh\ML_Tagging\Output_Files\dm_spl_release_human_rx_part2.csv',
#     r'C:\Users\nirmiti.deshmukh\ML_Tagging\Output_Files\dm_spl_release_human_rx_part3.csv',
#     r'C:\Users\nirmiti.deshmukh\ML_Tagging\Output_Files\dm_spl_release_human_rx_part4.csv',
#     r'C:\Users\nirmiti.deshmukh\ML_Tagging\Output_Files\dm_spl_release_human_rx_part5.csv'
# ]

# output_dir = r'C:\Users\nirmiti.deshmukh\ML_Tagging\Processed_Files'

# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# for file in csv_files:
#     process_csv_file(file, output_dir)


# #{publication data classify+ tag} in a way ki drug, disease, sara publication type, ingrdients, product name, teritory, company values
# #{clinical trial data ko tag} krna on basis  phases meinn tag karna, konse drug ke liye uske liye tag krna + what disease ke liye bhi, sponsors, company value, 

#CLINICAL TAGGING

def process_clinical_trial_data(file_path, output_dir):
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Initialize the vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    
    # Tagging for phases
    phases_tfidf = vectorizer.fit_transform(
    df[['Trial characteristics_1', 'Trial characteristics_2', 'Developmental phase']].fillna('').agg(' '.join, axis=1))    
    kmeans_phases = KMeans(n_clusters=5, random_state=42)
    df['Phase_Tags'] = kmeans_phases.fit_predict(phases_tfidf)
    
    # Tagging for drugs used (from Interventions/Control columns)
    interventions = df[['Interventions/Control_1', 'Interventions/Control_2', 'Interventions/Control_3',
                        'Interventions/Control_4', 'Interventions/Control_5', 'Interventions/Control_6',
                        'Interventions/Control_7', 'Interventions/Control_8', 'Interventions/Control_9',
                        'Interventions/Control_10']].fillna('').agg(' '.join, axis=1)
    drugs_tfidf = vectorizer.fit_transform(interventions)
    kmeans_drugs = KMeans(n_clusters=5, random_state=42)
    df['Drug_Tags'] = kmeans_drugs.fit_predict(drugs_tfidf)
    
    # Tagging for diseases relevancy from Condition column
    diseases_tfidf = vectorizer.fit_transform(
        df[['Condition','Classification by specialty']].fillna('').agg(' '.join, axis=1))
    kmeans_diseases = KMeans(n_clusters=5, random_state=42)
    df['Disease_Tags'] = kmeans_diseases.fit_predict(diseases_tfidf)
    
    # Tagging for sponsors relevancy from Organization and Category of Funding Organization
    sponsors_tfidf = vectorizer.fit_transform(df[['Organization', 'Category of Funding Organization']].fillna('').agg(' '.join, axis=1))
    kmeans_sponsors = KMeans(n_clusters=5, random_state=42)
    df['Sponsor_Tags'] = kmeans_sponsors.fit_predict(sponsors_tfidf)
    
    output_csv_file = os.path.join(output_dir, os.path.basename(file_path).replace('.csv', '_tagged.csv'))
    df.to_csv(output_csv_file, index=False)
    
    print(f"Tagged data saved as CSV at: {output_csv_file}")

input_file_path = r'japan_data\Japan_Clinical_Trial_Dataset 1.csv'
output_dir = r'japan_data\japan_tagged'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

process_clinical_trial_data(input_file_path, output_dir)
