import numpy as np
import scipy
import pandas as pd
from tqdm import tqdm
import random
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
import math
import os
import csv

from procedure import p1_load_data

# Process 1
donations_df, projects, donations, donors = p1_load_data()
# print(donations_df.loc[0])

# Process 2 projects||donations_full_df
# Deal with missing values
donations_df["Donation Amount"] = donations_df["Donation Amount"].fillna(0)

# Define event strength as the donated amount to a certain project
donations_df['eventStrength'] = donations_df['Donation Amount']


def smooth_donor_preference(x):
    return math.log(1 + x, 2)


donations_full_df = donations_df.groupby(['Donor ID', 'Project ID'])['eventStrength'].sum().apply(
    smooth_donor_preference).reset_index()

# Update projects dataset
project_cols = projects.columns
projects = donations_df[project_cols].drop_duplicates()

print('# of projects: %d' % len(projects))
print('# of unique user/project donations: %d' % len(donations_full_df))

# Process 3 donations_full_indexed_df||donations_train_indexed_df||donations_test_indexed_df
donations_train_df, donations_test_df = train_test_split(donations_full_df, test_size=0.20, random_state=42)

print('# donations on Train set: %d' % len(donations_train_df))
print('# donations on Test set: %d' % len(donations_test_df))

# Indexing by Donor Id to speed up the searches during evaluation
donations_full_indexed_df = donations_full_df.set_index('Donor ID')
donations_train_indexed_df = donations_train_df.set_index('Donor ID')
donations_test_indexed_df = donations_test_df.set_index('Donor ID')

# Process 4 deal with text
'''Content-Based Filtering model'''
# Preprocessing of text data
textfeats = ["Project Title", "Project Essay"]
for cols in textfeats:
    projects[cols] = projects[cols].astype(str)
    projects[cols] = projects[cols].astype(str).fillna('')  # FILL NA
    projects[cols] = projects[
        cols].str.lower()  # Lowercase all text, so that capitalized words dont get treated differently

text = projects["Project Title"] + ' ' + projects["Project Essay"]
vectorizer = TfidfVectorizer(strip_accents='unicode',
                             analyzer='word',
                             lowercase=True,  # Convert all uppercase to lowercase
                             stop_words='english',
                             # Remove commonly found english words ('it', 'a', 'the')
                             # which do not typically contain much signal
                             max_df=0.9,
                             # Only consider words that appear in fewer than max_df percent of all documents
                             # max_features=5000 # Maximum features to be extracted
                             )
project_ids = projects['Project ID'].tolist()
tfidf_matrix = vectorizer.fit_transform(text)
tfidf_feature_names = vectorizer.get_feature_names()


# Process 5
def get_project_profile(project_id):
    idx = project_ids.index(project_id)
    project_profile = tfidf_matrix[idx:idx + 1]
    return project_profile


def get_project_profiles(ids):
    project_profiles_list = [get_project_profile(x) for x in np.ravel([ids])]
    project_profiles = scipy.sparse.vstack(project_profiles_list)
    return project_profiles


def build_donors_profile(donor_id, donations_indexed_df):
    donations_donor_df = donations_indexed_df.loc[donor_id]
    donor_project_profiles = get_project_profiles(donations_donor_df['Project ID'])
    donor_project_strengths = np.array(donations_donor_df['eventStrength']).reshape(-1, 1)
    # Weighted average of project profiles by the donations strength
    weighted_donor_project_profile = donor_project_profiles.multiply(donor_project_strengths)
    donor_project_profiles_sum = weighted_donor_project_profile.sum(axis=0)
    donor_project_strengths_weighted_avg = donor_project_profiles_sum / (np.sum(donor_project_strengths) + 1)
    # if not np.all(donor_project_strengths_weighted_avg == 0):
    donor_profile_norm = sklearn.preprocessing.normalize(donor_project_strengths_weighted_avg)
    return donor_profile_norm


def build_donors_profiles():
    donations_indexed_df = donations_full_df[donations_full_df['Project ID'].isin(projects['Project ID'])].set_index(
        'Donor ID')
    donor_profiles = {}
    for donor_id in tqdm(donations_indexed_df.index.unique()):
        donor_profiles[donor_id] = build_donors_profile(donor_id, donations_indexed_df)
    return donor_profiles


# Process build donor profiles
donor_profiles = build_donors_profiles()
cache_donor_profile_path = os.getcwd() + '/../io/cached_donor_profile_matrix.csv'
print("# of donors with profiles: %d" % len(donor_profiles))

mydonor1 = "6d5b22d39e68c656071a842732c63a0c"
mydonor1_profile = pd.DataFrame(sorted(zip(tfidf_feature_names,
                                           donor_profiles[mydonor1].flatten().tolist()),
                                       key=lambda x: -x[1])[:10],
                                columns=['token', 'relevance'])
print(mydonor1_profile)

if os.path.exists(cache_donor_profile_path) is False or os.path.getsize(cache_donor_profile_path) == 0:
    with open(cache_donor_profile_path, 'wb') as f:
        w = csv.DictWriter(f, donor_profiles.keys())
        w.writeheader()
        w.writerow(donor_profiles)
