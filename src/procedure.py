import os
import numpy as np
import scipy
import pandas as pd

test_mode = True
chunk_mode = False
chunk_size = 10 * 6

donations_file_path = os.getcwd() + '/../io/Donations.csv'
cached_donations_file_path = os.getcwd() + '/../io/cached_donations.csv'
projects_file_path = os.getcwd() + '/../io/Projects.csv'
cached_projects_file_path = os.getcwd() + '/../io/cached_projects.csv'
donors_file_path = os.getcwd() + '/../io/Donors.csv'
cached_donors_file_path = os.getcwd() + '/../io/cached_donors.csv'

cache_file_path = os.getcwd() + '/../io/cached_donations_df.csv'


def p1_load_data():
    projects, donations, donors = None, None, None
    if os.path.exists(cache_file_path) is False or os.path.getsize(cache_file_path) == 0:
        # Read datasets
        if chunk_mode:
            projects = pd.read_csv(projects_file_path, sep=',', header=0, keep_default_na=True, chunksize=chunk_size,
                                   iterator=True).get_chunk(10 ** 5)
            donations = pd.read_csv(donations_file_path, sep=',', header=0, keep_default_na=True, chunksize=chunk_size,
                                    iterator=True).get_chunk(10 ** 5)
            donors = pd.read_csv(donors_file_path, sep=',', header=0, keep_default_na=True, chunksize=chunk_size,
                                 iterator=True).get_chunk(10 ** 5)
        else:
            projects = pd.read_csv(projects_file_path, sep=',', header=0, keep_default_na=True)
            donations = pd.read_csv(donations_file_path, sep=',', header=0, keep_default_na=True)
            donors = pd.read_csv(donors_file_path, sep=',', header=0, keep_default_na=True)

        print(projects.shape)
        print(donations.shape)
        print(donors.shape)

        # this piece of code converts Project_ID which is a 32-bit Hex int digits 10-1010
        # create column "project_id" with sequential integers
        f = len(projects)
        projects['project_id'] = np.nan
        g = list(range(10, f + 10))
        g = pd.Series(g)
        projects['project_id'] = g.values

        # Merge datasets
        donations = donations.merge(donors, on="Donor ID", how="inner")
        df = donations.merge(projects, on="Project ID", how="inner")
        if test_mode:
            df = df.head(10000)
        donations_df = df
        df.to_csv(cache_file_path)
        print('donation_df have writen to scv file:' + cache_file_path)
        projects = projects.loc[projects['Project ID'].isin(donations_df['Project ID'])]
        projects.to_csv(cached_projects_file_path)
        print('cached file1' + cached_projects_file_path)
        donations = donations.loc[donations['Donation ID'].isin(donations_df['Donation ID'])]
        donations.to_csv(cached_donations_file_path)
        print('cached file2' + cached_donations_file_path)
        donors = donors.loc[donors['Donor ID'].isin(donations_df['Donor ID'])]
        donors.to_csv(cached_donors_file_path)
        print('cached file3' + cached_donors_file_path)
    else:
        print('read from scv file:' + cache_file_path)
        donations_df = pd.read_csv(cache_file_path)
        print('read projects, donations, donors')
        if chunk_mode:
            projects = pd.read_csv(projects_file_path, sep=',', header=0, keep_default_na=True, chunksize=chunk_size,
                                   iterator=True).get_chunk(10 ** 5)
            donations = pd.read_csv(donations_file_path, sep=',', header=0, keep_default_na=True, chunksize=chunk_size,
                                    iterator=True).get_chunk(10 ** 5)
            donors = pd.read_csv(donors_file_path, sep=',', header=0, keep_default_na=True, chunksize=chunk_size,
                                 iterator=True).get_chunk(10 ** 5)
        else:
            projects = pd.read_csv(cached_projects_file_path, sep=',', header=0, keep_default_na=True)
            donations = pd.read_csv(cached_donations_file_path, sep=',', header=0, keep_default_na=True)
            donors = pd.read_csv(cached_donors_file_path, sep=',', header=0, keep_default_na=True)

    print('donation_df\'s shape:' + str(donations_df.shape))
    return donations_df, projects, donations, donors
