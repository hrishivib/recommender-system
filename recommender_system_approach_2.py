import pandas as pd
import scipy.sparse as sp
import numpy as np
import itertools as it
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from statistics import mean
from time import time, ctime
from math import ceil
import warnings
warnings.filterwarnings("ignore")
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error


# create CSR matrix

def createCSR(df): #this method is just for creating sparse matrix of the train.dat

    # rating
    data = []
    # user
    row = []
    # movie
    column = []

    # iterate and tokenize each line
    for ele in df:
        row.append(int(ele[0]))
        column.append(int(ele[1]))
        data.append(float(ele[2]))

    return sp.coo_matrix((data, (row,column)), shape = [df.size, 100000]).tocsr()



def createCSR2(df, dict1=None, dict2=None, flag=False):

    # rating
    data = []
    # row, user
    row = []
    # column, movie
    column = []

    # iterate and tokenize each line
    for ele in df:
        #if dictionary sent, use it to translate values
        #otherwise use the value raw
        if dict1==None:
            row.append(ele[0])
        else:
            row.append(dict1[ele[0]])
        if dict2==None:
            column.append(ele[1])
        else:
            column.append(dict2[ele[1]])
        #if flag falso, assume third column exists with data values
        #otherwise just fill with ones
        if flag==False:
            data.append(ele[2])
        else:
            data.append(1)
    return sp.coo_matrix((data, (row,column)), shape = [np.amax(row)+1, np.amax(column)+1]).tocsr()
    

def knn(csr_data, test_array,k):
    test_y = [] 
    i=0
    for index, line in enumerate(test_array):
        i+=1
        print(i)
        cos_similarity = cosine_similarity(csr_data[line[0],:], csr_data).flatten()
        # get the indices of nearest neighbors based on k parameter        
        indices = cos_similarity.argsort()[:-(k+1):-1].tolist()
        indices = indices[1:]
        ratings = []
        for userID in indices:
            ratings.append(csr_data[userID, line[1]])
   
        ratings = np.asarray(ratings)
        ratings[ratings == 0] = np.nan
        if all(np.isnan(ratings)):
            # if no neighbors have a rating, take the average of all people
            #ratings = train_file[:,line[1]].data.tolist()
   
            # average rating for user and impute
            ratings = csr_data[line[0],:].data.tolist()
            # when there are no ratings for a given movie in the entire matrix,
            # assign a rating of 3.0
            try:            
                rating = mean(ratings)
            except:
                try:
                    # get average rating for movie
                    ratings = csr_data[:,line[1]].data.tolist()
                    rating = mean(ratings)
                except:
                    # assign rating of 3 if all else fails
                    rating = 3.0
        else:
            rating = np.nanmean(ratings)
        test_y.append(rating)
        
        if index == 71298:
            print(index)

        
    return pd.DataFrame(test_y)        
        

def knn2(csr_data, test_array, csr_tags, users, movies, k=20):
    
    test_y = []# output

    user_cos_similarity = cosine_similarity(csr_data, csr_data, dense_output=False)   
    movie_cos_similarity = cosine_similarity(csr_data.transpose(), csr_data.transpose(), dense_output=False)
    tag_cos_similarity = cosine_similarity(csr_tags, csr_tags, dense_output=False)

    # iterate through all lines in the test reviews and classify them
    for index, line in enumerate(test_array):
        # cosine similarity

        user = users[line[0]]
        movie = movies[line[1]]
        
        # k nearest neighbors for users       
        user_ni = user_cos_similarity[user].todense()
        user_ni = np.array(user_ni).flatten()
        user_ni = user_ni.argpartition(-k-1)[-k-1:]
        user_ni = user_ni[np.where(user_ni != user)]

        user_ratings = []
        for userID in user_ni:
            user_ratings.append(csr_data[userID, movie])
        
        # user rating
        user_ratings = np.array(user_ratings)
        user_ratings = user_ratings[np.nonzero(user_ratings)]

        if user_ratings.size == 0:
            # if no ratings, average
            #ratings = train_file[:,line[1]].data.tolist()
            user_ratings = csr_data.tocsc()[user,:].data
            # if no rating for given movie, give 0
            try:            
                user_rating = user_ratings.mean()
            except:
                # 0 if all else fails
                user_rating = 0
        else:
            user_rating = user_ratings.mean()
            
        # k nearest neighbors for movies
        movie_ni = movie_cos_similarity[movie].todense()
        movie_ni = np.array(movie_ni).flatten()
        movie_ni = movie_ni.argpartition(-k-1)[-k-1:]
        movie_ni = movie_ni[np.where(movie_ni != movie)]
      
        movie_ratings = []
        for movieID in movie_ni:
            movie_ratings.append(csr_data[user, movieID])
        
        # calculate rating for user
        movie_ratings = np.array(movie_ratings)
        movie_ratings = movie_ratings[np.nonzero(movie_ratings)]

        if movie_ratings.size == 0:
           # if no ratings, average
            #ratings = train_file[:,line[1]].data.tolist()
        
            movie_ratings = csr_data[:, movie].data #gets nonzero data
            movie_rating = movie_ratings.mean()
            
            if (np.isnan(movie_rating)): 
                #if nan, assign 0
                movie_rating = 0
        else:
            movie_rating = movie_ratings.mean()
                 
        # k nearest neighbors for movie tags 
        tag_ni = tag_cos_similarity[movie].todense()
        tag_ni = np.array(tag_ni).flatten()
        tag_ni = tag_ni.argpartition(-k-1)[-k-1:]
        tag_ni = tag_ni[np.where(tag_ni != movie)]
      
        new_ratings = csr_data[:, tag_ni]
        tag_ratings = np.true_divide(new_ratings.sum(0),(new_ratings!=0).sum(0))
            
        tag_rating = np.nanmean(tag_ratings)

        ratings = np.array([user_rating, movie_rating, tag_rating])
        
        ratings = ratings[np.nonzero(ratings)]
 
       
        if ratings.size == 0:
            new_rating = 3.5
        else:
            new_rating = np.mean(ratings)
                    
        test_y.append(new_rating)

        #test code to see if it returns nan
        if np.isnan(new_rating):
            print(index)
            print(ratings)
        
        #verbose code so that the user knows progress is being made
        if(index % 1000 == 0):
            print(ratings)
            print(index)
            print(ctime())
        
    return np.array(test_y)
        

def optK(train):
    
    rating = train['rating']
    X_train, X_test, y_train, y_test = train_test_split(train, rating, test_size=0.20, random_state=1)
    X_test = X_test.drop(columns=['rating'])
    
    csr_data = createCSR(X_train.to_numpy())

    mse=[]
    index = []
    start = time.time()
    for i in range(0,21):
        k = 100 + 10*i
        index.append(k)
        y_pred = knn(csr_data,X_test.to_numpy(), k)
        mse.append(mean_squared_error(y_test, y_pred))
    print("execution time:", time.time()-start)
    print("optimum k = ",index[np.argmin(mse)])
    print("min MSE = ", np.min(mse))
    plt.figure(figsize=(16,8))
    plt.plot(index, mse, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Mean Squared Error')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()


def runner2():
    
    # reading data to numpy arrrays
    test_array = pd.read_table("test.dat", skip_blank_lines=False, delim_whitespace=True).to_numpy()
    train_array = pd.read_table("train.dat", skip_blank_lines=False, delim_whitespace=True).to_numpy()
    genre_array = pd.read_table("movie_genres.dat", skip_blank_lines=False, delim_whitespace=True).to_numpy()
    movie_tag_array = pd.read_table("movie_tags.dat", skip_blank_lines=False).to_numpy()
    actor_array = pd.read_table("movie_actors.dat", skip_blank_lines=False).to_numpy()
    actor_array = np.delete(actor_array, 2, 1)
    director_array = pd.read_table("movie_directors.dat", skip_blank_lines=False).to_numpy()[:,0:2]



# all the unique IDs in dictionary
    users = np.unique(np.concatenate((train_array[:,0], test_array[:,0])))
    users = dict(zip(users, np.arange(users.size))) 

    movies = np.unique(np.concatenate((train_array[:,1], test_array[:,1], np.asarray(genre_array[:,0], dtype=int), movie_tag_array[:,1], np.asarray(director_array[:,0], dtype=int), np.asarray(actor_array[:,0], dtype=int),)))

    movies = dict(zip(movies, np.arange(movies.size)))

    genres = np.unique(genre_array[:,1])
    genres = dict(zip(genres, np.arange(genres.size)))

    actors = np.unique(actor_array[:,1])
    actors = dict(zip(actors, np.arange(actors.size)))
    directors = np.unique(director_array[:,1])
    directors = dict(zip(directors, np.arange(directors.size)))

    # conversion to sparse matrices
    train_array = createCSR2(train_array, users, movies)
    genre_array = createCSR2(genre_array, movies, genres, flag=True)
    movie_tag_array = createCSR2(movie_tag_array, movies, None)
    actor_array = createCSR2(actor_array, movies, actors)
    director_array = createCSR2(director_array, movies, directors, flag=True)

    tfidf = TfidfTransformer()
    genre_array = tfidf.fit_transform(genre_array)
    movie_tag_array = tfidf.fit_transform(movie_tag_array)
    actor_array = tfidf.fit_transform(actor_array)
    director_array = tfidf.fit_transform(director_array)
    submission = knn2(train_array, test_array, movie_tag_array, users, movies, k=150)
    np.savetxt("submission_test_miner" + "_" + str(ceil(time())) + ".txt", submission, fmt='%s')



def runner():
#     input the data and format it to usable format

    test = pd.read_table("test.dat", header=None, skip_blank_lines=False, delim_whitespace=True)
    header = test.iloc[0]
    print(header)
    test = test.iloc[1:]
    test.columns = header
    test['userID']= pd.to_numeric(test['userID'], errors='coerce')
    test['movieID']= pd.to_numeric(test['movieID'], errors='coerce')
    test_array = test.to_numpy()

#     converting just the train data to a sparse csr matrix

    train_df = pd.read_table("train.dat", header = None, delim_whitespace=True)
    header = train_df.iloc[0]
    train_df = train_df.iloc[1:]
    train_df.columns = header
    print("train:\n\n\n", train_df)

    train_df['userID']= pd.to_numeric(train_df['userID'], errors='coerce')
    train_df['movieID']= pd.to_numeric(train_df['movieID'], errors='coerce')
    train_df['rating']= pd.to_numeric(train_df['rating'], errors='coerce')
    csr_data = createCSR(train_df.to_numpy())
    print(csr_data)
    submission = knn(csr_data, test_array, 150)
    np.savetxt(r'submission_150_', submission.values, fmt='%s')
      
# Main execution     
runner2()