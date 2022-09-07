import time
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor



def read_data():
    train_df = pd.read_csv("data/additional_data/train.dat", sep= ' ',encoding='latin-1')
    test_df = pd.read_csv("data/additional_data/test.dat", sep= ' ')
    movieActors_df = pd.read_csv("data/additional_data/movie_actors.dat", sep= '\t',encoding='latin-1')
    movieDirectors_df = pd.read_csv("data/additional_data/movie_directors.dat", sep= '\t',encoding='latin-1')
    movieGenres_df = pd.read_csv("data/additional_data/movie_genres.dat", sep= '\t',encoding='latin-1')
    movieTags_df = pd.read_csv("data/additional_data/movie_tags.dat", sep= '\t',encoding='latin-1')
    tags_df = pd.read_csv("data/additional_data/tags.dat", sep= '\t',encoding='latin-1')
    userTaggedMovies_df = pd.read_csv("data/additional_data/user_taggedmovies.dat", sep= ' ',encoding='latin-1')
    
    return train_df,test_df,movieActors_df,movieDirectors_df,movieGenres_df,movieTags_df,tags_df,userTaggedMovies_df



def pre_processing_data(movieDirectors_df, movieActors_df, train_df, test_df):
    #removing director name
    movieDirectors_df = movieDirectors_df.drop(columns=['directorName'])
    
    #remove actor name and ranking
    movieActors_df = movieActors_df.drop(columns=['actorName','ranking'])
    
    # merge usertag and main train data (userid, movieid)
    final_train_df = train_df.merge(userTaggedMovies_df, on=['userID','movieID'], how ='left')
    
    # merge new train and movietags 
    final_train_df = final_train_df.merge(movieTags_df, on=['movieID','tagID'], how = 'left')
    
    # add genre to new train obtained above
    final_train_df = final_train_df.merge(movieGenres_df, on='movieID', how = 'left')
    
    #add the actor to new trained obtained above
    final_train_df = final_train_df.merge(movieActors_df, on='movieID', how = 'left')
    
    # add director to new train obtained above
    #final_train_df = final_train_df.merge(movieDirectors_df, on='movieID', how = 'left')
    
    #Do the merging for the test data as we earlier did for train data
    final_test_df = test_df.merge(userTaggedMovies_df, on=['userID','movieID'], how = 'left')
    
    # merge new test and movietags 
    final_test_df = final_test_df.merge(movieTags_df, on= ['movieID','tagID'], how = 'left')
    
    # add genre to new test obtained above
    final_test_df = final_test_df.merge(movieGenres_df, on= ['movieID'], how = 'left')
    
    # add actor to new obtained test
    final_test_df = final_test_df.merge(movieActors_df, on =['movieID'], how = 'left')
    
    # add moviedirector to new test obtained
    #final_test_df = final_test_df.merge(movieDirectors_df,on=['movieID'], how = 'left')
    
    return movieDirectors_df, movieActors_df,final_train_df, final_test_df


def normalizing_data(final_train_df, final_test_df):
    #using minmaxscalar for normalization
    min_max_scaler = preprocessing.MinMaxScaler()
    
    temp_df = final_train_df[['userID','movieID']]
    temp_var = temp_df.values
    temp_scaled_var = min_max_scaler.fit_transform(temp_var)
    temp_df = pd.DataFrame(temp_scaled_var)
    temp_df.columns = ['userID','movieID']
    final_train_df[['userID','movieID']] = temp_df[['userID','movieID']]
    
    temp_df = final_test_df[['userID','movieID']]
    temp_var = temp_df.values
    temp_scaled_var = min_max_scaler.fit_transform(temp_var)
    temp_df = pd.DataFrame(temp_scaled_var)
    temp_df.columns = ['userID','movieID']
    final_test_df[['userID','movieID']] = temp_df[['userID','movieID']]    
    
    #removing tagID as it is not that relevant from both train and test
    final_train_df = final_train_df.drop(columns=['tagID'])
    final_test_df = final_test_df.drop(columns=['tagID'])
    
    return final_train_df, final_test_df


def encoding_data(final_train_df, final_test_df):
    #converting cateogircal data into numbers -> genre and actorID for TRAIN data
    final_train_df.genre = pd.Categorical(final_train_df.genre)
    final_train_df['genre'] = final_train_df.genre.cat.codes
    #final_train_df.directorID = pd.Categorical(final_train_df.directorID)
    #final_train_df['directorID'] = final_train_df.directorID.cat.codes
    final_train_df.actorID = pd.Categorical(final_train_df.actorID)
    final_train_df['actorID'] = final_train_df.actorID.cat.codes
    
    #converting cateogircal data into numbers -> genre and actorID for TEST data
    final_test_df.genre = pd.Categorical(final_test_df.genre)
    final_test_df['genre'] = final_test_df.genre.cat.codes
    # final_test_df.directorID = pd.Categorical(final_test_df.directorID)
    # final_test_df['directorID'] = final_test_df.directorID.cat.codes
    final_test_df.actorID = pd.Categorical(final_test_df.actorID)
    final_test_df['actorID'] = final_test_df.actorID.cat.codes
    
    return final_train_df, final_test_df




def normalize_actor_genre(final_train_df, final_test_df):
    # normalize directorID, genre because again huge numbers due to categorical conversion
    min_max_scaler = preprocessing.MinMaxScaler()    
    temp_df = final_train_df[['actorID','genre']]
    temp_var = temp_df.values #returns a numpy array    
    temp_var_scaled = min_max_scaler.fit_transform(temp_var)
    temp_df = pd.DataFrame(temp_var_scaled)    
    temp_df.columns=['actorID','genre']    
    final_train_df[['actorID','genre']] = temp_df[['actorID','genre']]
    
    temp_df = final_test_df[['actorID','genre']]
    temp_var = temp_df.values #returns a numpy array    
    temp_var_scaled = min_max_scaler.fit_transform(temp_var)
    temp_df = pd.DataFrame(temp_var_scaled)    
    temp_df.columns=['actorID','genre']    
    final_test_df[['actorID','genre']] = temp_df[['actorID','genre']]
    
    return final_train_df, final_test_df


def modelling(final_train_df,final_test_df):
    #choosing some random samples from TRAIN DATA because data is too big to handle so tried with multiple size of training data
    temp_df = final_train_df.sample(n = 100000)    
    
    #removing ratings
    X = temp_df.drop(columns=['rating'])

    # train test split
    X_train = X.values[:,:] #data from train file except "ratings"
    y_train = df.values[:,2] ##data from train file only "ratings"
    X_test =  final_test_df.values[:,:] #data from test file to predict the ratings
    
#    KNN regressor for regression to predict the ratings for test data
#     knn_regressor = KNeighborsRegressor(n_neighbors=140)
#     knn_regressor.fit(X_train, y_train)
#     y_pred = knn_regressor.predict(X_test)
#     final_test_df['rating'] = list(y_pred)
    
    #Random forest regressor to predict the ratings for test data
    rf_regressor = RandomForestRegressor()
    rf_regressor.fit(X_train, y_train)
    y_pred = rf_regressor.predict(X_test)
    
    #appending final prediction of ratings to the dataframe
    final_test_df['rating'] = list(y_pred)
    
    #Removing the columns from dataframe
    final_test_df = final_test_df.drop(columns=['tagWeight','genre','actorID'], axis = 1)
    
    #grouping the dataframe using userID and movieID
    final_test_df = final_test_df.groupby(['userID', 'movieID']).mean()
    
    pred_list = list(final_test_df['rating'])
    pred_formatted_list = [ round(elem,1) for elem in pred_list ]
    predicted_df = pd.DataFrame(pred_formatted_list)
    
    return predicted_df



# Main execution
start_time = time.time()


#Reading data from files
train_df,test_df,movieActors_df,movieDirectors_df,movieGenres_df,movieTags_df,tags_df,userTaggedMovies_df = read_data()


#Pre-processing on the data
movieDirectors_df, movieActors_df, train_df, test_df = pre_processing_data(movieDirectors_df, movieActors_df, train_df, test_df)

#Normalizing the data
final_train_df, final_test_df = normalizing_data(train_df, test_df)


#Encoding the categorical data
final_train_df, final_test_df = encoding_data(final_train_df, final_test_df)


#Fillna for tagWeight by 0 for TRAIN and TEST
final_train_df['tagWeight'] = final_train_df['tagWeight'].fillna(0)
final_test_df['tagWeight'] = final_test_df['tagWeight'].fillna(0)


#Normalizing the actorID and genre
final_train_df, final_test_df = normalize_actor_genre(final_train_df, final_test_df)


#Modelling and prediction
predicted_df = modelling(final_train_df,final_test_df)


#Savings the prediction in the csv file for miner submission
#predicted_df.to_csv("test_data_ratings.csv", index=None, header=False)


print("Execution time: ",time.time()-start_time,"secs.")