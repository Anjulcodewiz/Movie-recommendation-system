import pandas as pd
import pickle
import numpy as np
from flask import  Flask, render_template,request, url_for, jsonify, make_response
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel
import requests
import json

# building the recommender engine
with open('movie_dict5.pkl','rb') as f:
    movie_dict = pickle.load(f)
movies = pd.DataFrame(movie_dict)

# text processing
tfv = TfidfVectorizer(min_df=3,max_features = None,strip_accents='unicode',analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1,3),stop_words='english')
matrix = tfv.fit_transform(movies['combined'])

# model building
sig = sigmoid_kernel(matrix,matrix)

# function for getting recommended movies
def recommender(movie):
  if movie not in movies['title'].unique():
    no_movie = 'Sorry!The movie you requested is not in our database. Please check the spelling or try with some other movies'
    return no_movie
  else:
    movie_index = movies[movies['title'] == movie].index[0]
    distance =sig[movie_index]
    movie_list = sorted(list(enumerate(distance)),reverse = True,key = lambda x:x[1])[0:9]
    L = []
    id_num = []
    for i in movie_list:
        L.append((movies.iloc[i[0]].title))
        id_num.append((movies.iloc[i[0]].movie_id))
    return L,id_num

app = Flask(__name__)
@app.route("/")
def main():
  return render_template('index.html')

#movie suggestion engine
@app.route("/movies",methods = ['POST'])
def movie_info():
  if request.method == 'POST':
    req = request.get_json()
    input = req['letter']
    movie_suggestion = []
    counter = 0
    for i in movies['title']:
        if input.lower() in i.lower():
            movie_suggestion.append(i)
            counter = counter + 1
            if counter == 6:
                break
    res = make_response(jsonify({'movie_list':movie_suggestion}))
    return res
#getting searched movie
@app.route("/recommend1",methods = ['POST'])
def recommend1():
  if request.method == 'POST':
    req = request.get_json()
    searched_movie = req['movie_entered']
    
    #recommending
    r = recommender(searched_movie)
    movie_name = r[0]
    id = r[1]
    id_to_int = []
    for i in id:
      id_to_int.append(int(i))
    
    res = make_response(jsonify({'movie_names':movie_name,'movie_id':id_to_int}))
    return res

#getting clicked movie
@app.route("/recommend2",methods = ['POST'])
def recommend2():
  if request.method == 'POST':
    req2 = request.get_json()
    clicked_movie = req2['movie_clicked']
  
    #recommending
    name = movies[movies['movie_id'] == int(clicked_movie)]['title']
    
    r = recommender(name.values[0])
    movie_name = r[0]
    id = r[1]
    print(id)
    id_to_int = []
    for i in id:
      id_to_int.append(int(i))
    
    
    res = make_response(jsonify({'movie_names':movie_name,'movie_id':id_to_int}))
    return res



if __name__== "__main__":
  app.run(debug = True)