from flask import Flask,render_template,request
import pickle
app = Flask(__name__)

with open('similarity.pkl','rb') as file:
    movies = pickle.load(file)

titles = movies['title']

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances=similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True ,key=lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(movies.iloc[i[0]].title)
        print(i[0])

@app.route('/',methods=['POST','GET'])
def hello_world():
    user_input = request('user-input')
    return render_template('movie.html',titles=titles)


app.run(debug=True)