from flask import *
from sqlite3 import *
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from string import punctuation
import warnings


app = Flask(__name__)
app.secret_key = "developed_by_vedant_kadam"

warnings.filterwarnings('ignore')

data = pd.read_csv("Books.csv")

newdata = data.dropna(how="any")
print(newdata.isnull().sum())

print(data.duplicated().sum())

def clean_title(txt):
        txt = txt.lower()
        txt = re.sub("[^A-z]", "", txt)     # except A-z removes everything other
        return txt.strip()                     # remove blank space after last word

def clean_author(txt):
        txt = txt.lower()
        return txt.strip()

def clean_publisher(txt):
        txt = txt.lower()
        txt = re.sub("[^A-z]", "", txt)     # except A-z removes everything other
        return txt.strip()

newdata["clean_title"] = newdata["Book-Title"].apply(clean_title)
newdata["clean_author"] = newdata["Book-Author"].apply(clean_author)
newdata["clean_publisher"] = newdata["Publisher"].apply(clean_publisher)
# print(newdata)

cv = CountVectorizer()
vector_author = cv.fit_transform(newdata["clean_author"])


@app.route("/", methods=["GET", "POST"])
def home():    
    return render_template("home.html")


@app.route("/books")
def books():
    top_50_books = data.head(50).to_dict('records')
    return render_template("books.html", top_books=top_50_books)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/recommend", methods=["GET", "POST"])
def recommend():
    if request.method == "POST":
        if 'na' in request.form:
            na = request.form["na"]
            na = clean_title(na)
            res = newdata[newdata.clean_title.str.contains(na)]

            if res.shape[0] > 0:
                res_author = res.head(1).clean_author
                sres_author = " ".join(res_author)
                vector_sres_author = cv.transform([sres_author])
                cs = cosine_similarity(vector_sres_author, vector_author)
                fcs = cs.flatten()
                indices = fcs.argsort()[-11:-1][::-1]  # Getting top 5 indices
                results = data.iloc[indices]
                res = results.to_dict('records')
                return render_template("recommend.html", msg=res)
            else:
                return render_template("recommend.html", msg="Book Not Found")
      
    return render_template("recommend.html")


    
@app.errorhandler(404)
def page_not_found(e):
        return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)