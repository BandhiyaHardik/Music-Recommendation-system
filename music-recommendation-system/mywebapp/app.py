from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Read the data
df = pd.read_csv("data/song-dataset.csv", low_memory=False)[:1000]

# Remove duplicates
df = df.drop_duplicates(subset="Song Name")

# Drop Null values
df = df.dropna(axis=0)

# Drop the non-required columns
df = df.drop(df.columns[3:], axis=1)

# Removing spaces from "Artist Name" column
df["Artist Name"] = df["Artist Name"].str.replace(" ", "")

# Combine all columns and assign as a new column
df["data"] = df.apply(lambda value: " ".join(value.astype("str")), axis=1)

# Models
vectorizer = CountVectorizer()
vectorized = vectorizer.fit_transform(df["data"])
similarities = cosine_similarity(vectorized)

# Assign the new DataFrame with `similarities` values
df_tmp = pd.DataFrame(similarities, columns=df["Song Name"], index=df["Song Name"]).reset_index()

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    error_message = ""

    if request.method == 'POST':
        input_song = request.form.get('song')
        if input_song in df_tmp.columns:
            recommendation = df_tmp.nlargest(11, input_song)["Song Name"]
            recommendations = recommendation.values[1:]
        else:
            error_message = "Sorry, there is no song name in our database. Please try another one."

    count = len(recommendations)

    return render_template('index.html', count=count, recommendations=recommendations, error=error_message)

  
if __name__ == '__main__':
    app.run(debug=True)
