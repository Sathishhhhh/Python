import pandas as pd

data = pd.read_csv("movies.csv")

#print(data.head(5))

#print(data.tail(5))

#print(data.mean)

#print(data.median)

#print(data.mode)

#print(data.multiply)

#print(data.shape)

#print(data[3:5])

#print(data["imdb_rating"])

#print(data["imdb_rating"].max(), data["imdb_rating"].min())


#print(data[data.industry == "Bollywood"])
#print(data[data.industry == "Hollywood"])

#print(data.columns)

#print(data.industry.unique)

#print(data.language.value_counts())

#print(data[["title", "imdb_rating"]]) 

print(data[(data.release_year >= 2000 & (data.release_year <= 2010))])
print(data.to_csv("home.csv"))