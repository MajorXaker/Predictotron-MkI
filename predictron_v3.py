import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


music_data = pd.read_csv("music.csv")

X = music_data.drop(columns=["genre"])
# X - is for input data (convention)
y = music_data["genre"]
# y - is for result data (convention)

model = DecisionTreeClassifier()
model.fit(X, y)

tree.export_graphviz(
    model,
    out_file="music-recommender.dot",
    feature_names=["age", "gender"], # to show the rules of the node
    class_names=sorted(y.unique()), # to show the class for each node
    label="all", # every note has labels
    rounded=True, # each box is rounded
    filled=True, # each box is filled with color
)

aaa = 5
