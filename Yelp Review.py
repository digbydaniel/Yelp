import pandas as pd
business = pd.read_json('yelp_business.json', lines=True)
review = pd.read_json('yelp_review.json', lines=True)
user = pd.read_json('yelp_user.json', lines=True)
checkin = pd.read_json('yelp_checkin.json', lines=True)
tip = pd.read_json('yelp_tip.json', lines=True)
photo = pd.read_json('yelp_photo.json', lines=True)

# pd.options.display.max_columns = 60
pd.options.display.max_colwidth = 500
business.head()
review.head()
user.head()
checkin.head()
tip.head()
photo.head()

len(business)
review.columns

user.describe()

business[business['business_id']  == '5EvUIR4IzCWUOm0PsUZXjA']['stars']
df = pd.merge(business, review, how='left', on='business_id')
df = pd.merge(df, user, how = 'left', on = 'business_id')
df = pd.merge(df, checkin, how = 'left', on = 'business_id')
df = pd.merge(df, tip, how = 'left', on = 'business_id')
df = pd.merge(df, photo, how = 'left', on = 'business_id')
print(df.columns)

features_to_remove = ['address','attributes','business_id','categories','city','hours','is_open','latitude','longitude','name','neighborhood','postal_code','state','time']
df.drop(features_to_remove, axis=1, inplace=True)
#determine if there is any NaN in our data set
df.isna().any()
#fill in any NaN
df.fillna({'weekday_checkins':0, 'weekend_checkins':0, 'average_tip_length':0, 'number_tips':0, 'average_caption_length':0, 'number_tips':0, 'number_pics':0}, inplace=True)
df.isna().any()

df.corr()

from matplotlib import pyplot as plt

# plot average_review_sentiment against stars here
plt.scatter(df['average_review_sentiment'], df['stars'], alpha=0.025)
plt.show

# plot average_review_length against stars here
plt.scatter(df['average_review_length'], df['stars'], alpha=0.025)
plt.show

# plot average_review_age against stars here
plt.scatter(df['average_review_age'], df['stars'], alpha=0.025)
plt.show

# plot number_funny_votes against stars here
plt.scatter(df['number_funny_votes'], df['stars'], alpha=0.025)
plt.show

features = df[['average_review_length', 'average_review_age']]
ratings = df['stars']

#split data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, ratings, test_size = 0.2, random_state = 1)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

model.score(X_train, y_train)
model.score(X_test, y_test)