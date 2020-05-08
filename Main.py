import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn import tree
import time

########################################################
# car_df is the dataframe load from veh_tx
# df is the car_df dataframe that already group by, calculate the parking duration and assign shopping_point
########################################################

#################################################### STEP 1: Load Dataset #####################################################################

car_df1 = pd.read_csv(
    'G:\\My Drive\\ITS_DA\\veh_tx_data\\2019-11-10_15.zip', compression='zip')
car_df2 = pd.read_csv(
    'G:\\My Drive\\ITS_DA\\veh_tx_data\\2019-11-10_18.zip', compression='zip')

#############################################################################################################################################

#################################################### STEP 1.5: Data Processing ##############################################################
# car_df1.isna().sum()
# car_df2.isna().sum()
car_df1.dropna(inplace=True)
car_df2.dropna(inplace=True)

car_df = pd.concat([car_df1, car_df2])
# print(car_df.isna().sum())

# print(car_df.info())
# print(car_df.head(50))
# print(shopping_df.head())


# STEP 2: Data Exploration

# Calculate Time Elpased to detemine that which point is a shopping point
car_df["time_stamp"] = pd.to_datetime(car_df["time_stamp"])


##########################################################################################################

# Round Latitude Longitude to 1 digit float
car_df["latx"] = round(car_df["lat"], 1)
car_df["lonx"] = round(car_df["lon"], 1)

# Round Latitude Longitude to 2 digit float
car_df["latxx"] = round(car_df["lat"], 2)
car_df["lonxx"] = round(car_df["lon"], 2)

# Group by the value to calculate the duration of parking
test = car_df[car_df['speed'] == 0].groupby(['vid', 'latxx', 'lonxx', 'unit_type'])[
    'time_stamp'].min()  # first time_stamp
test2 = car_df[car_df['speed'] == 0].groupby(['vid', 'latxx', 'lonxx', 'unit_type'])[
    'time_stamp'].max()  # last time_stamp
ans = test2-test

df1 = pd.DataFrame(data=ans.index, columns=['vid'])
df2 = pd.DataFrame(data=ans.values, columns=['duration'])
df = pd.merge(df1, df2, left_index=True, right_index=True)
# print(df.head(30))

car_df["parking"] = ""
df["parking"] = ""
df["latxx"] = ""
df["lonxx"] = ""
df["unit_type"] = ""
df[['vid', 'latxx', 'lonxx', 'unit_type']] = pd.DataFrame(
    df['vid'].tolist(), index=df.index)
# print(df.head(20))

############################# filter others province ##############################

df.drop(df[(df.latxx < 18.72) | (df.lonxx < 98.94)].index, inplace=True)
df.drop(df[(df.latxx > 18.87) | (df.lonxx > 99.06)].index, inplace=True)

# print(df.head(30))

############################ So now it has only points in Chiang Mai ################

################ Loop to calculate the duration of parking #################################
################ If it's a parking point then the column "parking" will be 1,  if not then will be 0 #####################
list = []
for row in df.itertuples():
    time_stamp = row.duration

    seconds = time_stamp.seconds
    hours = seconds//3600
    minutes = (seconds//60) % 60

    # print("minute = ", minutes)
    # print("hours  = ", hours)
    # print(" ")

    vid = row.vid
    latx = row.latxx
    lonx = row.lonxx

    indicator = 0

    if (minutes >= 30 and hours < 1) or (hours >= 1 and hours < 2) or (hours >= 2 and minutes <= 30):  # 2 hours and half
        indicator = 1
    else:
        indicator = 0

    list.append(indicator)

df['parking'] = list  # assign parking indicator to dataframe


# ตอนนี้ได้จุดจอดแล้ว เหลือนำไปสร้าง feature
# แล้วก็กรองให้เหลือแต่เชียงใหม่แล้ว
#########################################################################################################

######################## Load Dataset of Shopping Point ###############################
shopping_df = pd.read_csv(
    'C:\\Users\\Fantasticboy\\Documents\\GitHub\\Shopping_Point_Classification\\shoppingCenter4.csv')
# print(shopping_df.head(30))

df['shopping_point'] = ""
shopping_list = []
indicator2 = 0
shopping_len = len(shopping_df['latxx'])
iterator = 0

################# Compare coordinate of the dataframe and coordiante of the shopping point to assign the shopping point to column "Shopping" ##############

for row in df.itertuples():
    dflatxx = row.latxx
    dflonxx = row.lonxx
    dfunit_type = row.unit_type
    dfparking = row.parking
    for a, b in zip(shopping_df.latxx, shopping_df.lonxx):
        if dflatxx == a and dflonxx == b:
            shopping_list.append(1)
            iterator = 0
            break

        iterator += 1
        if iterator == shopping_len:
            shopping_list.append(0)

    iterator = 0

############# If the length of 2 dataframe is not match then fill by 0 #################
# shoplist_len = len(shopping_list)
# df_len = len(df['shopping_point'])
# different = df_len - shoplist_len

# if shoplist_len < df_len:
#     for i in range (1,different+1):
#         shopping_list.append(0)
# print(len(shopping_list))
df['shopping_point'] = shopping_list
# print(df.head(30))
# print(df.tail(30))
# print(shopping_list)
# print(df.tail(30))
# print(df[df['shopping_point'] == 1])
# print(df.where(df['shopping_point']==1))
# print(len(df['shopping_point']))

####################################################### Calculate the point that has a lot of cars ###########################
df['is_has_a_lot_of_cars'] = ""
# temp_df['number_of_cars'] = ""
# temp_df = df.groupby(['latxx','lonxx'])
temp = df[df['parking'] == 1].groupby(['latxx', 'lonxx'])['vid'].count()
temp_df1 = pd.DataFrame(data=temp.values, columns=['count'])
temp_df2 = pd.DataFrame(data=temp.index, columns=['latxx_lonxx'])
temp_df = pd.merge(temp_df1, temp_df2, left_index=True, right_index=True)
# temp_df = df[['vid','parking','latxx','lonxx']]
# print(temp_df1.head())
# print(temp_df2.head())
# print(temp_df.head())


temp_df["latxx"] = ""
temp_df["lonxx"] = ""

temp_df[['latxx', 'lonxx']] = pd.DataFrame(
    temp_df['latxx_lonxx'].tolist(), index=temp_df.index)
# df[['vid', 'latxx' ,'lonxx','unit_type']] = pd.DataFrame(df['vid'].tolist(),index=df.index)
# print(temp_df.head(10))

iterator2 = 0
tempdf_len = len(temp_df['latxx'])

for a, b in zip(df.latxx, df.lonxx):

    for row in temp_df.itertuples():
        count = row.count
        temp_latxx = row.latxx
        temp_lonxx = row.lonxx

        if temp_latxx == a and temp_lonxx == b:
            if count >= 15:
                # df[(df['latxx'] == a) & (df['lonxx'] == b)]['is_has_a_lot_of_cars'] = 1
                df.loc[(df['latxx'] == a) & (df['lonxx'] == b),
                       'is_has_a_lot_of_cars'] = 1
                iterator2 = 0
                break

        iterator2 += 1
        if iterator2 == tempdf_len:
            # df[(df['latxx'] == a) & (df['lonxx'] == b)]['is_has_a_lot_of_cars'] = 0
            df.loc[(df['latxx'] == a) & (df['lonxx'] == b),
                   'is_has_a_lot_of_cars'] = 0
            iterator2 = 0

# print(df.head(20))
#######################################################

# Find Correlation
# print(sns.heatmap(df.corr()))

################################## Find Correlation of all feature #############################
print(df.corr().sort_values('shopping_point')['shopping_point'])

################################# Feature Selection ###################################
X = df[['parking', 'unit_type', 'is_has_a_lot_of_cars']]
Y = df['shopping_point']

################################ Split the train and test dataset ########################
# print(X)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=101)

# print(Y_test)
# ############################## Build a model ####################################
clf = LogisticRegression()
clf.fit(X_train, Y_train)

# # clf = LogisticRegression()
# # clf.fit(X, Y)

# # clf = tree.DecisionTreeClassifier()
# # clf.fit(X, Y)

# clf = tree.DecisionTreeClassifier()
# clf.fit(X_train, Y_train)

# clf = svm.SVC()
# clf.fit(X_train,Y_train)

# # clf = svm.SVC()
# # clf.fit(X, Y)

# clf = KNeighborsClassifier(n_neighbors=5)
# clf.fit(X_train,Y_train)
##################### Predict #####################
Y_pred = clf.predict(X_test)
# print(Y_pred)

##################### Report #####################
print("*********Confusion Matrix*********")
cm_labels = df['shopping_point'].unique()
print(cm_labels)
print(confusion_matrix(Y_test, Y_pred, labels=cm_labels))

print("**************Report**************")
print(classification_report(Y_test, Y_pred))

print("************* F1 ***************")
f1 = f1_score(Y_test, Y_pred, average='weighted')
print('F1 = ', f1)
