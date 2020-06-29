# to update Spyder
# conda update spyder

# Pandas (panal data)
# Series - 1d
# dataframe = 2d
# panal data = 3d and so on

# Series
#pandas.Series(data=None, index=None, dtype=None, name=None, copy=False, fastpath=False
import pandas as pd

list1 = [1,2,3,4]
ds1 = pd.Series(list1)
print (ds1)

list1 = [1,2,3,4]
ds2 = pd.Series(list1, index = ['a','b','c','d'])
print (ds2)


x = ['d','e','f','g']
y = [5,6,7,8]
ds3 = pd.Series(y, index = x)
print (ds3)

ds3 = pd.Series(index = y, data = x)

ds2+ds3

x1 = ('d','e','f','g')
y1 = (5,6,7,8)
ds4 = pd.Series(y1, index = x1)

# Dict to series
x2 = ['d','e','f','g']
y2 = [5,6,7,8]
dict1 = dict(zip(x,y))
print (dict1)

ds5 = pd.Series(dict1)
:'a-z'
ds5['e']
ds5[2]
ds5 [1:3]
ds5 ['d':'f']

############## Dataframes (2d)
list1 = [1,2,3,4]
df1 = pd.DataFrame(list1)
print (df1)

dict2 = {'fruits': ['apple', 'muskmelon', 'mangos'],
         'count' : [15,20,25]}
df2 = pd.DataFrame(dict2)
print (df2)

df2[1:3]

#####################series to dataframe
series1 = pd.Series([5,10], index = ['s', 't'])
df3 = pd.DataFrame(series1)
print (df3)

# numpy to dataframe
import numpy as np
num_arr= np.array([1,2,3,4])
print (pd.DataFrame(num_arr))

import numpy as np
num_arr1= np.array([[1,2,3,4],['a','b','c','d']])
df5 = pd.DataFrame({'names': num_arr1[1], 
                    'Numbers':num_arr1[0]})

###=====================================
A = [1,2,3,4]
B = [5,6,7,8]
C = [1,2,3,4]
D = [5,6,7,3]
E = [1,2,3,4]

df6 = pd.DataFrame([A,B,C,D,E], ['a','b','c','d','e'], [1,2,3,'a'])
print (df6)

df6['aa'] = [1,5,3,4,2]
df6['sum'] = df6[4] + df6['aa']

df6.drop('sum', axis = 1)
df6.drop('e', axis = 0)
df6.drop('e')

df6.drop('sum', axis = 1, inplace = True)

print (df6)


############ Recording


# tab or ctrl+space

# conditional acessing
df6

df6[df6>3]

df6[df6[3]>3]
df6[df6[4]>3]
df6[df6[1]>3]
df6[df6[2]>3]

''
df6[df6[3]>3][[3]]
df6[df6[3]>3][['a']]

df6[df6[3]>3][['a', 3]]

# & and or (XOR)

df6[(df6[3]>3) & (df6['a']>5)]

df6[(df6[3]>3) | (df6['a']>5)]

df6[(df6[3]>3) ^ (df6['a']>5)]


######## Missing data
# Nan
import numpy as np
dict1 = {'a': [1,2,3,4,5,np.nan],
         'b': [5,6,7,8,np.nan, np.nan],
         'c': [9,1,2,np.nan,np.nan,np.nan],
         'd': [6,np.nan,np.nan,np.nan,np.nan,np.nan],
         'e': [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]}

df1 = pd.DataFrame(dict1)
print (df1)

df1.dropna()
df1.dropna(axis = 'rows')
df1.dropna(axis = 1)
df1c = df1.dropna(axis = 'columns')

df1.dropna(axis = 0, how = 'all', inplace = True)
df1.dropna(axis = 1, inplace = True)

df1.fillna(2)

df1['a'].fillna(value = df1['a'].mean(), inplace = True)

df1['a'].fillna(value = df1['c'].mean())
df1.fillna(value = df1.mean(), inplace = True)

df1.drop('e', axis = 1)
df1.drop(columns = 'e')

df1['a'].fillna(value = df1['a'].mean())
df1.fillna(value = df1.mean())

df1.drop('d', axis = 1)
df1.drop(columns = 'd')

df1.dropna(axis = 0)

# Merge, Join, concatenation

import pandas as pd

player = ['Sachin', 'kohli', 'dhoni']
score = [55, 59, 62]
title = ['batsman', 'captain', 'wkt']

df1 = pd.DataFrame({'player': player, 'score': score, 
                   'title':title})
print (df1)


player1 = ['kohli', 'Bumrah', 'shami']
wicket = [1, 5, 2]
title1 = ['batsman', 'bowler', 'bowler']

df2 = pd.DataFrame({'Player':player1, 'Wickets': wicket,
                    'title': title1})
print(df2)

# Merge (Inner, left, right, outer)
pd.merge(df1, df2)
# or
df1.merge(df2)

pd.merge(df1, df2, on = 'title', how = 'inner')
df1.merge(df2, on = 'title', how = 'inner')

pd.merge(df1, df2, how = 'left')
pd.merge(df1, df2, how = 'right')
pd.merge(df1, df2, how = 'outer')

# join (Inner, left, right, outer)
# player = ['Sachin', 'kohli', 'dhoni']
# score = [55, 59, 62]
# title = ['batsman', 'captain', 'wkt']

# df1_copy = pd.DataFrame({'player': player, 'score': score, 
#                    'title':title})
# print (df1_copy)


# player1 = ['Sachin', 'Bumrah', 'shami']
# wicket = [1, 5, 2]
# title1 = ['batsman', 'bowler', 'bowler']

# df2_copy = pd.DataFrame({'player':player1, 'Wickets': wicket,
#                     'title1': title1})
# print(df2_copy)

# df1_copy.join(df2_copy)

# df1_copy.join(df2_copy, how = 'inner')
# df1_copy.join(df2_copy, how = 'left')
# df1_copy.join(df2_copy, how = 'right')

############ Concatenation
df1
df2
pd.concat([df1, df2])
pd.concat([df1, df2], axis = 0)
pd.concat([df1, df2], axis = 1)

# Data Analytics

import pandas as pd
df = pd.read_excel('weather.xlsx', 'Sheet1')
df1 = pd.read_excel('C:\\Users\\DELL\\Desktop\\Python DS\\Pandas DataSets\\weather.xlsx')

df = pd.read_excel('weather.xlsx', skiprows = 1)
df = pd.read_excel('weather.xlsx', skiprows = 2)

df = pd.read_excel('weather.xlsx', header = 2)
df = pd.read_excel('weather.xlsx', header = 2)
df = pd.read_excel('weather.xlsx', header = None)

df = pd.read_excel('weather.xlsx', header = 0, 
                   names = ['day', 'temp', 'air', 'event1' ])

df = pd.read_excel('weather.xlsx', nrows = 3)

df.columns
# df.rows

df.shape

rows, columns = df.shape
print (rows)
print (columns)

df.head
df.tail
`
df.head(2)
df.tail(2)
df[2:4]

df.info (null_counts = True)

# to save as an excel file
# file will be saved in directory
df.to_excel('test.xlsx')
df.to_csv('test1.csv')

df.temperature
# or
df['temperature']
df[['temperature', 'windspeed']]
df[['temperature', 'windspeed', 'event']]

df['temperature'].max()
df['temperature'].min()

df.mean()
df['temperature'].mean()

df.median()
df['temperature'].median()

df.mode()
df['temperature'].mode()

df.std()
df['temperature'].std()

df.describe()

# Condtional accessing

df[df.temperature >= 32]

df[df['temperature'] >= 32]


df[(df['temperature'] >= 31) & (df['temperature'] <= 34)]
df[(df['temperature'] >= 31) | (df['temperature'] <= 34)]
df[(df['temperature'] >= 31) ^ (df['temperature'] <= 34)]


df[df.temperature == df.temperature.max()]
df[df.temperature == df.temperature.min()]

df[df['temperature'] == df['temperature'].max()]

#### to call multiple columns
df[['temperature', 'windspeed']][df.temperature == df.temperature.max()]

df[['temperature', 'windspeed', 'event']][df.temperature == df.temperature.max()]


df.index

df.set_index('day', inplace = True)

df.reset_index(inplace = True)

df.to_csv('test1231.csv')

df.set_index('event', inplace = True)

df.to_csv('test12312.csv', columns = ['temperature', 'windspeed'])

df.to_csv('expo1.csv', columns = ['temperature', 'windspeed'])

df1 = df.rename(columns = {'temperature':'temp', 'windspeed':'Air'})

df.to_csv('new.csv', header = False)

######### Missing Data
import pandas as pd

df = pd.read_excel('weather.xlsx', parse_dates = ['day'])
type(df.day[0])

df.set_index('day', inplace = True)

### fill na values
df1 = df.fillna(0)

df.temperature = df.temperature.fillna(df.temperature.mean())

df.temperature = df.temperature.fillna(df.temperature.median())

# df.temperature = df.temperature.fillna(df.temperature.mode())

df1 = df.fillna({'temperature':20,
                'windspeed': 5,
                'event': 'No Event'})

df2 = df.fillna(method = 'ffill')

df2 = df.fillna(method = 'ffill', axis = 1)

df.temperature = df.temperature.fillna[df.temperature.mode()]

df3 = df.fillna(method = 'backfill', axis = 0)
df3 = df.fillna(method = 'bfill', axis = 0)

df2 = df.fillna(method = 'ffill', limit = 1)

# interpolate
df3 = df.interpolate()

# dropna
df4 = df.dropna()
df4 = df.dropna(how = 'all')

df5 = df.dropna(thresh = 1)
df5 = df.dropna(thresh = 2) # minimum 2 values

df6 = df.dropna(how = 'all', thresh = 2)

########
dt = pd.date_range('2020-01-01', '2020-01-06')
# dt = pd.DatetimeIndex(dt)
df7 = df.reindex(dt)

######## Cor relation
df8 = df[['temperature', 'windspeed']].corr()
df8 = df[['temperature', 'windspeed', 'event']].corr()
    
df8 = df[['temperature', 'ice']].corr()


# import pandas as pd
# dfc = pd.read_excel('weather.xlsx', 'Sheet2')
# df8c = dfc[['temp', 'ice']].corr()  
    
df.info(null_counts = True)
df.info()
df.info(null_counts = False)

# Data Manipulation
# loc (location) and iloc (integer location)

df.iloc[:,3]
df.iloc[:,:]
df.iloc[:5,1:3]
df.iloc[2:5,1::2]

df.loc[:5, 'temperature']
df.loc[:5, 'temperature':'event']

# Ascending and descending
df.sort_values(by = 'windspeed')
df.sort_values(by = 'windspeed', ascending = False)

##### Group by
import pandas as pd
df = pd.read_csv('weather1.csv')

grp = df.groupby('city')

for city, city_df in grp:
    print (city)
    print (city_df)

grp.get_group('Delhi')
grp.get_group('Amritsar')

grp.max()
grp.min()
grp.mean()
grp.describe().transpose()

# Pivot table
import pandas as pd
df = pd.read_csv('weather1.csv')

df.pivot_table(index = 'day', columns = 'city')
df.pivot_table(index = 'day', columns = 'city', 
               aggfunc = 'sum')
df.pivot_table(index = 'city', columns = 'day', 
               aggfunc = 'count')

df.pivot_table(index = 'day', columns = 'city', 
               aggfunc = 'sum', margins = True)

df.pivot_table(index = 'day', columns = 'city', 
               aggfunc = 'count', margins = True)

df.pivot_table(index = 'day', columns = 'city',
               aggfunc = 'sum', margins = True)

df.pivot_table(index = 'day', columns = 'city',
               aggfunc = '', margins = True)


df.pivot_table(index = 'day', columns = 'city', margins = True)









    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



