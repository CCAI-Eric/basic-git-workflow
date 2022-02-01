#!/usr/bin/env python
# coding: utf-8

# ___
#
# <a href='https://www.datamics.com/courses/online-courses/'><img src='../DATA/bg_datamics_top.png'/></a>
# ___
# <center><em>© Datamics</em></center>
# <center><em>Besuche uns für mehr Informationen auf <a href='https://www.datamics.com/courses/online-courses/'>www.datamics.com</a></em></center>
#
# # RNN-Beispiel für Sinuswellen

# In[1]:


import pandas as pd
import numpy as np

# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# ## Daten
#
# Lasst uns NumPy verwenden, um eine Sinuswelle zu erzeugen.

# In[2]:


x = np.linspace(0, 50, 501)
y = np.sin(x)


# In[3]:


x


# In[4]:


y


# In[5]:


plt.plot(x, y)


# Let's turn this into a DataFrame

# In[6]:


df = pd.DataFrame(data=y, index=x, columns=["Sine"])


# In[7]:


df


# ## Aufteilung in Trainings- und Testdaten
#
# Beachte! Dies ist sehr verschieden von unserer üblichen Aufteilungsmethode!

# In[8]:


len(df)


# In[9]:


test_percent = 0.1


# In[10]:


len(df) * test_percent


# In[11]:


test_point = np.round(len(df) * test_percent)


# In[12]:


test_ind = int(len(df) - test_point)


# In[13]:


test_ind


# In[14]:


train = df.iloc[:test_ind]
test = df.iloc[test_ind:]


# In[15]:


train


# In[16]:


test


# ## Daten skalieren

# In[17]:


from sklearn.preprocessing import MinMaxScaler


# In[18]:


scaler = MinMaxScaler()


# In[19]:


# IGNORE WARNING ITS JUST CONVERTING TO FLOATS
# WE ONLY FIT TO TRAININ DATA, OTHERWISE WE ARE CHEATING ASSUMING INFO ABOUT TEST SET
scaler.fit(train)


# In[20]:


scaled_train = scaler.transform(train)
print(scaled_train.shape)
scaled_test = scaler.transform(test)


# # TimeSeries (Zeitserie) erzeugen
#
# Diese Klasse verwendet eine Sequenz von in gleichen Intervallen
# erzeugten Datenpunkten, zusammen mit TimeSeries-Parametern wie
# stride (Schrittweite), length of history (Dauer), etc., um
# Batches für Training und Validierung zu erzeugen.
#
# #### Argumente
#     data: Indizierbarer Generator (z.B. Liste oder NumPy-Array)
#         mit konsekutiven Datenpunkten (Zeitschritten). Die
#         Daten sollten in 2D vorliegen mit Achse0 als Zeitdimension.
#     targets: Zu den Zeitschritten korrespondierende Ziele in `data`.
#         Sollte die gleiche Länge wie `data` haben.
#     length: Länge der Ausgabesequenz (in Anzahl Zeitschritte).
#     sampling_rate: Periode zwischen aufeinanderfolgenden,
#         individuellen Zeitschritten. Für die Rate `r`werden die
#         Zeitschritte `data[i]`, `data[i-r]`, ... `data[i - length]
#         verwendet, um die Samplesequenz zu erzeugen.
#     stride: Periode zwischen zwei aufeinanderfolgenden Ausgabesequenzen.
#         Für stride `s` werden die Ausgabesequenzen
#         data[i]`, `data[i+s]`, `data[i+2*s]`, etc. erzeugt.
#     start_index: Datenpunkte früher als `start_index` werden nicht für
#         Ausgabesequenzen verwendet. Dies ist nützlich, um einen Teil
#         der Daten für Test und Validierung übrig zu lassen.
#     end_index: Datenpunkte später als `end_index` werden nicht für
#         Ausgabesequenzen verwendet. Dies ist nützlich, um einen Teil
#         der Daten für Test und Validierung übrig zu lassen.
#     shuffle: Bestimmt, ob die Samples gemischt oder in chronologischer
#         Reihenfolge geordnet werden sollen.
#     reverse: Bool'sch: Wenn `true` wird die chronologische Reihenfolge
#         invertiert.
#     batch_size: Anzahl von Timeseries-Samples pro Batch (evtl.
#         ausgenommen die letzte Batch)

# In[ ]:


from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


# In[22]:


# scaled_train


# In[23]:


# define generator
length = 2  # Length of the output sequences (in number of timesteps)
batch_size = 1  # Number of timeseries samples in each batch
generator = TimeseriesGenerator(
    scaled_train, scaled_train, length=length, batch_size=batch_size
)


# In[24]:


len(scaled_train)


# In[25]:


len(generator)  # n_input = 2


# In[26]:


# scaled_train


# In[27]:


# What does the first batch look like?
X, y = generator[0]


# In[28]:


print(f"Given the Array: \n{X.flatten()}")
print(f"Predict this y: \n {y}")


# In[29]:


# Let's redefine to get 10 steps back and then predict the next step out
length = 10  # Length of the output sequences (in number of timesteps)
generator = TimeseriesGenerator(scaled_train, scaled_train, length=length, batch_size=1)


# In[30]:


# What does the first batch look like?
X, y = generator[0]


# In[31]:


print(f"Given the Array: \n{X.flatten()}")
print(f"Predict this y: \n {y}")


# In[32]:


length = 50  # Length of the output sequences (in number of timesteps)
generator = TimeseriesGenerator(scaled_train, scaled_train, length=length, batch_size=1)


# Du bist jetzt in der Lage, die Länge zu verändern, so dass es für deine Timeseries Sinn ergibt!

# ### Modell erzeugen

# In[33]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN


# In[34]:


# We're only using one feature in our time series
n_features = 1


# In[35]:


# define model
model = Sequential()

# Simple RNN layer
model.add(SimpleRNN(50, input_shape=(length, n_features)))

# Final Prediction
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")


# In[36]:


model.summary()


# In[37]:


# fit model
model.fit_generator(generator, epochs=5)


# In[38]:


model.history.history.keys()


# In[39]:


losses = pd.DataFrame(model.history.history)
losses.plot()


# ## Evaluation mit Testdaten

# In[40]:


first_eval_batch = scaled_train[-length:]


# In[41]:


first_eval_batch


# In[42]:


first_eval_batch = first_eval_batch.reshape((1, length, n_features))


# In[43]:


model.predict(first_eval_batch)


# In[44]:


scaled_test[0]


# Lasst uns jetzt diese Logik in einen for-Loop packen, um die Zukuft der gesamten Testreihe vorherzusagen.
#
# ----

# In[45]:


test_predictions = []

first_eval_batch = scaled_train[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))


# In[46]:


current_batch.shape


# In[47]:


current_batch


# In[48]:


np.append(current_batch[:, 1:, :], [[[99]]], axis=1)


# **BEACHTE: ÜBERPRÜFE HIER DIE AUSGABE UND IHRE DIMENSIONEN GENAU. FÜGE EIGENE PRINT()-STATEMENTS HINZU, UM ZU VERSTEHEN, WAS WIRKLICH VORGEHT!!**

# In[49]:


test_predictions = []

first_eval_batch = scaled_train[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))

for i in range(len(test)):

    # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])
    current_pred = model.predict(current_batch)[0]

    # store prediction
    test_predictions.append(current_pred)

    # update batch to now include prediction and drop first value
    current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)


# In[50]:


test_predictions


# In[51]:


scaled_test


# ## Inverse Transformationen und Vergleich

# In[52]:


true_predictions = scaler.inverse_transform(test_predictions)


# In[53]:


true_predictions


# In[54]:


test


# In[55]:


# IGNORE WARNINGS
test["Predictions"] = true_predictions


# In[56]:


test


# In[57]:


test.plot(figsize=(12, 8))


# ## Füge einen frühzeitigen Abbruch und Validierungsgenerator hinzu

# In[58]:


from tensorflow.keras.callbacks import EarlyStopping


# In[59]:


early_stop = EarlyStopping(monitor="val_loss", patience=2)


# In[60]:


length = 49
generator = TimeseriesGenerator(scaled_train, scaled_train, length=length, batch_size=1)


validation_generator = TimeseriesGenerator(
    scaled_test, scaled_test, length=length, batch_size=1
)


# # LSTMS

# In[61]:


# define model
model = Sequential()

# Simple RNN layer
model.add(LSTM(50, input_shape=(length, n_features)))

# Final Prediction
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")


# In[62]:


model.fit_generator(
    generator, epochs=20, validation_data=validation_generator, callbacks=[early_stop]
)


# In[63]:


test_predictions = []

first_eval_batch = scaled_train[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))

for i in range(len(test)):

    # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])
    current_pred = model.predict(current_batch)[0]

    # store prediction
    test_predictions.append(current_pred)

    # update batch to now include prediction and drop first value
    current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)


# In[82]:


# IGNORE WARNINGS
true_predictions = scaler.inverse_transform(test_predictions)
test["LSTM Predictions"] = true_predictions
test.plot(figsize=(12, 8))


# # Vorhersage
#
# Vorhersage einer unbekannten Reihe. Wir sollten alle unsere Daten für die Vorhersage nutzen!

# In[65]:


full_scaler = MinMaxScaler()
scaled_full_data = full_scaler.fit_transform(df)


# In[66]:


length = 50  # Length of the output sequences (in number of timesteps)
generator = TimeseriesGenerator(
    scaled_full_data, scaled_full_data, length=length, batch_size=1
)


# In[67]:


model = Sequential()
model.add(LSTM(50, input_shape=(length, n_features)))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mse")
model.fit_generator(generator, epochs=6)


# In[68]:


forecast = []

first_eval_batch = scaled_full_data[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))

for i in range(len(test)):

    # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])
    current_pred = model.predict(current_batch)[0]

    # store prediction
    forecast.append(current_pred)

    # update batch to now include prediction and drop first value
    current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)


# In[69]:


forecast = scaler.inverse_transform(forecast)


# In[70]:


# forecast


# In[71]:


df


# In[72]:


len(forecast)


# In[73]:


50 * 0.1


# In[74]:


forecast_index = np.arange(50.1, 55.1, step=0.1)


# In[75]:


len(forecast_index)


# In[76]:


plt.plot(df.index, df["Sine"])
plt.plot(forecast_index, forecast)


# # Gut gemacht!
