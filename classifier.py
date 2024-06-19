import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

data_dict = pickle.load(open('data.pickle','rb'))
# Inspect the data
max_length = max(len(seq) for seq in data_dict['data'])
# Function to pad sequences
def pad_sequences(sequences, maxlen):
    padded_sequences = np.zeros((len(sequences), maxlen))
    for i, seq in enumerate(sequences):
        padded_sequences[i, :len(seq)] = seq
    return padded_sequences

# Pad the data
data = pad_sequences(data_dict['data'], max_length)
labels=np.asarray(data_dict['labels'])

#splitting dataset in training set and test set
#stratify means we are going to keep same proportion of labels
X_train,X_test,y_train,y_test=train_test_split(data,labels,test_size=0.2,shuffle=True,stratify=labels)
model=RandomForestClassifier()
model.fit(X_train,y_train)
y_predict=model.predict(X_test)
score=accuracy_score(y_predict,y_test)
print(score*100)

f=open('model.p','wb')
pickle.dump({'model' : model } , f)
f.close()







