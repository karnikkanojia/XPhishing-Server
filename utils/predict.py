import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import pickle
from utils.URLFeaturizer import UrlFeaturizer
from tensorflow.keras.models import load_model
from utils.metrics import (
    f1_m,
    precision_m,
    recall_m,
)

order = ['bodyLength', 'bscr', 'dse', 'dsr', 'entropy', 'hasHttp', 'hasHttps',
       'has_ip', 'numDigits', 'numImages', 'numLinks', 'numParams',
       'numTitles', 'num_%20', 'num_@', 'sbr', 'scriptLength', 'specialChars',
       'sscr', 'urlIsLive', 'urlLength']

def predict(url):
    a = UrlFeaturizer(url).run()
    test = []
    for i in order:
        test.append(a[i])

    encoder = LabelEncoder()
    encoder.classes_ = np.load('models/lblenc.npy',allow_pickle=True)
    scalerfile = 'models/scaler.sav'
    scaler = pickle.load(open(scalerfile, 'rb'))
    model = load_model("models/Model_v1.h5", custom_objects={'f1_m':f1_m,"precision_m":precision_m, "recall_m":recall_m})
    test = pd.DataFrame(test).replace(True,1).replace(False,0).to_numpy().reshape(1,-1)
    predicted = np.argmax(model.predict(scaler.transform(test)),axis=1)
    return encoder.inverse_transform(predicted)[0]

if __name__ == "__main__":
   print(predict('www.google.com'))