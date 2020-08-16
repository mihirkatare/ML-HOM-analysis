import numpy as np
import joblib
import pandas as pd
import pickle
def invt(arr, scaler):
    d1,d2,d3 = arr.shape
    arr = np.reshape(arr, (d1*d2, d3))
    arr = scaler.inverse_transform(arr)
    arr = np.reshape(arr, (d1, d2, d3))
    return arr

def predict(model, doo, type, predsavepath, scaler = None):
    if(type == 1):
        #Denoising Autoencoder
        for i in range(doo.__len__()):
            y = model.predict(doo.__getitem__(i))
            if(scaler!=None):
                y = invt(y, scaler)
            pd.DataFrame(y[:,:,0].T).to_csv(predsavepath + str(doo.indexes[i][1]) + "_" + str(doo.indexes[i][0]) + ".csv", header=False, index=False)
    if(type == 0):
        results = dict()
        for i in range(doo.__len__()):
            y = model.predict(doo.__getitem__(i))
            if(int(doo.indexes[i][1][0]) not in list(results.keys())):
                results[int(doo.indexes[i][1][0])] = []
            results[int(doo.indexes[i][1][0])].append(y[0,0])
        with open(predsavepath + 'results.pickle', 'wb') as handle:
            pickle.dump(results, handle)
