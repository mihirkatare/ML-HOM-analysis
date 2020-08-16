import time
start_time = time.time()
import numpy as np
from data import data
from gen import DataGenerator
from gen2 import DG2
from model import model
from genparser import parse
import os
import helpers
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from predictions import predict

Inputfile = "Input Examples/Input_growthid_train.txt"
vd = parse(Inputfile)
print(vd)

'''SET VARIABLES'''
if(vd['UseGPU'] == False): os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
bsize = vd['batch_size']
dir = vd['datasets']
ksize = vd['kernel_size']
nskip = vd['bunches_to_skip']
timebloc = vd['truncate_at_timestep']


'''Provide paths to data'''
obj = data()
obj.addpath(dir)
exarr = helpers.skipcol(obj.load_example_array(), skip = nskip)
print(obj.details())
if(vd['UseObjective'] == 3):
    exit()

'''Choose and compile Model'''
foo = model()
foo.metadata(ex_array=exarr, skip = nskip, _timebloc = timebloc)

if(vd['UseObjective'] == 0):
    if(vd['UseCase'] == 0):
        foo.conv_growth(_batch_size=bsize)
    if(vd['UseCase'] == 1):
        foo.conv_da(_batch_size=bsize, kernelsize = ksize)
    foo.compile()

elif(vd['UseObjective'] == 1 or vd['UseObjective'] == 2):
    foo.load_savedmodel(vd['savedpath'])
    print("loaded")
    if(vd['UseObjective'] == 2):
        foo.model.save_weights("post/temp.h5")
        if(vd['UseCase'] == 0):
            foo.conv_growth(_batch_size=None)
        if(vd['UseCase'] == 1):
            foo.conv_da(_batch_size=None, kernelsize = ksize)
        foo.model.load_weights("post/temp.h5")
        print(foo.model.summary())
#
#
'''Inititalize custom data generator'''
if(vd['UseObjective'] == 2):
    if(vd['UseCase'] == 0):
        doo = DataGenerator(_n_ts=foo.n_ts, _n_feat=foo.n_feat, obj=obj, _skip = foo.skip, batch_size=bsize, type="mode_growth", _timebloc = timebloc
        , n_scans_use=vd['num_scans_use'], _shuffle = False, _grpath = vd['path_to_gr'], x_only = True)

    if(vd['UseCase'] == 1):
        doo = DG2(_n_ts=foo.n_ts, _n_feat=foo.n_feat, obj=obj, _skip = foo.skip, batch_size=bsize, type="denac2"
        , _timebloc = timebloc, n_scans_use=vd['num_scans_use'], _shuffle = False
        , _reduce = vd['reduced_timesteps'], _avgpath = vd['path_to_avg'], x_only = True)
else:
    if(vd['UseCase'] == 0):
        doo = DataGenerator(_n_ts=foo.n_ts, _n_feat=foo.n_feat, obj=obj, _skip = foo.skip, batch_size=bsize, type="mode_growth", _timebloc = timebloc
        , n_scans_use=vd['num_scans_use'], _shuffle = vd['shuffle_samples'], _grpath = vd['path_to_gr'])

    if(vd['UseCase'] == 1):
        doo = DG2(_n_ts=foo.n_ts, _n_feat=foo.n_feat, obj=obj, _skip = foo.skip, batch_size=bsize, type="denac2"
        , _timebloc = timebloc, n_scans_use=vd['num_scans_use'], _shuffle = vd['shuffle_samples'], _reduce = vd['reduced_timesteps'], _avgpath = vd['path_to_avg'])

'''Callbacks and model fit'''
if(vd['UseObjective'] == 0 or vd['UseObjective'] == 1):
    chkpt = ModelCheckpoint(vd['Modelpath'],monitor='loss',mode='min',save_best_only=True,verbose=1)
    lrs = LearningRateScheduler(helpers.scheduler)

    foo.model.fit(doo, epochs = vd['epochs'], verbose=1, callbacks=[chkpt, lrs], max_queue_size = vd['max_queue'])

'''Predictions'''
if(vd['UseObjective'] == 2):
    predict(model = foo.model, doo = doo, type = vd['UseCase'], predsavepath = vd['predsavepath'], scaler = doo.scaler)

# '''Testing and Validation'''
# # test(doo, type = "denac" ,model_path = "post/model_den2.h5", item = 5772)
