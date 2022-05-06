def Position(L):  
     import numpy as np2
     pos  = np2.zeros((L,L))
     xc   = L//2
     yc   = L//2
     for h in range(L):
          for k in range(L):
               pos[h,k]=np2.max([np2.abs(h-xc),np2.abs(k-yc)])+1
     print(pos)
     pos  = pos.reshape((1,L*L)) 
     return pos

def CTN(input_shape=(11,11,64),num_clas=16):  
     from keras.layers       import Dense,Input,Reshape,GlobalAveragePooling1D,Add
     from Transformer        import CNNT,GELU
     from keras.models       import Model
     
     x_input = Input(shape=input_shape)
     x_pos   = Input(shape=(input_shape[0],input_shape[1],1))
     
     # 1. 向量嵌入
     x       = x_input     
     y       = x_pos
     y       = Dense(64,activation=GELU)(y)
     x       = Add()([x,y])
     
     # 2. Transformer
     x       = CNNT(x)
     x       = CNNT(x)
          
     x       = Reshape((input_shape[0]*input_shape[1],input_shape[2]))(x)
     x       = GlobalAveragePooling1D()(x)
     
     # 3. classification
     x       = Dense(32,activation=GELU)(x)
     x       = Dense(num_clas,activation='softmax')(x)
     model   = Model(inputs=[x_input,x_pos], outputs=x)
     
     from keras.optimizers   import Adam
     op      = Adam(lr=0.001)
     model.compile(optimizer=op,loss='categorical_crossentropy',metrics=['acc'])
     model.summary()
     
     return model
 
#==============================================================================
#  程序入口
#==============================================================================
import func3d as fc
import numpy as np
import keras

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1" 
DB    =  'IP'
RATE  =  0.1
BLOCK =  15


data,samples,num_clas = fc.readData(DB,PCAID=1,WHITE=1)

for k in range(10):  
    
    # 1.get train data
    trainD,testD          = fc.splitData(samples,num_clas,r=RATE,rs=k)
    trX,trY               = fc.getData(data,trainD,block=BLOCK)
    num,rows,cols,bands   = trX.shape
    pos                   = Position(L=BLOCK)
    X_pos                 = np.dot(np.ones((num,1)),pos)
    X_pos                 = X_pos.reshape((num,BLOCK,BLOCK,1))
    
    # 2.build model 
    input_shape           = (rows,cols,bands)
    model                 = CTN(input_shape,num_clas)
    
    # 3.train model
    X_train               = trX.reshape(num,rows,cols,bands)
    Y_train               = keras.utils.to_categorical(trY,num_clas)
    
    from keras.callbacks import ModelCheckpoint
    checkpoint            = ModelCheckpoint(filepath  = 'bestModel.hdf5',
                                          monitor = 'loss', mode='min',
                                          save_weights_only = True,
                                          save_best_only    = True,
                                          verbose           = 0)

    model.fit([X_train,X_pos],Y_train,epochs= 300,batch_size= 32,callbacks= [checkpoint])
    
    # 4.test samples
    model.load_weights('bestModel.hdf5')
    teX,teY      = fc.getData(data,testD,block=BLOCK)
    X_pos2       = np.dot(np.ones((teX.shape[0],1)),pos)
    X_pos2       = X_pos2.reshape((teX.shape[0],BLOCK,BLOCK,1))
    
    X_test       = teX.reshape(-1,rows,cols,bands)
    Y_test       = keras.utils.to_categorical(teY,num_clas)
    score        = model.evaluate([X_test,X_pos2], Y_test)        
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
















 
 
 
 
 
 
 
 
 
 
 
 