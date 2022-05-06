def  readData(db="IP",PCAID=0,WHITE=0):
     import numpy    as np
     import scipy.io as scio
     
     if db=="IP":
         clas    = 16
         NUM     = 10249  #数据中的有效样本个数,12类10062,16类是10249
         datafile         = scio.loadmat("data\Indian_pines_corrected.mat")
         labelfile        = scio.loadmat("data\Indian_pines_gt.mat")
         data             = datafile['indian_pines_corrected']  #data.shape = [145,145,200]
         label            = labelfile['indian_pines_gt']        #label [145,145]
     
     if db=="UP":
         clas    = 9
         NUM     = 42776  #数据中的有效样本个数,12类10062,16类是10249     
         datafile         = scio.loadmat("data\PaviaU.mat")
         labelfile        = scio.loadmat("data\PaviaU_gt.mat")
         data             = datafile['paviaU']  #data.shape = [145,145,200]
         label            = labelfile['paviaU_gt']        #label [145,145]


     data             = np.float32(data)
     label            = np.int32(label)
     data             = data/data.max()

#============================================================================
     from sklearn.preprocessing import StandardScaler
     shapeor      = data.shape
     data         = data.reshape(-1, data.shape[-1])
     if PCAID == 1:
          from sklearn.decomposition import PCA
          num_components   = 64
          data         = PCA(n_components=num_components).fit_transform(data)
          shapeor      = np.array(shapeor)
          shapeor[-1]  = num_components
     if WHITE ==1:
          data         = StandardScaler().fit_transform(data)
     data         = data.reshape(shapeor)
# =============================================================================    
    
    # 记录有效标记样本的坐标和标签，格式为（横坐标，纵坐标，标签值）
     rows,cols,bands  = data.shape
     samples          = np.zeros((NUM,3),dtype="int32")#NUM为有效样本点个数
     index            = 0
     for i in range(0,rows):
             for j in range(0,cols):
                 if label[i][j] != 0:
                      samples[index][0]     = i
                      samples[index][1]     = j
                      samples[index][2]     = label[i][j]-1#将有效标签写成0-15，共16类
                      index                 = index+1
     return data,samples,clas

def splitData(samples,c,r=0.1,rs=1024):
     from sklearn.model_selection    import train_test_split as ts
     import numpy
          
     #划分第0类的样本
     tmpS                      = samples[samples[:,-1]==0]
     numS                      = tmpS.shape[0]
     if r<1:
         num                       = int((numS)*r+0.5)#因为总样本中5，四舍五入
     else:
         num = r
         if num>numS:
             num = (numS+1)//2#因为总样本中5，四舍五入
     
     trainX,testX,trainY,testY = ts(tmpS,tmpS,train_size = num,random_state = rs)
     trainD                    = trainX
     testD                     = testX
     print(tmpS.shape[0],trainX.shape[0])
     
     #划分其他类别样本
     for i in range(1,c):
        tmpS                      = samples[samples[:,-1]==i]
        numS                      = tmpS.shape[0]
        if r<1:
            num                       = int((numS)*r+0.5)#因为总样本中5，四舍五入
        else:
            num = r
            if num>numS:
                num = (numS+1)//2#因为总样本中5，四舍五入
        trainX,testX,trainY,testY = ts(tmpS,tmpS,train_size = num,random_state = rs)
        trainD = numpy.vstack((trainD,trainX))    
        testD  = numpy.vstack((testD,testX))
        print(tmpS.shape[0],trainX.shape[0])

# =============================================================================
#      #保存选取样本的坐标
#     import scipy.io
#     scipy.io.savemat('data_Pos.mat',{'trainD':trainD,'testD':testD})
# =============================================================================
     
     return trainD,testD

def getData(data,trD,block=9):
     import numpy

     num             = trD.shape[0]
     rows,cols,bands = data.shape
     bk              = int(block//2)
     
     
     # 扩展填充
     tmpData                              = numpy.zeros((rows+bk*2,cols+bk*2,bands),dtype="float32")
     tmpData[bk:bk+rows,bk:bk+cols,:]     = data[:,:,:]
     tmpData[0:bk,:,:]                    = numpy.expand_dims(tmpData[bk,:,:],axis=0).repeat(bk,axis=0)
     tmpData[rows+bk:rows+2*bk,:,:]       = numpy.expand_dims(tmpData[bk+rows,:,:],axis=0).repeat(bk,axis=0)
     tmpData[:,0:bk,:]                    = numpy.expand_dims(tmpData[:,bk,:],axis=1).repeat(bk,axis=1)
     tmpData[:,cols+bk:cols+2*bk,:]       = numpy.expand_dims(tmpData[:,bk+cols,:],axis=1).repeat(bk,axis=1)
     
     
     # 提取样本点
     trX             = numpy.zeros((num,block,block,bands),dtype="float32")
     trY             = numpy.zeros(num,dtype="int32")
          
     for i in range(num):
          posX             = trD[i,0]+bk
          posY             = trD[i,1]+bk
          trX[i,:,:,:]     = tmpData[(posX-bk):(posX+bk+1),(posY-bk):posY+bk+1,:]
          trY[i]           = trD[i,2]
     
     return trX,trY