from cmapssdata import CMAPSSDataset
import numpy as np
'''
we don't have random in this subset for the training and testing subset
'''
class data():
    def __init__(self, batches, data, label, batchSize ):
        self.batches = batches
        self.trainDataFeature = data
        self.trainDataLabel = label
        self.batchSize = batchSize
        self.batchpoint = 0

    def nextBatch(self):
        if self.batchpoint >= self.batches - 1:
            self.batchpoint = 0
            return self.trainDataFeature[(self.batches - 1) * self.batchSize: self.batches * self.batchSize, :, :], \
                   self.trainDataLabel[(self.batches - 1) * self.batchSize: self.batches * self.batchSize]


        else:
            self.batchpoint = self.batchpoint + 1
            return self.trainDataFeature[(self.batchpoint - 1) * self.batchSize:self.batchpoint * self.batchSize, :, :], \
                   self.trainDataLabel[(self.batchpoint - 1) * self.batchSize:self.batchpoint * self.batchSize]
class dataLoader():
    def __init__(self, bs = 10, sl = 50, fd_number = '4'):
        self.batchSize = bs
        self.sequence_length = sl
        cmpass = CMAPSSDataset(fd_number=fd_number, batch_size=bs, sequence_length=sl)
        train_Data = cmpass.get_train_data()
        train_DataFeature = cmpass.get_feature_slice(train_Data)
        train_DataLabel = cmpass.get_label_slice(train_Data)

        test_Data = cmpass.get_test_data()
        test_DataFeature = cmpass.get_feature_slice(test_Data)
        test_DataLabel = cmpass.get_label_slice(test_Data)

        index = np.random.permutation(train_DataFeature.shape[0])
        self.trainDataFeature = train_DataFeature[index, :, :]
        self.trainDataLabel = train_DataLabel[index]
        self.train_batches = self.trainDataFeature.shape[0]//self.batchSize

        index = np.random.permutation(test_DataFeature.shape[0])
        self.testDataFeature = test_DataFeature[index, :, :]
        self.testDataLabel = test_DataLabel[index]
        self.test_batches = self.testDataFeature.shape[0]//self.batchSize
        leng = self.testDataFeature.shape[0]

        self.valiData =  self.testDataFeature[int(leng*0.8):, :, :]
        self.valiLabel = self.testDataLabel[int(leng*0.8):]
        self.vali_batches = self.valiData.shape[0]//self.batchSize
        # self.batchpoint = 0

        self.trainLoader = data(self.train_batches, self.trainDataFeature, self.trainDataLabel, self.batchSize)
        self.testLoader  = data(self.test_batches, self.testDataFeature, self.testDataLabel, self.batchSize)
        self.valiLoader = data(self.vali_batches, self.valiData, self.valiLabel, self.batchSize)


    def get_testBatch(self):
        return self.testDataFeature, self.testDataLabel




if __name__ == '__main__':
    d = dataLoader(60,13)
    z, v= d.testLoader.nextBatch()
    co = 1






