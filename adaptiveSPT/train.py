import torch
import numpy as np
import argparse
import time
import os
import util
from engine import trainer
from cmpassDataloader import dataLoader
# from cmpassDataloader2 import dataLoader
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--fd_number',type=str,default='1',help='subset id')
parser.add_argument('--load',type=str,default=r'E:\zhangyuxuan\code\SPTRUL\garage\fd01_y1\epoch_100_60.6.pth',help='load path')
parser.add_argument('--save',type=str,default='./tt/fd01_y1/',help='save path')
parser.add_argument('--gcn_bool',action='store_true',default = True, help='whether to add graph convolution layer')
parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
parser.add_argument('--addaptadj',action='store_true', default = True , help='whether add adaptive adj')
parser.add_argument('--randomadj',action='store_true', help='whether random initialize adaptive adj')
parser.add_argument('--seq_length',type=int,default=12,help='')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=18,help='number of nodes')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=150,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
#parser.add_argument('--seed',type=int,default=99,help='random seed')


parser.add_argument('--expid',type=int,default=1,help='experiment id')

args = parser.parse_args()

def main():
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    #load data
    device = torch.device(args.device)
    # sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata,args.adjtype)
    # supports = [torch.tensor(i).to(device) for i in adj_mx]
    print(args)
    adjinit = None
    supports = None
    engine = trainer(args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                         adjinit)
    print("start training...", flush=True)
    train_his_loss = []
    his_loss =[]
    val_time = []
    train_time = []
    data = dataLoader( bs = args.batch_size, sl = args.seq_length, fd_number =args.fd_number)

    for i in range(1,args.epochs+1):
        #if i % 10 == 0:
            #lr = max(0.000002,args.learning_rate * (0.1 ** (i // 10)))
            #for g in engine.optimizer.param_groups:
                #g['lr'] = lr
        train_loss = []
        train_asf= []
        train_rmse = []

        t1 = time.time()
        for iter in  range(data.train_batches):
            trainx, trainy  = data.trainLoader.nextBatch()
            trainx = trainx[:, :, :, np.newaxis]
            trainx = torch.Tensor(trainx).to(device)
            trainx = trainx.permute((0, 3, 2, 1))
            trainy = torch.Tensor(trainy).to(device)
            metrics = engine.train(trainx, trainy)
            # mc = engine.eval(trainx,trainy)
            train_loss.append(metrics[0])
            train_asf.append(metrics[2])
            train_rmse.append(metrics[1])
            if iter % args.print_every == 0 :
                log = 'Iter: {:03d}, Train Loss: {:.4f},Train RMSE: {:.4f},Train asf: {:.4f}'
                print(log.format(iter, train_loss[-1],train_rmse[-1],train_asf[-1]),flush=True)

        t2 = time.time()
        train_time.append(t2-t1)
        #validation
        valid_loss = []
        valid_asf = []
        valid_rmse = []

        s1 = time.time()
        for iter in range(data.test_batches):
            testx, testy  = data.testLoader.nextBatch()
            testx = testx[:, :, :, np.newaxis]
            testx = torch.Tensor(testx).to(device)
            testx = testx.permute((0, 3, 2, 1))
            testy = torch.Tensor(testy).to(device)
            metrics = engine.eval(testx, testy)
            valid_loss.append(metrics[0])
            valid_rmse.append(metrics[1])
            valid_asf.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i,(s2-s1)))
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_rmse = np.mean(train_rmse)
        mtrain_asf = np.mean(train_asf)

        mvalid_loss = np.mean(valid_loss)
        mvalid_rmse = np.mean(valid_rmse)
        mvalid_asf = np.mean(valid_asf)
        his_loss.append(mvalid_loss)
        train_his_loss.append(train_loss)
        log = 'Epoch: {:03d}, Train Loss: {:.4f},Train RMSE: {:.4f}, Train_asf:{:.4f}, Valid Loss: {:.4f},Valid RMSE: {:.4f}, Valid_asf:{:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_rmse, mtrain_asf, mvalid_loss, mvalid_rmse, mvalid_asf, (t2 - t1)),flush=True)
        torch.save(engine.model.state_dict(), args.save+"epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+".pth")
        for k in range(5):
            test_sample(k+2, engine, data, device, args.save+"epoch_"+str(i)+'_'+str(k))
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))
    np.save(args.save + "valiloss.npy", his_loss)
    np.save(args.save + "trainloss.npy", train_his_loss)


    #testing

def test_sample(sample_num, engine, data, device, path):
    test_feature, test_label = data.get_one_piece(sample_num)
    test_feature_b = np.reshape(test_feature, [len(test_feature), args.seq_length, 18])
    test_feature_b = test_feature_b[:, :, :, np.newaxis]
    test_feature_b = torch.Tensor(test_feature_b).to(device)
    test_feature_b = test_feature_b.permute((0, 3, 2, 1))
    result = engine.eval2(test_feature_b)
    leng = len(result)
    plt.figure()
    plt.ylim([40, 150])

    plt.plot(range(leng), result, marker='|', color='coral', linewidth=1.0, linestyle='--', label='Prediction')
    plt.plot(range(leng), test_label, linestyle='-', label='Label ')
    plt.ylabel("RUL(Cycle)")
    plt.xlabel("Time(Cycle)")
    plt.title(r"RUL Prediction Sample")
    plt.legend()
    plt.savefig(path)
    # plt.show()



if __name__ == "__main__":
    # test_sample(2)
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
