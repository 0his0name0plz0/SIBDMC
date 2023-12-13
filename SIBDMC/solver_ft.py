import os

import paddle
from paddle import optimizer
import numpy as np

from dataset import return_data
from model import MIDSCNetLoss
import matplotlib.pyplot as plt
from metric import thrC, post_proC, err_rate
from metric import normalized_mutual_info_score, f1_score, rand_index_score, adjusted_rand_score
import scipy.io as sio
import utils

class Solver_ft(object):
    def __init__(self, args):
        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = 'gpu' if paddle.device.is_compiled_with_cuda() else 'cpu'
        self.ft_epoch = args.ft_epoch
        self.global_iter = args.global_iter
        self.batch_size = args.batch_size

        self.result = []

        self.z_dim = args.z_dim
        self.y_dim = args.y_dim

        self.gamma = args.gamma
        self.alpha = args.alpha
        self.cmi = args.cmi
        self.rec = args.rec
        self.c2 = args.c2
        self.selfExp = args.selfExp

        self.lr = args.lr
        self.show = args.show_freq

        self.data_loader = return_data(args)

        self.mvdscnet = MIDSCNetLoss(args).to(self.device)
        # self.optimizerSf = optim.Adam(self.mvdscnet.parameters(), lr=self.lr)
        self.optimizerSf = optimizer.Adam(learning_rate=self.lr, parameters=self.mvdscnet.parameters())

        self.save_name = args.save_name
        self.ckpt_name = args.ckpt_name
        if args.train and self.save_name is not None:
            self.load_checkpoint(self.ckpt_name)
            print("Pretrained network weights are loaded successfully.")

    def finetune(self):
        for batch_index, data in enumerate(self.data_loader):
            v1, v2, target = data
            # v1 = v1.to(self.device)
            # v2 = v2.to(self.device)
            v1 = utils.cuda(v1, self.device == 'gpu')
            v2 = utils.cuda(v2, self.device == 'gpu')
            # target = target.to('cpu').numpy()
            target = target.cpu().numpy()
            K = len(np.unique(target))
            best_acc, best_epoch = -1, -1

            for epoch in range(self.ft_epoch):
                # 这里会调用mvdscnet的forward函数，直接输出loss   参考DeepInfoMax写法
                total_loss, loss_rec, loss_mkl, loss_cmi, loss_D, loss_MLP1, loss_MLP2, loss_c, loss_selfExp = self.mvdscnet(v1, v2)
                self.optimizerSf.clear_grad()
                total_loss.backward()
                self.optimizerSf.step()

                self.result.append(total_loss.item())

                alpha2 = max(0.4 - (K-1)/10 * 0.1, 0.1)
                dim_subspace = 3
                ro = 1
                if epoch % self.show == 0 or epoch == self.ft_epoch - 1:
                    C1 = self.mvdscnet.self_expression1.Coefficient.detach().cpu().numpy()
                    C2 = self.mvdscnet.self_expression2.Coefficient.detach().cpu().numpy()
                    C = (C1 + C2) / 2

                    print('***** Finetune Epoch [{}/{}], Batch [{}/{}] : Total-loss = {:.4f}, REC-Loss = {:.4f}, '
                          'mKL-Loss = {:.4f}, Cmi-Loss = {:.4f}, D-Loss = {:.4f}, MLP1-Loss = {:.4f}, MLP2-Loss = {:.4f},'
                          'loss_c = {:.4f}, loss_selfExp = {:.4f}'
                          .format(epoch, self.ft_epoch, batch_index, 500 / self.batch_size,
                                  total_loss.item(), loss_rec.item(), loss_mkl.item(), loss_cmi.item(), loss_D.item(),
                                  loss_MLP1.item(), loss_MLP2.item(),
                                  loss_c.item(), loss_selfExp.item()))

                    Coef = thrC(C, alpha2)
                    Coef1 = thrC(C1, alpha2)
                    Coef2 = thrC(C2, alpha2)


                    y_x, L = post_proC(Coef, K, dim_subspace, ro)
                    target = target.squeeze()
                    missrate_x, c_x = err_rate(target, y_x)
                    acc_x = 1 - missrate_x
                    nmi = normalized_mutual_info_score(target, y_x)
                    f_measure = f1_score(target, y_x)
                    ri = rand_index_score(target, y_x)
                    ar = adjusted_rand_score(target, y_x)
                    print("epoch-Coef: [%d]" % epoch, "nmi: %.4f" % nmi, "accuracy: %.4f" % acc_x, "F-measure: %.4f" % f_measure, "RI: %.4f" % ri, "AR: %.4f" % ar)

                    y_x1, L = post_proC(Coef1, K, dim_subspace, ro)
                    missrate_x1, c_x1 = err_rate(target, y_x1)
                    acc_x1 = 1 - missrate_x1
                    nmi1 = normalized_mutual_info_score(target, y_x1)
                    f_measure1 = f1_score(target, y_x1)
                    ri = rand_index_score(target, y_x1)
                    ar1 = adjusted_rand_score(target, y_x1)
                    print("epoch-Coef1: [%d]" % epoch, "nmi1: %.4f" % nmi1, "accuracy1: %.4f" % acc_x1, "F-measure1: %.4f" % f_measure1, "RI: %.4f" % ri, "AR1: %.4f" % ar1)

                    y_x2, L = post_proC(Coef2, K, dim_subspace, ro)
                    missrate_x2, c_x2 = err_rate(target, y_x2)
                    acc_x2 = 1 - missrate_x2
                    nmi2 = normalized_mutual_info_score(target, y_x2)
                    f_measure2 = f1_score(target, y_x2)
                    ri = rand_index_score(target, y_x2)
                    ar2 = adjusted_rand_score(target, y_x2)
                    print("epoch-Coef2: [%d]" % epoch, "nmi2: %.4f" % nmi2, "accuracy2: %.4f" % acc_x2, "F-measure: %.4f" % f_measure2, "RI: %.4f" % ri, "AR: %.4f" % ar2)

                    if acc_x > best_acc:
                        best_acc = acc_x
                        best_epoch = epoch
                        print("******* the current best acc:", best_acc, acc_x1, acc_x2, 'in epoch:', best_epoch)
                        if best_acc > 0.40:
                            # print("******* the current best acc:", best_acc, acc_x1, acc_x2, 'in epoch:', best_epoch)
                            print("******* the truth:", target)
                            print("******* the Clustering result:", c_x)  # c_x为聚类得到的label
                            print("******* the Clustering result1:", c_x1)
                            print("******* the Clustering result2:", c_x2)
                            # print("************* C11.shape:", C11.shape)
                        if best_acc > 0.40:
                            savemat_dir = './mat_result/'
                            if not os.path.exists(savemat_dir):
                                os.makedirs(savemat_dir, exist_ok=True)
                            sio.savemat('./mat_result/rgbd_coef_tsne.mat', dict([('coef', Coef), ('target', target)]))
                            sio.savemat('./mat_result/rgbd_coef1_tsne.mat', dict([('coef1', Coef1), ('target', target)]))
                            sio.savemat('./mat_result/rgbd_coef2_tsne.mat', dict([('coef2', Coef2), ('target', target)]))
                            print("******** Save rgbd_coef for tsne successfully. best_acc:", best_acc)
            #break


    def load_checkpoint(self, epoch):
        netE1_path = "checkpoints/{}/netE1_{}.pth".format(self.save_name, epoch)
        netE2_path = "checkpoints/{}/netE2_{}.pth".format(self.save_name, epoch)
        netG1_path = "checkpoints/{}/netG1_{}.pth".format(self.save_name, epoch)
        netG2_path = "checkpoints/{}/netG2_{}.pth".format(self.save_name, epoch)
        netD1_path = "checkpoints/{}/netD1_{}.pth".format(self.save_name, epoch)
        netCmi1_path = "checkpoints/{}/netCmi1_{}.pth".format(self.save_name, epoch)
        netCmi2_path = "checkpoints/{}/netCmi2_{}.pth".format(self.save_name, epoch)
        netZ2Y_1_path = "checkpoints/{}/netZ2Y_1_{}.pth".format(self.save_name, epoch)
        netY2Z_1_path = "checkpoints/{}/netY2Z_1_{}.pth".format(self.save_name, epoch)
        netZ2Y_2_path = "checkpoints/{}/netZ2Y_2_{}.pth".format(self.save_name, epoch)
        netY2Z_2_path = "checkpoints/{}/netY2Z_2_{}.pth".format(self.save_name, epoch)

        if os.path.isfile(netE1_path):
            self.mvdscnet.netE1.set_state_dict(paddle.load(netE1_path))
            self.mvdscnet.netE2.set_state_dict(paddle.load(netE2_path))
            self.mvdscnet.netG1.set_state_dict(paddle.load(netG1_path))
            self.mvdscnet.netG2.set_state_dict(paddle.load(netG2_path))
            self.mvdscnet.netD1.set_state_dict(paddle.load(netD1_path))
            self.mvdscnet.netCmi1.set_state_dict(paddle.load(netCmi1_path))
            self.mvdscnet.netCmi2.set_state_dict(paddle.load(netCmi2_path))
            self.mvdscnet.netZ2Y_1.set_state_dict(paddle.load(netZ2Y_1_path))
            self.mvdscnet.netY2Z_1.set_state_dict(paddle.load(netY2Z_1_path))
            self.mvdscnet.netZ2Y_2.set_state_dict(paddle.load(netZ2Y_2_path))
            self.mvdscnet.netY2Z_2.set_state_dict(paddle.load(netY2Z_2_path))
            print("=> loaded checkpoint '{} (iter {})'".format(netE1_path, self.global_iter))
        else:
            print("=> no checkpoint found at '{}'".format(netE1_path))
