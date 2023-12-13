import os
import paddle
import paddle.optimizer as optimizer
import paddle.nn.functional as F

from utils import cuda
from model import Encoder, Decoder, Discriminator, MLP, MIEstimator, LocalMIEstimator
from dataset import return_data

class Solver(object):
    def __init__(self, args):
        self.use_cuda = args.cuda and paddle.device.is_compiled_with_cuda()

        self.max_iter = args.max_iter
        self.global_iter = args.global_iter
        self.save_checkMin = args.save_checkMin

        self.result = []

        self.y_dim = args.y_dim
        self.z_dim = args.z_dim

        self.alpha = args.alpha
        self.mkl = args.mkl
        self.cmi = args.cmi
        self.gamma = args.gamma
        self.rec = args.rec

        self.lr = args.lr

        self.batch_size = args.batch_size

        self.data_loader = return_data(args)

        self.netE1 = cuda(Encoder(z_dim=self.z_dim, y_dim=self.y_dim, nc=3), self.use_cuda)  # RGB图  z-->z12  y-->z11
        self.netE2 = cuda(Encoder(z_dim=self.z_dim, y_dim=self.y_dim, nc=1), self.use_cuda)  # Depth图
        self.netG1 = cuda(Decoder(z_dim=self.z_dim, y_dim=self.y_dim, nc=3), self.use_cuda)
        self.netG2 = cuda(Decoder(z_dim=self.z_dim, y_dim=self.y_dim, nc=1), self.use_cuda)
        self.netD1 = cuda(MIEstimator(z_dim=self.z_dim), self.use_cuda)
        self.netCmi1 = cuda(LocalMIEstimator(), self.use_cuda)
        self.netCmi2 = cuda(LocalMIEstimator(), self.use_cuda)

        self.netZ2Y_1 = cuda(MLP(s_dim=self.z_dim, t_dim=self.y_dim), self.use_cuda)
        self.netY2Z_1 = cuda(MLP(s_dim=self.y_dim, t_dim=self.z_dim), self.use_cuda)
        self.netZ2Y_2 = cuda(MLP(s_dim=self.z_dim, t_dim=self.y_dim), self.use_cuda)
        self.netY2Z_2 = cuda(MLP(s_dim=self.y_dim, t_dim=self.z_dim), self.use_cuda)

        self.optimizerG1 = optimizer.Adam(
            learning_rate=self.lr,
            parameters=[
                {'params': self.netE1.parameters()}, {'params': self.netG1.parameters()},
                {'params': self.netE2.parameters()}, {'params': self.netG2.parameters()},
                {'params': self.netZ2Y_1.parameters()}, {'params': self.netY2Z_1.parameters()},
                {'params': self.netZ2Y_2.parameters()}, {'params': self.netY2Z_2.parameters()},
                {'params': self.netD1.parameters()},
                {'params': self.netCmi1.parameters()}, {'params': self.netCmi2.parameters()}
            ]
        )
        self.save_name = args.save_name
        self.ckpt_dir = os.path.join('./checkpoints/', args.save_name)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)

        self.ckpt_name = args.ckpt_name
        self.save_step = args.save_step

    def train(self):
        self.net_mode(train=True)
        out = False
        best_past_loss = 100000
        counter = 0

        while not out:
            for batch_index, data in enumerate(self.data_loader):
                self.global_iter += 1

                v1, v2, target = data
                v1 = cuda(v1, self.use_cuda)
                v2 = cuda(v2, self.use_cuda)

                # self.optimizerG1.zero_grad()
                self.optimizerG1.clear_grad()

                z12, z12_m, z11, mu1, logvar1 = self.netE1(v1)
                z22, z22_m, z21, mu2, logvar2 = self.netE2(v2)

                rec_z11 = self.netZ2Y_1(z12)
                rec_z12 = self.netY2Z_1(z11)
                rec_z21 = self.netZ2Y_2(z22)
                rec_z22 = self.netY2Z_2(z21)

                rec_v1 = self.netG1(z12, z11)
                rec_v2 = self.netG2(z22, z21)

                mi_JS1, mi_estimation1 = self.netD1(z22.detach(), z12)
                loss_D1 = mi_JS1.sum() / self.batch_size
                mi_JS2, mi_estimation2 = self.netD1(z12.detach(), z22)
                loss_D2 = mi_JS2.sum() / self.batch_size
                loss_D = - (loss_D1 + loss_D2) / 2.

                kl1 = 0.5 * paddle.sum(paddle.exp(logvar1) + paddle.pow(mu1, 2) - 1 - logvar1).divide(paddle.to_tensor(self.batch_size,dtype=paddle.float32))
                kl2 = 0.5 * paddle.sum(paddle.exp(logvar2) + paddle.pow(mu2, 2) - 1 - logvar2).divide(paddle.to_tensor(self.batch_size,dtype=paddle.float32))
                loss_mkl = (kl1 + kl2) / 2.

                loss_cmi1 = self.netCmi1(z22_m, z12)
                loss_cmi2 = self.netCmi2(z12_m, z22)
                loss_cmi = (loss_cmi1 + loss_cmi2) / 2.

                loss_MLP1 = (F.mse_loss(rec_z11, z11.detach(), reduction='sum') / self.batch_size
                             + F.mse_loss(rec_z12, z12.detach(), reduction='sum') / self.batch_size)
                loss_MLP2 = (F.mse_loss(rec_z21, z21.detach(), reduction='sum') / self.batch_size
                             + F.mse_loss(rec_z22, z22.detach(), reduction='sum') / self.batch_size)
                loss_MLP_tmp = (loss_MLP1 + loss_MLP2) / 2.

                loss_MLP = paddle.exp(-loss_MLP_tmp)

                loss_total_rec1 = F.mse_loss(rec_v1, v1, reduction='sum') / self.batch_size
                loss_total_rec2 = F.mse_loss(rec_v2, v2, reduction='sum') / self.batch_size
                loss_rec = (loss_total_rec1 + loss_total_rec2) / 2.

                loss_total = self.rec * loss_rec + self.mkl * loss_mkl + self.cmi * loss_cmi + self.alpha * loss_D + self.gamma * loss_MLP

                loss_total.backward()
                self.optimizerG1.step()

                if self.global_iter == 1 or self.global_iter % 5 == 0:
                    # global_iter是全局迭代次数
                    print('[Iter-{}/{}], Batch [{}/{}] : loss_MLP1: {:.5f}, loss_MLP2: {:.5f}, loss_KL1: {:.5f}, loss_KL2: {:.5f}, loss_D1: {:.5f}, loss_D2: {:.5f}, loss_cmi1: {:.5f}, loss_cmi2: {:.5f}, loss_rec1: {:.5f}, loss_rec2: {:.5f}'
                          .format(self.global_iter, self.max_iter, batch_index, 500 / self.batch_size, loss_MLP1.item(), loss_MLP2.item(),
                                  kl1.item(), kl2.item(), loss_D1.item(), loss_D2.item(), loss_cmi1.item(), loss_cmi2.item(), loss_total_rec1.item(), loss_total_rec2.item()))  # 这里应该把每项loss都打出来看看
                    print('[Iter-{}]: loss_total: {:.5f}, loss_MLP: {:.5f}, loss_D: {:.5f}, loss_mkl: {:.5f}, loss_cmi: {:.5f}, loss_rec: {:.5f}'
                        .format(self.global_iter, loss_total.item(), loss_MLP_tmp.item(), loss_D.item(), loss_mkl.item(), loss_cmi.item(), loss_rec.item()))
                    self.result.append(loss_total.item())

                if self.global_iter % self.save_checkMin == 0:
                    cur_loss = loss_total.item()
                    if cur_loss < best_past_loss:
                        best_past_loss = cur_loss
                        self.save_checkpoint('best_min')
                        counter = 0

                        if self.global_iter > 300:
                            print('Saved current minimum loss in checkpoint(iter:{})'.format(self.global_iter))
                            print('[Iter-{}]: loss_total: {:.5f}, loss_MLP: {:.5f}, loss_D: {:.5f}, loss_mkl: {:.5f}, loss_cmi: {:.5f}, loss_rec: {:.5f}'
                                .format(self.global_iter, loss_total.item(), loss_MLP_tmp.item(), loss_D.item(), loss_mkl.item(), loss_cmi.item(), loss_rec.item()))
                    else:  # early stopping
                        counter += 1
                        if counter >= 2600:
                            print(f'======= EarlyStopping counter: {counter} out of 2600, at epoch: {self.global_iter} ========')
                            out = True
                            break

                if self.global_iter >= self.max_iter:  # 这里是 epoch*batch个数 的迭代循环  end
                    out = True
                    break

    def test(self):
        self.net_mode(train=False)

    def net_mode(self, train):
        if not isinstance(train, bool):
            raise('Only bool type is supported. True or False')

        if train:
            self.netE1.train()
            self.netE2.train()
            self.netG1.train()
            self.netG2.train()
            self.netD1.train()
            self.netCmi1.train()
            self.netCmi2.train()
            self.netZ2Y_1.train()
            self.netY2Z_1.train()
            self.netZ2Y_2.train()
            self.netY2Z_2.train()
        else:
            self.netE1.eval()
            self.netE2.eval()
            self.netG1.eval()
            self.netG2.eval()
            self.netD1.eval()
            self.netCmi1.eval()
            self.netCmi2.eval()
            self.netZ2Y_1.eval()
            self.netY2Z_1.eval()
            self.netZ2Y_2.eval()
            self.netY2Z_2.eval()

    def save_checkpoint(self, epoch):
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

        if not os.path.exists("checkpoints"):
            os.mkdir("checkpoints")

        if not os.path.exists("checkpoints/{}".format(self.save_name)):
            os.mkdir("checkpoints/{}".format(self.save_name))
        paddle.save(self.netE1.state_dict(), netE1_path)
        paddle.save(self.netE2.state_dict(), netE2_path)
        paddle.save(self.netG1.state_dict(), netG1_path)
        paddle.save(self.netG2.state_dict(), netG2_path)
        paddle.save(self.netD1.state_dict(), netD1_path)
        paddle.save(self.netCmi1.state_dict(), netCmi1_path)
        paddle.save(self.netCmi2.state_dict(), netCmi2_path)
        paddle.save(self.netZ2Y_1.state_dict(), netZ2Y_1_path)
        paddle.save(self.netY2Z_1.state_dict(), netY2Z_1_path)
        paddle.save(self.netZ2Y_2.state_dict(), netZ2Y_2_path)
        paddle.save(self.netY2Z_2.state_dict(), netY2Z_2_path)
    
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
            self.netE1.set_state_dict(paddle.load(netE1_path))
            self.netE2.set_state_dict(paddle.load(netE2_path))
            self.netG1.set_state_dict(paddle.load(netG1_path))
            self.netG2.set_state_dict(paddle.load(netG2_path))
            self.netD1.set_state_dict(paddle.load(netD1_path))
            self.netCmi1.set_state_dict(paddle.load(netCmi1_path))
            self.netCmi2.set_state_dict(paddle.load(netCmi2_path))
            self.netZ2Y_1.set_state_dict(paddle.load(netZ2Y_1_path))
            self.netY2Z_1.set_state_dict(paddle.load(netY2Z_1_path))
            self.netZ2Y_2.set_state_dict(paddle.load(netZ2Y_2_path))
            self.netY2Z_2.set_state_dict(paddle.load(netY2Z_2_path))
            # 从最后一次last 处load checkpoint，然后从第 global_iter 次开始训练
            print("=> loaded checkpoint '{} (iter {})'".format(netE1_path, self.global_iter))
        else:
            print("=> no checkpoint found at '{}'".format(netE1_path))
