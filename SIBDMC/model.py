import math

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn.functional import softplus
# import torch.nn.init as init

import paddle
import paddle.nn as nn
from paddle.nn.functional import softplus
from paddle.nn.functional import pad
from paddle.nn.functional import mse_loss


def reparametrize(mu, logvar):
    std = logvar.divide(paddle.to_tensor(2,dtype=paddle.float32)).exp()
    eps = paddle.randn(shape=std.shape)
    return mu + std*eps


class View(paddle.nn.Layer):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return paddle.reshape(tensor, shape=self.size)


class Discriminator(paddle.nn.Layer):
    def __init__(self, z_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim * 2, z_dim),
            # nn.LeakyReLU(0.2, True),
            nn.LeakyReLU(0.2),
            nn.Linear(z_dim, z_dim),
            # nn.LeakyReLU(0.2, True),
            nn.LeakyReLU(0.2),
            nn.Linear(z_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, y):
        return self.net(y).squeeze()


class MIEstimator(paddle.nn.Layer):
    def __init__(self, z_dim):
        super(MIEstimator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(z_dim * 2, z_dim),
            # nn.LeakyReLU(0.2, True),
            nn.LeakyReLU(0.2),
            nn.Linear(z_dim, z_dim),
            # nn.LeakyReLU(0.2, True),
            nn.LeakyReLU(0.2),
            nn.Linear(z_dim, 1),
        )

    # Gradient for JSD mutual information estimation
    def forward(self, z1, z2):
        # pos = self.net(torch.cat([z1, z2], 1))
        pos = self.net(paddle.concat([z1, z2], 1))
        # neg = self.net(torch.cat([torch.roll(z1, 1, 0), z2], 1))
        neg = self.net(paddle.concat([paddle.roll(z1, 1, 0), z2], 1))
        return -softplus(-pos).sum() - softplus(neg).sum(), pos.sum() - neg.exp().sum() + 1


class LocalMIEstimator(paddle.nn.Layer):
    def __init__(self):
        super(LocalMIEstimator, self).__init__()

        self.local_mi = nn.Sequential(  #  [bs, 64, 16, 16]
            # nn.Conv2d(64, 128, kernel_size=1),  # [bs, 128, 16, 16]
            nn.Conv2D(64, 128, kernel_size=1),
            # nn.ReLU(True),
            nn.ReLU(),
            # nn.Conv2d(128, 128, kernel_size=1),  # [bs, 128, 16, 16]
            nn.Conv2D(128, 128, kernel_size=1),
            # nn.ReLU(True),
            nn.ReLU(),
            # nn.Conv2d(128, 1, kernel_size=1),  # [bs, 1, 16, 16]
            nn.Conv2D(128, 1, kernel_size=1)
        )

    # Gradient for JSD mutual information estimation and EB-based estimation
    def forward(self, z_m, z):
        ## z_map: [bs, 32, 16, 16]  z:[bs, 16*8*8]
        z_exp = paddle.tile(z.reshape((-1, 16, 8, 8)), repeat_times=[1, 2, 2, 2])
        z_m_prime = paddle.concat((z_m[1:], z_m[0].unsqueeze(0)), axis=0)

        z_cat = paddle.concat((z_m, z_exp), axis=1)
        z_cat_prime = paddle.concat((z_m_prime, z_exp), axis=1)

        Ej = -softplus(-self.local_mi(z_cat)).mean()
        Em = softplus(self.local_mi(z_cat_prime)).mean()
        local_mi = -1 * (Ej - Em)   # -1 * Jensen-Shannon MI estimator
        return local_mi


class MLP(paddle.nn.Layer):
    def __init__(self, s_dim, t_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, s_dim),
            # nn.LeakyReLU(0.2, True),
            nn.LeakyReLU(0.2),
            nn.Linear(s_dim, t_dim),
            # nn.LeakyReLU(0.2, True),
            nn.LeakyReLU(0.2),
            nn.Linear(t_dim, t_dim),
            nn.ReLU()
        )

    def forward(self, s):
        t = self.net(s)
        return t


class Conv2dSamePad(paddle.nn.Layer):
    def __init__(self, kernel_size, stride):
        super(Conv2dSamePad, self).__init__()
        self.kernel_size = kernel_size if type(kernel_size) in [list, tuple] else [kernel_size, kernel_size]
        self.stride = stride if type(stride) in [list, tuple] else [stride, stride]

    def forward(self, x):  # x: [bs,3,64,64]
        in_height = x.shape[2]
        in_width = x.shape[3]
        out_height = math.ceil(float(in_height) / float(self.stride[0]))  # math.ceil向上取整 >=x
        out_width = math.ceil(float(in_width) / float(self.stride[1]))
        pad_along_height = ((out_height - 1) * self.stride[0] + self.kernel_size[0] - in_height)
        pad_along_width = ((out_width - 1) * self.stride[1] + self.kernel_size[1] - in_width)
        pad_top = math.floor(pad_along_height / 2)
        pad_left = math.floor(pad_along_width / 2)
        pad_bottom = pad_along_height - pad_top
        pad_right = pad_along_width - pad_left
        # return F.pad(x, [pad_left, pad_right, pad_top, pad_bottom], 'constant', 0)
        return pad(x, pad=[pad_top, pad_bottom, pad_left, pad_right], mode='constant', value=0)


class ConvTranspose2dSamePad(paddle.nn.Layer):
    def __init__(self, kernel_size, stride):
        super(ConvTranspose2dSamePad, self).__init__()
        self.kernel_size = kernel_size if type(kernel_size) in [list, tuple] else [kernel_size, kernel_size]
        self.stride = stride if type(stride) in [list, tuple] else [stride, stride]

    def forward(self, x):
        # in_height = x.size(2)
        in_height = x.shape[2]
        # in_width = x.size(3)
        in_width = x.shape[3]
        pad_height = self.kernel_size[0] - self.stride[0]
        pad_width = self.kernel_size[1] - self.stride[1]
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        return x[:, :, pad_top:in_height - pad_bottom, pad_left: in_width - pad_right]


class Encoder(paddle.nn.Layer):
    def __init__(self, z_dim, y_dim, nc):
        super(Encoder, self).__init__()

        self.nc = nc  # input_channels
        self.z_dim = z_dim
        self.y_dim = y_dim

        sequence = [Conv2dSamePad(3, 2),
                    # nn.Conv2d(nc, 64, 3, 2),
                    nn.Conv2D(nc, 64, 3, 2),
                    # nn.ReLU(True),
                    nn.ReLU(),
                    Conv2dSamePad(3, 2),
                    # nn.Conv2d(64, 32, 3, 2),
                    nn.Conv2D(64, 32, 3, 2),
                    # nn.ReLU(True)
                    nn.ReLU()
                    ]

        sequence_z = [Conv2dSamePad(3, 2),
                      # nn.Conv2d(32, 16, 3, 2),
                      nn.Conv2D(32, 16, 3, 2),
                      # nn.ReLU(True),
                      nn.ReLU(),
                      View((-1, 16 * 8 * 8)),
                      nn.Linear(16*8*8, z_dim * 2)]

        sequence_y = [Conv2dSamePad(3, 2),
                      # nn.Conv2d(32, 16, 3, 2),
                      nn.Conv2D(32, 16, 3, 2),
                      # nn.ReLU(True),
                      nn.ReLU(),
                      View((-1, 16 * 8 * 8)),
                      nn.Linear(16*8*8, y_dim)]

        self.encoder_com = nn.Sequential(*sequence)
        self.encoder_z = nn.Sequential(*sequence_z)
        self.encoder_y = nn.Sequential(*sequence_y)

    def forward(self, x):
        z_map = self.encoder_com(x)
        distributions = self.encoder_z(z_map)

        mu = distributions[:, : self.z_dim]
        logvar = distributions[:, self.z_dim : ]
        z = reparametrize(mu, logvar)

        y = self.encoder_y(z_map)
        return z, z_map, y, mu, logvar


class Decoder(paddle.nn.Layer):
    def __init__(self, z_dim, y_dim, nc):
        super(Decoder, self).__init__()

        self.nc = nc  # input_channel
        self.z_dim = z_dim
        self.y_dim = y_dim

        self.decoder_y = nn.Sequential(nn.Linear(y_dim, 16*8*8),
                View((-1, 16, 8, 8)),
                nn.Conv2DTranspose(16, 32, 3, 2),
                ConvTranspose2dSamePad(3, 2),
                nn.ReLU())

        self.decoder_z = nn.Sequential(nn.Linear(z_dim, 16 * 8 * 8),
                View((-1, 16, 8, 8)),
                nn.Conv2DTranspose(16, 32, 3, 2),
                ConvTranspose2dSamePad(3, 2),
                nn.ReLU())

        self.decoder = nn.Sequential(
                nn.Conv2DTranspose(32, 64, 3, 2),
                ConvTranspose2dSamePad(3, 2),
                nn.ReLU(True),
                nn.Conv2DTranspose(64, nc, 3, 2),
                ConvTranspose2dSamePad(3, 2),
                nn.ReLU())

    def forward(self, z, y):
        z_ = self.decoder_z(z)
        y_ = self.decoder_y(y)
        rec = 0.5 * y_ + 0.5 * z_
        x_rec = self.decoder(rec)
        return x_rec


class SelfExpression(paddle.nn.Layer):
    def __init__(self, n):
        super(SelfExpression, self).__init__()
        self.Coefficient = paddle.create_parameter(shape=[n, n],
                                    dtype=paddle.float32,
                                    default_initializer=paddle.nn.initializer.Constant(1.0e-8))
        self.Coefficient.stop_gradient = False

    def forward(self, x):
        with paddle.no_grad():
            self.Coefficient.set_value(paddle.subtract(self.Coefficient, paddle.diag(paddle.diag(self.Coefficient))))
        x_ = paddle.matmul(self.Coefficient, x)
        return x_


class MIDSCNetLoss(paddle.nn.Layer):

    def __init__(self, args):
        super(MIDSCNetLoss, self).__init__()
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.batch_size = args.batch_size
        y_dim = args.y_dim
        z_dim = args.z_dim

        self.netE1 = Encoder(z_dim, y_dim, nc=3)
        self.netE2 = Encoder(z_dim, y_dim, nc=1)
        self.netG1 = Decoder(z_dim, y_dim, nc=3)
        self.netG2 = Decoder(z_dim, y_dim, nc=1)
        self.netD1 = MIEstimator(z_dim)
        self.netCmi1 = LocalMIEstimator()
        self.netCmi2 = LocalMIEstimator()

        self.netZ2Y_1 = MLP(z_dim, y_dim)
        self.netY2Z_1 = MLP(y_dim, z_dim)
        self.netZ2Y_2 = MLP(z_dim, y_dim)
        self.netY2Z_2 = MLP(y_dim, z_dim)

        self.self_expression1 = SelfExpression(self.batch_size)
        self.self_expression2 = SelfExpression(self.batch_size)

        self.gamma = args.gamma
        self.alpha = args.alpha
        self.mkl = args.mkl
        self.cmi = args.cmi
        self.rec = args.rec
        self.c2 = args.c2
        self.selfExp = args.selfExp

    def forward(self, v1, v2):
        z12, z12_m, z11, mu1, logvar1 = self.netE1(v1)
        z22, z22_m, z21, mu2, logvar2 = self.netE2(v2)

        z12_self = self.self_expression1(z12)
        z22_self = self.self_expression2(z22)

        kl1 = 0.5 * paddle.sum(paddle.exp(logvar1) + paddle.pow(mu1, 2) - 1 - logvar1)  # VAE的KL散度项

        kl2 = 0.5 * paddle.sum(paddle.exp(logvar2) + paddle.pow(mu2, 2) - 1 - logvar2)

        loss_mkl = (kl1 + kl2).sum().divide(paddle.to_tensor(self.batch_size,dtype=paddle.float32)) / 2.

        loss_cmi1 = self.netCmi1(z22_m, z12)
        loss_cmi2 = self.netCmi2(z12_m, z22)
        loss_cmi = (loss_cmi1 + loss_cmi2) / 2.

        rec_z11 = self.netZ2Y_1(z12)
        rec_z12 = self.netY2Z_1(z11)
        rec_z21 = self.netZ2Y_2(z22)
        rec_z22 = self.netY2Z_2(z21)

        loss_MLP1 = (mse_loss(rec_z11, z11.detach(), reduction='sum') / self.batch_size + mse_loss(rec_z12, z12.detach(), reduction='sum').divide(paddle.to_tensor(self.batch_size,dtype=paddle.float32)) )
        loss_MLP2 = (mse_loss(rec_z21, z21.detach(), reduction='sum') / self.batch_size + mse_loss(rec_z22, z22.detach(), reduction='sum').divide(paddle.to_tensor(self.batch_size,dtype=paddle.float32)) )

        loss_MLP = paddle.exp(-(loss_MLP1 + loss_MLP2) / 2.)

        loss_c1 = paddle.sum(paddle.pow(self.self_expression1.Coefficient, 2))  # Coef1的范数

        loss_selfExp1 = mse_loss(z12_self, z12, reduction='sum').divide(paddle.to_tensor(self.batch_size,dtype=paddle.float32))   # |z-zc1|

        loss_c2 = paddle.sum(paddle.pow(self.self_expression2.Coefficient, 2))  # Coef2的范数

        loss_selfExp2 = mse_loss(z22_self, z22, reduction='sum').divide(paddle.to_tensor(self.batch_size,dtype=paddle.float32))

        loss_c = (loss_c1 + loss_c2) / 2
        loss_selfExp = (loss_selfExp1 + loss_selfExp2) / 2

        rec_v1 = self.netG1(z12_self, z11)
        rec_v2 = self.netG2(z22_self, z21)

        loss_total_rec1 = mse_loss(rec_v1, v1, reduction='sum').divide(paddle.to_tensor(self.batch_size,dtype=paddle.float32))
        loss_total_rec2 = mse_loss(rec_v2, v2, reduction='sum').divide(paddle.to_tensor(self.batch_size,dtype=paddle.float32))
        loss_rec = (loss_total_rec1 + loss_total_rec2) / 2.

        mi_JS1, mi_estimation1 = self.netD1(z22.detach(), z12)
        loss_D1 = mi_JS1.sum() / self.batch_size
        mi_JS2, mi_estimation2 = self.netD1(z12.detach(), z22)
        loss_D2 = mi_JS2.sum() / self.batch_size
        loss_D = - (loss_D1 + loss_D2) / 2.

        loss_total = self.rec * loss_rec + self.mkl * loss_mkl + self.cmi * loss_cmi + self.alpha * loss_D + self.gamma * loss_MLP + self.c2 * loss_c + self.selfExp * loss_selfExp

        return loss_total, loss_rec, loss_mkl, loss_cmi, loss_D, loss_MLP1, loss_MLP2, loss_c, loss_selfExp
