import numpy as np
from paddle.vision import transforms
from paddle.io import DataLoader
import paddle
import scipy.io as sio

def Normalize(data):
    """
    :param data:Input data
    :return:normalized data
    """
    m = np.mean(data)
    mx = np.max(data)
    mn = np.min(data)
    return ((data - m) / (mx - mn))


class CustomDataset(paddle.io.Dataset):
    def __init__(self, transform=None):

        super().__init__()
        self.views = []
        self.view_shape = []
        self.transform = transform

        data = sio.loadmat('./data_base/rgbd_mtv.mat')
        features = data['X']
        self.label = data['gt']

        for v in features[0]:
            self.view_shape.append(v.shape[1])
            self.views.append(v)

        # self.target = torch.from_numpy(self.label)
        self.target = paddle.to_tensor(self.label)
        view1 = np.transpose(self.views[0], (0, 3, 1, 2))
        view2 = np.transpose(self.views[1], (0, 3, 1, 2))
        # self.view1 = torch.from_numpy(view1)
        self.view1 = paddle.to_tensor(view1)
        # self.view2 = torch.from_numpy(view2)
        self.view2 = paddle.to_tensor(view2)

    def __getitem__(self, index):
        return self.view1[index], self.view2[index], self.target[index]

    def __len__(self):
        return self.views[0].shape[0]  # 500


def return_data(args):
    name = args.dataset
    batch_size = args.batch_size
    num_workers = args.num_workers

    if name.lower() == 'rgbd':
        transform = transforms.Compose([transforms.ToTensor()])
        print("*********** loading rgbd dataset ************")
        train_data = CustomDataset(transform=transform)

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=True
                              )

    data_loader = train_loader
    return data_loader
