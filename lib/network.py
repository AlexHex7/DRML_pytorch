import torch
from torch import nn
from lib.region_layer import RegionLayer
from lib.replace_region_layer import ReplaceRegionLayer


class Network(nn.Module):
    def __init__(self, class_number=12):
        super(Network, self).__init__()

        self.class_number = class_number

        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=11, stride=1),
            RegionLayer(in_channels=32, grid=(8, 8)),
            # ReplaceRegionLayer(in_channels=32,),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=32),

            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=8, stride=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=8,),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=6, stride=2),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=16*27*27, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(in_features=4096, out_features=2048),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(in_features=2048, out_features=class_number)
        )

    def forward(self, x):
        """

        :param x:   (b, c, h, w)
        :return:    (b, class_number)
        """

        batch_size = x.size(0)

        output = self.extractor(x)

        output = output.view(batch_size, -1)
        output = self.classifier(output)
        return output

    @staticmethod
    def multi_label_sigmoid_cross_entropy_loss(pred, y, size_average=True):
        """

        :param pred: (b, class)
        :param y: (b, class)
        :return:
        """

        batch_size = pred.size(0)
        pred = nn.Sigmoid()(pred)

        # try:
        # pos_part = (y > 0).float() * torch.log(pred)
        pos_to_log = pred[y > 0]
        pos_to_log[pos_to_log.data == 0] = 1e-20
        pos_part = torch.log(pos_to_log).sum()

        # neg_part = (y < 0).float() * torch.log(1 - pred)
        neg_to_log = 1 - pred[y < 0]
        neg_to_log[neg_to_log.data == 0] = 1e-20
        neg_part = torch.log(neg_to_log).sum()
        # except Exception:
        #     # print(pred[y > 0].min())
        #     # print((1 - pred[y < 0]).min())
        #     pdb.set_trace()

        loss = -(pos_part + neg_part)

        if size_average:
            loss /= batch_size

        return loss

    @staticmethod
    def statistics(pred, y, thresh):
        batch_size = pred.size(0)
        class_nb = pred.size(1)

        pred = pred > thresh
        pred = pred.long()
        pred[pred == 0] = -1

        statistics_list = []
        for j in range(class_nb):
            TP = 0
            FP = 0
            FN = 0
            TN = 0
            for i in range(batch_size):
                if pred[i][j] == 1:
                    if y[i][j] == 1:
                        TP += 1
                    elif y[i][j] == -1:
                        FP += 1
                    else:
                        assert False
                elif pred[i][j] == -1:
                    if y[i][j] == 1:
                        FN += 1
                    elif y[i][j] == -1:
                        TN += 1
                    else:
                        assert False
                else:
                    assert False
            statistics_list.append({'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN})
        return statistics_list

    @staticmethod
    def calc_f1_score(statistics_list):
        f1_score_list = []

        for i in range(len(statistics_list)):
            TP = statistics_list[i]['TP']
            FP = statistics_list[i]['FP']
            FN = statistics_list[i]['FN']

            precise = TP / (TP + FP + 1e-20)
            recall = TP / (TP + FN + 1e-20)
            f1_score = 2 * precise * recall / (precise + recall + 1e-20)
            f1_score_list.append(f1_score)
        mean_f1_score = sum(f1_score_list) / len(f1_score_list)

        return mean_f1_score, f1_score_list

    @staticmethod
    def update_statistics_list(old_list, new_list):
        if not old_list:
            return new_list

        assert len(old_list) == len(new_list)

        for i in range(len(old_list)):
            old_list[i]['TP'] += new_list[i]['TP']
            old_list[i]['FP'] += new_list[i]['FP']
            old_list[i]['TN'] += new_list[i]['TN']
            old_list[i]['FN'] += new_list[i]['FN']

        return old_list

if __name__ == '__main__':
    from torch import nn
    from torch.autograd import Variable
    import numpy as np

    image = Variable(torch.randn(2, 3, 170, 170))
    label = Variable(torch.from_numpy(np.random.randint(3, size=[2, 12]) - 1))

    net = Network()
    opt = torch.optim.Adam(net.parameters(), lr=0.001)

    while True:
        pred = net(image)

        loss = net.multi_label_sigmoid_cross_entropy_loss(pred, label)
        print(loss.data[0])
        print('\n')
        opt.zero_grad()
        loss.backward()
        opt.step()


