import torch
from torch.autograd import Variable
from lib.network import Network
from lib.data_loader import DataSet
import config as cfg
import logging


def adjust_learning_rate(optimizer, decay_rate=.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

logging.basicConfig(level=logging.INFO,
                    format='(%(asctime)s %(levelname)s) %(message)s',
                    datefmt='%d %b %H:%M:%S',
                    filename='logs/region_layer.log',
                    filemode='w')

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('(%(levelname)s) %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

net = Network(cfg.class_number)
if torch.cuda.is_available():
    net.cuda(cfg.cuda_num)

opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)

dataset = DataSet(cfg)
train_sample_nb = len(dataset.train_dataset)
test_sample_nb = len(dataset.test_dataset)
train_batch_nb = len(dataset.train_loader)
test_batch_nb = len(dataset.test_loader)

logging.info('Train batch[%d] sample[%d]' % (train_batch_nb, train_sample_nb))
logging.info('Test batch[%d] sample[%d]\n' % (test_batch_nb, test_sample_nb))

for epoch_index in range(cfg.epoch):
    if (epoch_index + 1) % cfg.lr_decay_every_epoch == 0:
        adjust_learning_rate(opt, decay_rate=cfg.lr_decay_rate)

    for batch_index, (img, label) in enumerate(dataset.train_loader):
        img = Variable(img)
        label = Variable(label)

        if torch.cuda.is_available():
            img = img.cuda(cfg.cuda_num)
            label = label.cuda(cfg.cuda_num)

        pred = net(img)
        loss = net.multi_label_sigmoid_cross_entropy_loss(pred, label)
        opt.zero_grad()
        loss.backward()
        opt.step()

        statistics_list = net.statistics(pred.data, label.data, cfg.thresh)
        mean_f1_score, f1_score_list = net.calc_f1_score(statistics_list)

        f1_score_list = ['%.4f' % f1_score for f1_score in f1_score_list]

        # print('epoch[%d/%d] batch[%d/%d] loss:%.4f mean_f1_score:%.4f\n\t[%s]'
        #       % (epoch_index+1, cfg.epoch,
        #          batch_index+1, train_batch_nb,
        #          loss.data[0], mean_f1_score,
        #          ' '.join(f1_score_list)))
    logging.info('[TRAIN] epoch[%d/%d] loss:%.4f mean_f1_score:%.4f [%s]'
                 % (epoch_index+1, cfg.epoch, loss.data[0], mean_f1_score, ' '.join(f1_score_list)))

    if (epoch_index + 1) % cfg.test_every_epoch == 0:
        loss_total = 0
        total_statistics_list = []

        for batch_index, (img, label) in enumerate(dataset.test_loader):
            img = Variable(img, volatile=True)
            label = Variable(label, volatile=True)

            if torch.cuda.is_available():
                img = img.cuda(cfg.cuda_num)
                label = label.cuda(cfg.cuda_num)

            pred = net(img)

            loss = net.multi_label_sigmoid_cross_entropy_loss(pred, label, size_average=False)
            loss_total += loss

            new_statistics_list = net.statistics(pred.data, label.data, cfg.thresh)
            total_statistics_list = net.update_statistics_list(total_statistics_list, new_statistics_list)

        loss_mean = loss_total / test_sample_nb
        mean_f1_score, f1_score_list = net.calc_f1_score(total_statistics_list)
        f1_score_list = ['%.4f' % f1_score for f1_score in f1_score_list]

        logging.info('[TEST] epoch[%d/%d] loss:%.4f mean_f1_score:%.4f [%s]'
                     % (epoch_index+1, cfg.epoch, loss_mean.data[0], mean_f1_score, ','.join(f1_score_list)))


