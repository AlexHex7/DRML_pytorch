import matplotlib.pyplot as plt

with_region_log = 'logs/region_layer.log'
without_region_log = 'logs/without_region_layer.log'

log_every_epoch = 10


def extract_test_log(log_file):
    with open(log_file) as fp:
        line_list = fp.readlines()[3:]

    epoch_list = []
    loss_list = []
    mean_f1_list = []
    au_f1_dict = dict()

    count = 0
    for line in line_list:
        if '[TRAIN]' in line:
            continue

        line = line.strip()
        if len(line) == 0:
            continue

        count += 1
        if count % log_every_epoch != 0:
            continue

        info_list = line.split(' ')[5:]
        epoch, loss, mean_f1, au_f1 = info_list

        epoch = int(epoch.split('/')[0].split('[')[1])
        loss = float(loss.split(':')[1])
        mean_f1 = float(mean_f1.split(':')[1])

        au_list = au_f1.replace('[', '').replace(']', '').split(',')
        for au_id, au_score in enumerate(au_list):
            au_score = float(au_score)
            if au_f1_dict.get(au_id) is None:
                au_f1_dict[au_id] = [au_score]
            else:
                au_f1_dict[au_id].append(au_score)

        epoch_list.append(epoch)
        loss_list.append(loss)
        mean_f1_list.append(mean_f1)

    return epoch_list, loss_list, mean_f1_list, au_f1_dict


def extract_train_log(log_file):
    with open(log_file) as fp:
        line_list = fp.readlines()[3:]

    epoch_list = []
    loss_list = []

    count = 0
    for line in line_list:
        if '[TRAIN]' not in line:
            continue

        line = line.strip()
        if len(line) == 0:
            continue

        count += 1
        if count % log_every_epoch != 0:
            continue

        epoch, loss = line.split(' ')[5:7]
        epoch = int(epoch.split('/')[0].split('[')[1])
        loss = float(loss.split(':')[1])

        epoch_list.append(epoch)
        loss_list.append(loss)

    return epoch_list, loss_list


# +++++++++++++++++++++++++++ TEST ++++++++++++++++++++++++++++++++++++++++
epoch_list_1, loss_list_1, mean_f1_list_1, au_f1_dict_1 \
    = extract_test_log(with_region_log)

epoch_list_2, loss_list_2, mean_f1_list_2, au_f1_dict_2 \
    = extract_test_log(without_region_log)
# ========================== Test Loss ======================================
plt.title('Test Loss')
plt.xlabel('Epoch')
plt.ylabel('test loss')

plt.plot(epoch_list_1, loss_list_1, label='With Region Layer')
plt.plot(epoch_list_2, loss_list_2, label='Without Region Layer')

plt.legend()
plt.savefig('logs/test_loss.png')
plt.close()

# ========================== Mean F1 Score ======================================
plt.title('Mean F1 Score')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')

plt.plot(epoch_list_1, mean_f1_list_1, label='With Region Layer(max %.4f)' % max(mean_f1_list_1))
plt.plot(epoch_list_2, mean_f1_list_2, label='Without Region Layer(max %.4f)' % max(mean_f1_list_2))

plt.legend()
plt.savefig('logs/mean_f1_score.png')
plt.close()

# # ========================== AU F1 Score ======================================
# plt.title('AU F1 Score With Region')
# plt.xlabel('Epoch')
# plt.ylabel('AU Score')
#
# for key, value_list in au_f1_dict_1.items():
#
#     plt.plot(epoch_list_1, value_list, label='AU_ID:%d(max %.4f)' % (key, max(value_list)))
#
# plt.legend(shadow=True, fontsize='x-small')
# plt.savefig('logs/au_f1_score.png')
# plt.close()

# ++++++++++++++++++++++++++++++ TRAIN +++++++++++++++++++++++++++++++++++++
epoch_list_1, loss_list_1 = extract_train_log(with_region_log)
epoch_list_2, loss_list_2 = extract_train_log(without_region_log)

epoch_list_1 = epoch_list_1[:75]
loss_list_1 = loss_list_1[:75]
epoch_list_2 = epoch_list_2[:75]
loss_list_2 = loss_list_2[:75]

plt.title('Train Loss')
plt.xlabel('Epoch')
plt.ylabel('train loss')

plt.plot(epoch_list_1, loss_list_1, label='With Region Layer(min %.4f)' % min(loss_list_1))
plt.plot(epoch_list_2, loss_list_2, label='Without Region Layer(min %.4f)' % min(loss_list_2))
plt.legend()
plt.savefig('logs/train_loss.png')
plt.close()


