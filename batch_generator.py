import os
import datetime


if __name__ == '__main__':
    batch_size = [40]
    merge_layer = ['dot', 'multiply', 'l1', 'l2']
    epochs = [5]
    vgg_frozen_size = [19]
    batch_norm = [True]
    dropout = [0.1]
    learning_rate = [0.0001]
    decay = [0.0]
    optimizer = ['adam']
    block_to_remove = [1, 2]
    fine_tuning_iter = 0

    i = 0
    with open("cmd.txt", 'w') as f:
        for bs in batch_size:
            for ml in merge_layer:
                for e in epochs:
                    for vgg in vgg_frozen_size:
                        for bn in batch_norm:
                            for d in dropout:
                                for lr in learning_rate:
                                    for lrd in decay:
                                        for opt in optimizer:
                                            for btm in block_to_remove:
                                                cmd = "-b " + str(bs) + " " \
                                                      "-bn " + str(bn) + " " \
                                                      "-m " + str(ml) + " " \
                                                      "-vl " + str(vgg) + " " \
                                                      "-e " + str(e) + " " \
                                                      "-d " + str(d) + " " \
                                                      "-lr " + str(lr) + " " \
                                                      "-lrd " + str(lrd) + " " \
                                                      "-op " + str(opt) + " " \
                                                      "-vrb " + str(btm) + " " \
                                                      "-f " + str(fine_tuning_iter) + " " \
                                                      "-o runs_results/" + str(datetime.date.today()) + \
                                                                           "_" + str(i) + "\n"
                                                f.write(cmd)
                                                i += 1
    print(i)

