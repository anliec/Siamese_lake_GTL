import os
import datetime


if __name__ == '__main__':
    batch_size = [40]
    merge_layer = ['concatenate', 'dot', 'subtract', 'multiply']
    epochs = [10]
    vgg_frozen_size = [19, 15]
    batch_norm = [True, False]

    i = 0
    with open("cmd.txt", 'w') as f:
        for bs in batch_size:
            for ml in merge_layer:
                for e in epochs:
                    for vgg in vgg_frozen_size:
                        for bn in batch_norm:
                            cmd = "-b " + str(bs) + " " \
                                  "-bn " + str(bn) + " " \
                                  "-m " + str(ml) + " " \
                                  "-vl " + str(vgg) + " " \
                                  "-e " + str(e) + " " \
                                  "-o ~/runs_results/" + str(datetime.date.today()) + "_" + str(i) + "\n"
                            f.write(cmd)
                            i += 1
    print(i)

