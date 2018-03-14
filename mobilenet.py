import keras


if __name__ == '__main__':
    mobile_net_model = keras.applications.mobilenet.MobileNet(weights='imagenet')
    mobile_net_model.summary()

