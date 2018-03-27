def main_test(Model):

    print('\nmodel testing start!!\n')
    model = Model

    if GPU >= 0:
        chainer.cuda.get_device_from_id(GPU).use()
        model.to_gpu()

    serializers.load_npz(LOAD_MODEL, model)
    print('Load from {}'.format(LOAD_MODEL))

    test, filepaths = load_images(TEST_PATH, shuffle=False)

    print('Test images: {}\n'.format(len(test)))
    #image_visualize(test)

    count = 0
    print('{:^20s} : gt / predict (probability)'.format('image-name'))
    print('-----------------------------------------------------------------------')

    for index in range(len(test)):
        x, t = test[index]
        #plt.imshow((x.reshape(INPUT_WIDTH, INPUT_HEIGHT, 3))); plt.show()

        x = x[None, ...]

        #x = chainer.cuda.to_gpu(x, 0)
        y = model(x)
        y = y.data
        y = chainer.cuda.to_cpu(y)
        y = F.softmax(y).data

        pred_label = y.argmax(axis=1)

        print(' {:20s}: {} / {} ({:.3f})'.format(filepaths[index].split('/')[-1], t, pred_label[0], y.max()))

        if t == pred_label:
            count += 1

    print('-----------------------------------------------------------------------')
    print('accuracy: {} ({}/{})'.format(1. * count / len(test), count, len(test)))

    print('\nmodel testing finished!!\n')