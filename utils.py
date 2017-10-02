import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as misc


def iterate_minibatches(inputs, targets, small_targets, batchsize, shuffle=False):

    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt], small_targets[excerpt], class_weights(targets[excerpt]), \
              class_weights(small_targets[excerpt])


def iterate_membatches(inputs, targets, batchsize, dataset_loader, shuffle=False):

    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield dataset_loader(inputs[excerpt], targets[excerpt], inputs[excerpt].shape[0])


def class_weights(y):
    total_pos = np.count_nonzero(y)
    c = 1.02
    class_wgts = np.zeros((y.shape[0], y.shape[1], y.shape[2], y.shape[3])).astype(np.float32)

    for n in range(0, y.shape[1]):
        class_wgts[:, n, :, :] = 1 / np.log(c + np.count_nonzero(y[:, n, :, :]) / float(total_pos)).astype(np.float32)

    return class_wgts


def deprocess_image(image, scale=False):  
    im = np.zeros((360, 480, image.shape[0])).astype(np.float32)
    for chan in range(0, image.shape[0]):
        im[:, :, chan] = image[chan, :, :]
        
    if scale:
        im = misc.imresize(im, (45, 60, 3), interp='nearest')
    return im 


# Segmentation to segmentation comparison
# Encoder decoder training cycle
# Image to segmentation comparison
# Debug
def deprocess_segmentation(segmentation):
    
    colors = [(128, 128, 128), 	
              (128, 0, 0), (192, 192, 128),	
              (256, 69, 0), (128, 64, 128),	
              (60, 40, 222), (128, 128, 0),
              (192, 128, 128), (64, 64, 128),	
              (64, 0, 128), (64, 64, 0), (0, 128, 192)]
    seg = np.argmax(segmentation, axis=0)
    color_seg = np.zeros((segmentation.shape[1], segmentation.shape[2],
                          3)).astype(np.float32)
    for row in range(0, color_seg.shape[0]):
        for col in range(0, color_seg.shape[1]):
            color_seg[row, col, :] = colors[seg[row, col]]
    
    return color_seg / 255


def intersection_over_union(preds, gt):
    assert(preds.shape[0] == gt.shape[0])
    pds = np.squeeze(np.argmax(preds, axis=1))
    IU = 0.0
    for cls in range(0, preds.shape[1]):
        gt_cls = gt[:, cls, :, :]
        tp = np.count_nonzero(gt_cls[pds==cls])
        fp = np.count_nonzero(pds==cls) - tp
        fn = np.count_nonzero(gt[:, cls, :, :]) - tp
        IU += float(tp) / (tp + fp + fn)
    return float(IU) / (preds.shape[1])
        

# Shows graph of training statistics
# First is always generator error, then discriminator error, then a third error
def show_training_stats_graph(err, val_err, num_epochs, filename,
                              second_error_title):

    # Make X labels
    labels = []
    for n in range(0, num_epochs):
        labels.append(str(n))

    # Plot the 3 different subplots
    x = np.arange(0, num_epochs)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    ax1.set_title("Training Stats for ENET")
    ax1.plot(x, err)
    ax1.set_ylabel('Generator Error')

    ax2.plot(x, val_err)
    ax2.set_ylabel(second_error_title)

    ax3.set_xlabel('Epochs completed')

    fig.savefig(filename)
    plt.close(fig)


def show_examples(images, segmentations, targets, num_examples, epoch, filename):

    nc = 3
    h = segmentations.shape[2]
    w = segmentations.shape[3]
    
    if h == 45:
        scale=True
    else:
        scale=False

    image = np.zeros((h * num_examples, 3 * w, nc)).astype(np.float32)

    example_count = 0
    for example in range(0, num_examples):

        image[h * example : h* example + h, 0 : w, :] = deprocess_image(images[example_count, :, :, :], scale)
        
        select_im = deprocess_segmentation(segmentations[example_count, :, :, :])
        
        image[h * example : h * example + h , w : 2 * w, :] = deprocess_segmentation(targets[example_count, :, :, :])
        image[h * example : h * example + h ,2 * w : 3 * w, :] = select_im
        
        example_count += 1

    fig, ax = plt.subplots()

    ax.set_xticks([])
    ax.set_yticks([])

    if epoch is not None:
        ax.set_title("Example Segmentations: Epoch " + str(epoch))
    else:
        ax.set_title("Example Segmentations")

    plt.imshow(image)

    fig.savefig(filename, bbox_inches='tight', dpi=450)
    plt.close('all')
    


