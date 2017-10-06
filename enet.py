import os
import theano
import theano.tensor as T
import lasagne
import numpy as np
import time
from utils import *
import imageio


class Enet:
    def __init__(self, dataset, folder_name='enet', **kwargs):

        # set hyperparameters
        self.start_lr = kwargs.get('lr', 5e-4)
        self.lr = self.start_lr
        self.num_epochs = kwargs.get('num_epochs', 80)
        self.num_epochs_ed = kwargs.get('num_epochs_ed', 250)
        self.bz = kwargs.get('bz', 5)
        self.num_examples = kwargs.get('num_examples', 5)
        self.seed = kwargs.get('seed', 115)

        # set variables from dataset
        self.dataset = dataset
        self.X_files_train = dataset.X_files_train
        self.y_files_train = dataset.y_files_train
        self.X_files_val = dataset.X_files_val
        self.y_files_val = dataset.y_files_val
        self.X_files_test = dataset.X_files_test
        self.y_files_test = dataset.y_files_test
        self.h = dataset.h
        self.w = dataset.w
        self.nc = dataset.nc
        self.images_in_mem = dataset.images_in_mem

        # for video functionality
        self.video_loader = dataset.load_video_files
        self.X_files_video = dataset.X_files_video

        self.base = os.getcwd() + '/' + folder_name + '/'
        if not os.path.isdir(self.base):
            os.mkdir(self.base)
            os.mkdir(self.base + 'models/')
            os.mkdir(self.base + 'images/')
            os.mkdir(self.base + 'stats/')
            os.mkdir(self.base + 'video/')

    def bottleneck(self, encoder, prev_name, name, num_filters, out_num_filters, filter_size, use_relu, asymetric,
                   dilated, downsample, drop_amt):

        internal = out_num_filters / 4

        if downsample:
            stride = 2
        else:
            stride = 1

        identity = encoder[prev_name]

        layer_name = name + '_conv1'
        encoder[layer_name] = lasagne.layers.Conv2DLayer(encoder[prev_name], num_filters=internal,
                                                   filter_size=stride, stride=stride, pad='valid', b=None,
                                                   nonlinearity=None)

        prev_name = layer_name
        layer_name = name + '_bn1'
        encoder[layer_name] = lasagne.layers.BatchNormLayer(encoder[prev_name], epsilon=1E-3)

        if use_relu:
            prev_name = layer_name
            layer_name = name + '_prelu1'
            encoder[layer_name] = lasagne.layers.prelu(encoder[prev_name])

        prev_name = layer_name
        if not asymetric and dilated is None:
            layer_name = name + '_conv2'
            encoder[layer_name] = lasagne.layers.Conv2DLayer(encoder[prev_name], num_filters=internal,
                                                             filter_size=filter_size, stride=1, pad='same',
                                                             nonlinearity=None)
        elif asymetric:
            layer_name = name + '_conv2_asymetric'
            encoder[layer_name] = lasagne.layers.Conv2DLayer(encoder[prev_name], num_filters=internal,
                                                             filter_size=(filter_size, 1), stride=1, pad='same', b=None,
                                                             nonlinearity=None)
            prev_name = layer_name
            layer_name = name + '_conv2_asymetric2'
            encoder[layer_name] = lasagne.layers.Conv2DLayer(encoder[prev_name], num_filters=internal,
                                                             filter_size=(1, filter_size), stride=1, pad='same',
                                                             nonlinearity=None)
        elif dilated is not None:
            layer_name = name + '_conv2_pad'
            if dilated % 2 == 0:
                pad_amt = dilated
            else:
                pad_amt = (dilated - 1) / 2
            encoder[layer_name] = lasagne.layers.PadLayer(encoder[prev_name], width=pad_amt)
            prev_name = layer_name
            layer_name = name + '_conv2_dilated'
            encoder[layer_name] = lasagne.layers.DilatedConv2DLayer(encoder[prev_name], num_filters=internal,
                                                                    filter_size=3, dilation=(dilated, dilated))
            print(lasagne.layers.get_output_shape(encoder[layer_name]))

        # Batch norm and prelu
        prev_name = layer_name
        layer_name = name + '_bn2'
        encoder[layer_name] = lasagne.layers.BatchNormLayer(encoder[prev_name], epsilon=1E-3)

        if use_relu:
            prev_name = layer_name
            layer_name = name + '_prelu2'
            encoder[layer_name] = lasagne.layers.prelu(encoder[prev_name])

        # Second 1 x 1 convolution
        prev_name = layer_name
        layer_name = name + '_conv3'
        encoder[layer_name] = lasagne.layers.Conv2DLayer(encoder[prev_name], num_filters=out_num_filters,
                                                         filter_size=1, stride=1, b=None,
                                                         nonlinearity=None)

        prev_name = layer_name
        layer_name = name + '_bn3'
        encoder[layer_name] = lasagne.layers.BatchNormLayer(encoder[prev_name], epsilon=1E-3)

        prev_name = layer_name
        layer_name = name + '_dropout'
        encoder[layer_name] = lasagne.layers.DropoutLayer(encoder[prev_name], p=drop_amt)

        # Adds identity function to second branch
        layer_name = name + '_identity'
        encoder[layer_name] = lasagne.layers.NonlinearityLayer(identity, lasagne.nonlinearities.identity)

        if downsample:
            prev_name = layer_name
            layer_name = name + '_maxpool'
            encoder[layer_name] = lasagne.layers.MaxPool2DLayer(encoder[prev_name], pool_size=2, stride=2)

            prev_name = layer_name
            layer_name = name + '_pad2'

            pad_amt = (out_num_filters - num_filters) // 2

            encoder[layer_name] = lasagne.layers.PadLayer(encoder[prev_name], (0, pad_amt, 0, 0), batch_ndim=0)

        # Concatenation and prelu
        prev_name = layer_name
        layer_name = name + '_sum'
        encoder[layer_name] = lasagne.layers.ElemwiseSumLayer([encoder[prev_name], encoder[name + '_dropout']])

        prev_name = layer_name
        layer_name = name
        encoder[layer_name] = lasagne.layers.prelu(encoder[prev_name])
        return encoder[name]

    def bottleneck_decoder(self, decoder, prev_layer, name, encoder_name, num_filters, out_num_filters,
                           filter_size, upsample):

        internal = out_num_filters / 4

        if upsample:
            stride = 2
        else:
            stride = 1

        identity = prev_layer

        layer_name = name + '_conv1'
        decoder[layer_name] = lasagne.layers.Conv2DLayer(prev_layer, num_filters=internal,
                                                         filter_size=1, stride=1, b=None,
                                                         nonlinearity=None)
        prev_name = layer_name
        layer_name = name + '_bn1'
        decoder[layer_name] = lasagne.layers.BatchNormLayer(decoder[prev_name], epsilon=1E-3)


        prev_name = layer_name
        layer_name = name + '_relu1'
        decoder[layer_name] = lasagne.layers.NonlinearityLayer(decoder[prev_name],
                                                               nonlinearity=lasagne.nonlinearities.rectify)

        prev_name = layer_name
        layer_name = name + '_conv2'
        if stride == 1:
            decoder[layer_name] = lasagne.layers.Conv2DLayer(decoder[prev_name], num_filters=internal,
                                                             filter_size=3, stride=stride, pad='same',
                                                             nonlinearity=None)
        else:

            decoder[layer_name] = lasagne.layers.TransposedConv2DLayer(decoder[prev_name], num_filters=internal,
                                                             filter_size=3, stride=stride, crop='same',
                                                             nonlinearity=None)

            # Original implementation adds row to output image
            prev_name = layer_name
            layer_name = '_pad0'
            decoder[layer_name] = lasagne.layers.PadLayer(decoder[prev_name], width=((0, 1), (0, 1)))

        # Batch norm and prelu
        prev_name = layer_name
        layer_name = name + '_bn2'
        decoder[layer_name] = lasagne.layers.BatchNormLayer(decoder[prev_name], epsilon=1E-3)

        prev_name = layer_name
        layer_name = name + '_relu2'
        decoder[layer_name] = lasagne.layers.NonlinearityLayer(decoder[prev_name],
                                                               nonlinearity=lasagne.nonlinearities.rectify)

        # Second 1 x 1 convolution
        prev_name = layer_name
        layer_name = name + '_conv3'
        decoder[layer_name] = lasagne.layers.Conv2DLayer(decoder[prev_name], num_filters=out_num_filters,
                                                         filter_size=1, stride=1, b=None,
                                                         nonlinearity=None)

        prev_name = layer_name
        layer_name = name + '_bn3'
        decoder[layer_name] = lasagne.layers.BatchNormLayer(decoder[prev_name], epsilon=1E-3)

        # Adds identity function to second branch
        layer_name = name + '_identity'
        decoder[layer_name] = lasagne.layers.NonlinearityLayer(identity, lasagne.nonlinearities.identity)

        if upsample:
            
            prev_name = layer_name
            layer_name = name + '_maxunpool_conv'
            decoder[layer_name] = lasagne.layers.Conv2DLayer(decoder[prev_name], num_filters=out_num_filters,
                                                         filter_size=1, stride=1, b=None,
                                                         nonlinearity=None)     
            
            prev_name = layer_name
            layer_name = name + '_maxunpool_bn'
            decoder[layer_name] = lasagne.layers.BatchNormLayer(decoder[prev_name], epsilon=1E-3)
            
            prev_name = layer_name
            layer_name = name + '_maxunpool'
            decoder[layer_name] = lasagne.layers.InverseLayer(decoder[prev_name], decoder[encoder_name])



        # Concatenation and prelu
        prev_name = layer_name
        layer_name = name + '_sum'
        decoder[layer_name] = lasagne.layers.ElemwiseSumLayer([decoder[name + '_bn3'], decoder[prev_name]])

        prev_name = layer_name
        layer_name = name
        decoder[layer_name] = lasagne.layers.NonlinearityLayer(decoder[prev_name],
                                                               nonlinearity=lasagne.nonlinearities.rectify)

        return decoder[name]

    def build_encoder(self):

        input_im = T.tensor4()
        encoder = {}
        details = [['Layer Name', 'Dims in', 'Dims out']]

        name = 'in'
        input_shape = (None, 3, 360, 480)
        encoder[name] = lasagne.layers.InputLayer(shape=input_shape, input_var=input_im)
        output_dims = input_shape

        prev_name = name
        name = 'input_conv'
        encoder[name] = lasagne.layers.Conv2DLayer(encoder[prev_name], num_filters=13,
                                                   filter_size=3, stride=2, pad='same',
                                                   nonlinearity=None)
        name = 'input_pool'
        encoder[name] = lasagne.layers.MaxPool2DLayer(encoder[prev_name], pool_size=2, stride=2)
        name = 'concat'
        encoder[name] = lasagne.layers.ConcatLayer([encoder['input_conv'], encoder['input_pool']],
                                                   axis=1)
        prev_output_dims = output_dims
        output_dims = lasagne.layers.get_output_shape(encoder[name])
        details.append([name, str(prev_output_dims),
                        str(output_dims)])

        prev_name = name
        name = 'bottleneck1.0'
        encoder[name] = self.bottleneck(encoder, prev_name, name, 16, 64, 3, use_relu=True, asymetric=False,
                                        dilated=None, downsample=True, drop_amt=0.01)
        prev_output_dims = output_dims
        output_dims = lasagne.layers.get_output_shape(encoder[name])
        details.append([name, str(prev_output_dims),
                        str(output_dims)])

        for n in range(1, 5):
            prev_name = name
            name = 'bottleneck1.' + str(n)
            encoder[name] = self.bottleneck(encoder, prev_name, name, 64, 64, 3, use_relu=True, asymetric=False,
                                            dilated=None, downsample=False, drop_amt=0.01)
            prev_output_dims = output_dims
            output_dims = lasagne.layers.get_output_shape(encoder[name])
            details.append([name, str(prev_output_dims),
                            str(output_dims)])

        prev_name = name
        name = 'bottleneck2.0'
        encoder[name] = self.bottleneck(encoder, prev_name, name, 64, 128, 3, use_relu=True, asymetric=False,
                                        dilated=None, downsample=True, drop_amt=0.1)
        prev_output_dims = output_dims
        output_dims = lasagne.layers.get_output_shape(encoder[name])
        details.append([name, str(prev_output_dims),
                        str(output_dims)])

        for n in range(2, 4):

            prev_name = name
            name = 'bottleneck' + str(n) + '.1'
            encoder[name] = self.bottleneck(encoder, prev_name, name, 128, 128, 3, use_relu=True, asymetric=False,
                                            dilated=None, downsample=False, drop_amt=0.1)
            prev_output_dims = output_dims
            output_dims = lasagne.layers.get_output_shape(encoder[name])
            details.append([name, str(prev_output_dims),
                            str(output_dims)])

            prev_name = name
            name = 'bottleneck' + str(n) + '.2'
            encoder[name] = self.bottleneck(encoder, prev_name, name, 128, 128, 3, use_relu=True, asymetric=False,
                                            dilated=2, downsample=False, drop_amt=0.1)
            prev_output_dims = output_dims
            output_dims = lasagne.layers.get_output_shape(encoder[name])
            details.append([name, str(prev_output_dims),
                            str(output_dims)])

            prev_name = name
            name = 'bottleneck' + str(n) + '.3'
            encoder[name] = self.bottleneck(encoder, prev_name, name, 128, 128, 5, use_relu=True, asymetric=True,
                                            dilated=None, downsample=False, drop_amt=0.1)
            prev_output_dims = output_dims
            output_dims = lasagne.layers.get_output_shape(encoder[name])
            details.append([name, str(prev_output_dims),
                            str(output_dims)])

            prev_name = name
            name = 'bottleneck' + str(n) + '.4'
            encoder[name] = self.bottleneck(encoder, prev_name, name, 128, 128, 3, use_relu=True, asymetric=False,
                                            dilated=4, downsample=False, drop_amt=0.1)
            prev_output_dims = output_dims
            output_dims = lasagne.layers.get_output_shape(encoder[name])
            details.append([name, str(prev_output_dims),
                            str(output_dims)])

            prev_name = name
            name = 'bottleneck' + str(n) + '.5'
            encoder[name] = self.bottleneck(encoder, prev_name, name, 128, 128, 3, use_relu=True, asymetric=False,
                                            dilated=None, downsample=False, drop_amt=0.1)
            prev_output_dims = output_dims
            output_dims = lasagne.layers.get_output_shape(encoder[name])
            details.append([name, str(prev_output_dims),
                            str(output_dims)])

            prev_name = name
            name = 'bottleneck' + str(n) + '.6'
            encoder[name] = self.bottleneck(encoder, prev_name, name, 128, 128, 3, use_relu=True, asymetric=False,
                                            dilated=8, downsample=False, drop_amt=0.1)
            prev_output_dims = output_dims
            output_dims = lasagne.layers.get_output_shape(encoder[name])
            details.append([name, str(prev_output_dims),
                            str(output_dims)])

            prev_name = name
            name = 'bottleneck' + str(n) + '.7'
            encoder[name] = self.bottleneck(encoder, prev_name, name, 128, 128, 3, use_relu=True, asymetric=True,
                                            dilated=None, downsample=False, drop_amt=0.1)
            prev_output_dims = output_dims
            output_dims = lasagne.layers.get_output_shape(encoder[name])
            details.append([name, str(prev_output_dims),
                            str(output_dims)])

            prev_name = name
            name = 'bottleneck' + str(n) + '.8'
            encoder[name] = self.bottleneck(encoder, prev_name, name, 128, 128, 3, use_relu=True, asymetric=False,
                                            dilated=16, downsample=False, drop_amt=0.1)
            prev_output_dims = output_dims
            output_dims = lasagne.layers.get_output_shape(encoder[name])
            details.append([name, str(prev_output_dims),
                            str(output_dims)])

        encoder['small_out'] = encoder['bottleneck3.8']

        prev_name = name
        name = 'classifier'
        encoder[name] = lasagne.layers.Conv2DLayer(encoder[prev_name], num_filters=self.dataset.lab_ln,
                                                   filter_size=1, stride=1,
                                                   nonlinearity=lasagne.nonlinearities.sigmoid)
        prev_output_dims = output_dims
        output_dims = lasagne.layers.get_output_shape(encoder[name])
        details.append([name, str(prev_output_dims),
                        str(output_dims)])

        try:
            from tabulate import tabulate
            print(tabulate(details))
        except ImportError:
            pass

        return encoder, input_im

    def build_decoder(self, encoder):

        decoder = encoder
        details = [['Layer Name', 'Dims in', 'Dims out']]

        name = 'bottleneck4.0'
        decoder[name] = self.bottleneck_decoder(decoder, decoder['small_out'], name, 'bottleneck2.0_maxpool',
                                                128, 64, 3, True)
        prev_output_dims = lasagne.layers.get_output_shape(decoder['small_out'])
        output_dims = lasagne.layers.get_output_shape(decoder[name])
        details.append([name, str(prev_output_dims),
                        str(output_dims)])

        prev_name = name
        name = 'bottleneck4.1'
        decoder[name] = self.bottleneck_decoder(decoder, decoder[prev_name], name, None,
                                                64, 64, 3, False)
        prev_output_dims = output_dims
        output_dims = lasagne.layers.get_output_shape(decoder[name])
        details.append([name, str(prev_output_dims),
                        str(output_dims)])

        prev_name = name
        name = 'bottleneck4.2'
        decoder[name] = self.bottleneck_decoder(decoder, decoder[prev_name], name, None,
                                                64, 64, 3, False)
        prev_output_dims = output_dims
        output_dims = lasagne.layers.get_output_shape(decoder[name])
        details.append([name, str(prev_output_dims),
                        str(output_dims)])

        prev_name = name
        name = 'bottleneck5.0'
        decoder[name] = self.bottleneck_decoder(decoder, decoder[prev_name], name, 'bottleneck1.0_maxpool',
                                                64, 16, 3, True)
        prev_output_dims = output_dims
        output_dims = lasagne.layers.get_output_shape(decoder[name])
        details.append([name, str(prev_output_dims),
                        str(output_dims)])

        prev_name = name
        name = 'bottleneck5.1'
        decoder[name] = self.bottleneck_decoder(decoder, decoder[prev_name], name, None,
                                                16, 16, 3, False)
        prev_output_dims = output_dims
        output_dims = lasagne.layers.get_output_shape(decoder[name])
        details.append([name, str(prev_output_dims),
                        str(output_dims)])

        prev_name = name
        name = 'out'
        decoder[name] = lasagne.layers.TransposedConv2DLayer(decoder[prev_name], num_filters=12,
                                                             filter_size=2, stride=(2, 2),
                                                             nonlinearity=lasagne.nonlinearities.sigmoid)
        prev_output_dims = output_dims
        output_dims = lasagne.layers.get_output_shape(decoder[name])
        details.append([name, str(prev_output_dims),
                        str(output_dims)])

        try:
            from tabulate import tabulate
            print(tabulate(details))
        except ImportError:
            pass

        return decoder #, input_var

    def build_train_fns(self):
        # defines variables
        print("Building model and compiling functions...")
        small_targets = T.tensor4()
        targets = T.tensor4()
        weights = T.tensor4()


        # Builds discriminator and generator
        encoder, inputs = self.build_encoder()
        decoder = self.build_decoder(encoder)
        print(lasagne.layers.count_params(decoder['out']))

        # Gets outputs
        e_train_out = lasagne.layers.get_output(encoder['classifier'], inputs)
        ed_train_out = lasagne.layers.get_output(decoder['out'], inputs)

        e_val_out = lasagne.layers.get_output(encoder['classifier'], deterministic=True)
        ed_val_out = lasagne.layers.get_output(decoder['out'], inputs, deterministic=True)

        # Create loss expressions
        train_e_loss = lasagne.objectives.binary_crossentropy(e_train_out, small_targets) * weights
        train_e_loss = train_e_loss.mean()
        train_ed_loss = lasagne.objectives.binary_crossentropy(ed_train_out, targets) * weights
        train_ed_loss = train_ed_loss.mean()

        val_e_loss = lasagne.objectives.binary_crossentropy(e_val_out, small_targets) * weights
        val_e_loss = val_e_loss.mean()
        val_ed_loss = lasagne.objectives.binary_crossentropy(ed_val_out, targets) * weights
        val_ed_loss = val_ed_loss.mean()

        # Updates the paramters
        encoder_params = lasagne.layers.get_all_params(encoder['classifier'], trainable=True)
        ed_params = lasagne.layers.get_all_params(decoder['out'], trainable=True)
        lr = T.fscalar('lr')
        encoder_updates = lasagne.updates.adam(
            train_e_loss, encoder_params, learning_rate=lr, beta1=0.5)
        ed_updates = lasagne.updates.adam(train_ed_loss, ed_params, learning_rate=lr, beta1=0.5)

        # Compiles functions
        e_train_fn = theano.function([inputs, small_targets, weights, lr], [train_e_loss], updates=encoder_updates)
        ed_train_fn = theano.function([inputs, targets, weights, lr], [train_ed_loss], updates=ed_updates)

        e_val_fn = theano.function([inputs, small_targets, weights], [val_e_loss, e_val_out])
        ed_val_fn = theano.function([inputs, targets, weights], [val_ed_loss, ed_val_out])
        ed_test_fn = theano.function([inputs], [ed_val_out])

        print("...Done")
        return encoder, decoder, e_train_fn, ed_train_fn, e_val_fn, ed_val_fn, ed_test_fn

    def train(self):
        # Make training functions
        print("Making Training Functions...")
        encoder, decoder, e_train_fn, ed_train_fn, e_val_fn, ed_val_fn, _ = self.build_train_fns()

        # Load in params if training incomplete
        try:

            # Load training statistics
            start_epoch = np.load(self.base + 'stats/epoch_e.npy')[0]
            err = np.load(self.base + 'stats/err_e.npy')
            val_err = np.load(self.base + 'stats/val_err_e.npy')

            # Load models
            with np.load(self.base + 'models/encoder_e' + str(start_epoch) + '.npz') as f:
                param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(encoder['classifier'], param_values)

            start_epoch += 1
            print("...Loaded previous models")
        except IOError:
            start_epoch = 0
            err = np.zeros((self.num_epochs)).astype(np.float32)
            val_err = np.zeros((self.num_epochs)).astype(np.float32)
            print("...Loaded models")

        print("Starting ENET Encoder Training...")
        for epoch in range(start_epoch, self.num_epochs):
                                    
            if epoch % 10 == 0 and epoch > 30:
                self.lr = self.lr / float(2)            
            
            start_time = time.time()
            num_batches = 0
            err_epoch = 0
            im_count = 0

            # Load specified amount of images into memory at once
            for X_files_mem, y_files_mem, y_small_mem in iterate_membatches(self.X_files_train,
                                                               self.y_files_train,
                                                               self.images_in_mem,
                                                               self.dataset.load_files,
                                                               shuffle=True):

                for inputs, targets, small_targets, _, wgts_small in iterate_minibatches(X_files_mem, y_files_mem, y_small_mem,
                                                                          self.bz, shuffle=True):

                    # Train the encoder by itself
                    err_epoch = e_train_fn(inputs, small_targets, wgts_small, self.lr)[0]
                    err[epoch] += err_epoch
                    num_batches += 1

            # Display training stats
            print("Epoch {} of {} took {:.3f} minutes".format(epoch + 1, self.num_epochs,
                                                              (time.time() - start_time) / np.float32(60)))
            err[epoch] = err[epoch] / num_batches
            print("  Error:\t\t{}".format(err[epoch]))

            # Save stats + models
            np.save(self.base + 'stats/epoch_e.npy', np.array([epoch]))
            np.save(self.base + 'stats/err_e.npy', err)
            np.savez(self.base + 'models/encoder_e' + str(epoch) + '.npz',
                     *lasagne.layers.get_all_param_values(encoder['classifier']))

            # Do a pass on validation data every 3 epochs
            if (epoch + 1) % 5 == 0:
                val_err_epoch = 0
                val_targets = np.zeros((self.X_files_val.shape[0], small_targets.shape[1],
                                        small_targets.shape[2], small_targets.shape[3])).astype(np.float32)
                val_segs = np.zeros((self.X_files_val.shape[0], small_targets.shape[1],
                                     small_targets.shape[2], small_targets.shape[3])).astype(np.float32)
                val_images = np.zeros((self.X_files_val.shape[0], 3,
                                     targets.shape[2], targets.shape[3])).astype(np.float32)
                im_count = 0
                num_batches = 0
                for X_files_mem, y_files_mem, y_small_mem in iterate_membatches(self.X_files_val,
                                                                   self.y_files_val,
                                                                   self.images_in_mem,
                                                                   self.dataset.load_files,
                                                                   shuffle=False):
    
                    for inputs, targets, small_targets, _, wgts_small in iterate_minibatches(X_files_mem, y_files_mem, y_small_mem,
                                                                              self.bz, shuffle=True):
                        ve, val_ims = e_val_fn(inputs, small_targets, wgts_small)
                        val_err_epoch += ve
                        
                        val_images[im_count : im_count + small_targets.shape[0], :, :, :] = inputs
                        val_targets[im_count : im_count + small_targets.shape[0], :, :, : ] = small_targets
                        val_segs[im_count: im_count + small_targets.shape[0], :, :, :] = val_ims
                        im_count += small_targets.shape[0]
                        
                        num_batches += 1
                        
                show_examples(val_images, val_segs, val_targets, self.num_examples, 
                              epoch, self.base + 'images/e_epoch' + str(epoch) + '.png')
                val_err[epoch] = val_err_epoch / float(num_batches)
                print("  Validation Error:\t\t{}".format(val_err[epoch]))
                IU = intersection_over_union(val_segs, val_targets)
                print("  Validation Mean IU:\t\t{}".format(IU))
                np.save(self.base + 'stats/val_err_e.npy', val_err)
            
        print("...Finished Enet Encoder Training")

        print("Starting ENET Encoder-Decoder Training...")
        
        # Reset lr
        self.lr = self.start_lr
        
        with np.load(self.base + 'models/encoder_e' + str(self.num_epochs - 1) + '.npz') as f:
                param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(encoder['classifier'], param_values)

        try:

            # Load training statistics
            start_epoch = np.load(self.base + 'stats/epoch_ed.npy')[0]
            err = np.load(self.base + 'stats/err_ed.npy')
            val_err = np.load(self.base + 'stats/val_err_ed.npy')

            # Load models
            with np.load(self.base + 'models/decoder_ed' + str(start_epoch) + '.npz') as f:
                param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(decoder['out'], param_values)

            with np.load(self.base + 'models/encoder_ed' + str(start_epoch) + '.npz') as f:
                param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(encoder['classifier'], param_values)

            start_epoch += 1
            print("...Loaded previous models")
        except IOError:
            start_epoch = 0
            err = np.zeros((self.num_epochs_ed)).astype(np.float32)
            val_err = np.zeros((self.num_epochs_ed)).astype(np.float32)
            print("...Loaded models")

        for epoch in range(start_epoch, self.num_epochs_ed):
            
            if epoch >= 120 and (epoch-20) % 50 == 0:
                self.lr = np.float32(self.lr / np.float32(10))
            
            start_time = time.time()
            num_batches = 0
            err_epoch = 0

            # Load specified amount of images into memory at once
            for X_files_mem, y_files_mem, y_small_mem in iterate_membatches(self.X_files_train,
                                                               self.y_files_train,
                                                               self.images_in_mem,
                                                               self.dataset.load_files,
                                                               shuffle=True):

                for inputs, targets, _, wgts_targets, _ in iterate_minibatches(X_files_mem, y_files_mem, y_small_mem,
                                                                          self.bz, shuffle=True):

                    # Train the encoder by itself
                    err_epoch = ed_train_fn(inputs, targets, wgts_targets, self.lr)[0]
                    num_batches += 1
                    err[epoch]+= err_epoch

            # Display training stats
            err[epoch] = err[epoch] / float(num_batches)
            print("Epoch {} of {} took {:.3f} minutes".format(epoch + 1, self.num_epochs_ed,
                                                              (time.time() - start_time) / np.float32(60)))
            print("  Error:\t\t{}".format(err[epoch]))

            # Save stats + models
            np.save(self.base + 'stats/epoch_ed.npy', np.array([epoch]))
            np.save(self.base + 'stats/err_ed.npy', err)
            np.savez(self.base + 'models/encoder_ed' + str(epoch) + '.npz',
                     *lasagne.layers.get_all_param_values(encoder['classifier']))
            np.savez(self.base + 'models/decoder_ed' + str(epoch) + '.npz',
                     *lasagne.layers.get_all_param_values(decoder['out']))

            # Do a pass on validation data
            if(epoch + 1) % 5 == 0:
                val_err_epoch = 0
                val_images = np.zeros((self.X_files_val.shape[0], 3,
                                        targets.shape[2], targets.shape[3])).astype(np.float32)
                val_segs = np.zeros((self.X_files_val.shape[0], targets.shape[1],
                                     targets.shape[2], targets.shape[3])).astype(np.float32)
                val_targets = np.zeros((self.X_files_val.shape[0], targets.shape[1],
                                     targets.shape[2], targets.shape[3])).astype(np.float32)
                im_count = 0
                for X_files_mem, y_files_mem, y_small_mem in iterate_membatches(self.X_files_val,
                                                                   self.y_files_val,
                                                                   self.images_in_mem,
                                                                   self.dataset.load_files,
                                                                   shuffle=False):
    
                    for inputs, targets, small_targets, wgts_targets, _ in iterate_minibatches(X_files_mem, y_files_mem, y_small_mem,
                                                                              self.bz, shuffle=True):
    
                        ed_err, val_ims = ed_val_fn(inputs, targets, wgts_targets)
                        val_err_epoch += ed_err
                        val_images[im_count : im_count + targets.shape[0], :, :, :] = inputs
                        val_segs[im_count: im_count + targets.shape[0], :, :, :] = val_ims
                        val_targets[im_count: im_count + targets.shape[0], :, :, :] = targets
    
                        im_count += small_targets.shape[0]
                        num_batches += 1
    
                show_examples(val_images, val_segs, val_targets, self.num_examples, 
                              epoch, self.base + 'images/ed_epoch' + str(epoch) + '.png')
                np.save(self.base + 'stats/val_err_ed.npy', val_err)
                val_err[epoch] = val_err_epoch / float(num_batches)
                print("  Validation Error:\t\t{}".format(val_err[epoch]))
                IU = intersection_over_union(val_segs, val_targets)
                print("  Validation Mean IU:\t\t{}".format(IU))

        print("...Finished Enet Encoder-Decoder Training")

        # Save final models
        np.savez(self.base + 'models/encoder.npz', *lasagne.layers.get_all_param_values(encoder['classifier']))
        np.savez(self.base + 'models/decoder.npz', *lasagne.layers.get_all_param_values(decoder['out']))

        print("...ENET Training Complete")

    def test(self):
        encoder, decoder, _, _, e_val_fn, ed_val_fn, ed_test_fn = self.build_train_fns()

        np.savez(self.base + 'models/encoder.npz', *lasagne.layers.get_all_param_values(encoder['classifier']))
        np.savez(self.base + 'models/decoder.npz', *lasagne.layers.get_all_param_values(decoder['out']))

        # Load models
        with np.load(self.base + 'models/decoder.npz') as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(decoder['out'], param_values)

        with np.load(self.base + 'models/encoder.npz') as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(encoder['classifier'], param_values)

        # Do forward passes on test data and check result
        val_err_epoch = 0
        num_batches = 0
        val_images = np.zeros((self.X_files_test.shape[0], 3,
                               360, 480)).astype(np.float32)
        val_segs = np.zeros((self.X_files_val.shape[0], 12, 360, 480)).astype(np.float32)
        val_targets = np.zeros((self.X_files_val.shape[0], 12, 360, 480)).astype(np.float32)
        im_count = 0
        for X_files_mem, y_files_mem, y_small_mem in iterate_membatches(self.X_files_test,
                                                                        self.y_files_test,
                                                                        self.images_in_mem,
                                                                        self.dataset.load_files,
                                                                        shuffle=False):

            for inputs, targets, small_targets, wgts_targets, _ in iterate_minibatches(X_files_mem, y_files_mem,
                                                                                       y_small_mem,
                                                                                       self.bz, shuffle=True):
                ed_err, val_ims = ed_val_fn(inputs, targets, wgts_targets)
                val_err_epoch += ed_err
                val_images[im_count: im_count + targets.shape[0], :, :, :] = inputs
                val_segs[im_count: im_count + targets.shape[0], :, :, :] = val_ims
                val_targets[im_count: im_count + targets.shape[0], :, :, :] = targets

                im_count += small_targets.shape[0]
                num_batches += 1

        show_examples(val_images, val_segs, val_targets, self.num_examples, None,
                      self.base + 'images/test.png')
        IU = intersection_over_union(val_segs, val_targets)
        print("  Test Mean IU:\t\t{}".format(IU))

        # Time a forward pass with one image
        input_gpu = theano.shared(np.expand_dims(val_images[0], axis=0))
        start_t = time.time()
        output = ed_test_fn(input_gpu)
        print(time.time() - start_t)

        # Do forward passes on just images alone, stitch to video
        val_images = np.zeros((self.X_files_test.shape[0], 3,
                               360, 480)).astype(np.float32)
        count = 0
        for inputs in iterate_videobatches(self.X_files_video, 5, self.video_loader):
            segs = ed_test_fn(inputs)
            val_images[count : count + segs.shape[0], :, :, :] = tint_images(inputs, segs)
            count += segs.shape[0]

        for n in range(0, count):

            fn = 'img' + str(n)
            if len(fn) < 5:
                fn = '00' + fn
            elif len(fn) < 6:
                fn = '0' + fn

            fn += '.png'

            misc.imsave(self.base + 'video/' + fn, val_images[n, :, :, :])






