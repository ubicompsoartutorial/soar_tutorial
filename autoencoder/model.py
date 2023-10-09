import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, padding_mode='reflect', dropout_prob=0.2):
        super(ConvBlock, self).__init__()

        # 1D convolutional layer
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              padding_mode=padding_mode,
                              bias=False)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)
        # self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, inputs):
        conv = self.conv(inputs)
        # bn = self.bn(conv)
        relu = self.relu(conv)
        # dropout = self.dropout(relu)

        return relu


class AutoEncoder(nn.Module):
    def __init__(self, args):
        super(AutoEncoder, self).__init__()

        # Encoder
        self.encoder = Encoder(args=args)

        # Decoder
        self.decoder = Decoder(args=args)

    def forward(self, inputs):
        # Passing it through the encoder
        encoder = self.encoder(inputs)

        # Decoding
        decoder = self.decoder(encoder)

        # Transposing dimensions from BxCxT to BxTxC
        decoder = decoder.permute(0, 2, 1)

        return decoder


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.conv_1 = ConvBlock(in_channels=args['input_size'],
                                out_channels=32,
                                kernel_size=args['kernel_size'],
                                padding=args['padding'])
        self.conv_2 = ConvBlock(in_channels=32,
                                out_channels=64,
                                kernel_size=args['kernel_size'],
                                padding=args['padding'])
        self.conv_3 = ConvBlock(in_channels=64,
                                out_channels=128,
                                kernel_size=args['kernel_size'],
                                padding=args['padding'])

    def forward(self, inputs):
        inputs = inputs.transpose(1, 2)
        conv_1 = self.conv_1(inputs)
        conv_2 = self.conv_2(conv_1)
        conv_3 = self.conv_3(conv_2)

        return conv_3


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        # Decoder
        self.conv_1 = ConvBlock(in_channels=128,
                                out_channels=64,
                                kernel_size=args['kernel_size'],
                                padding=args['padding'])
        self.conv_2 = ConvBlock(in_channels=64,
                                out_channels=32,
                                kernel_size=args['kernel_size'],
                                padding=args['padding'])
        self.conv_3 = ConvBlock(in_channels=32,
                                out_channels=args['input_size'],
                                kernel_size=args['kernel_size'],
                                padding=args['padding'])

    def forward(self, inputs):
        conv_1 = self.conv_1(inputs)
        conv_2 = self.conv_2(conv_1)
        conv_3 = self.conv_3(conv_2)

        return conv_3


class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()

        # Encoder
        self.encoder = Encoder(args=args)

        # Softmax
        if args['classification_model'] == 'linear':
            self.softmax = nn.Linear(128, args['num_classes'])
        elif args['classification_model']== 'mlp':
            self.softmax = nn.Sequential(nn.Linear(128, 256),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(256, args['num_classes']))

    def forward(self, inputs):
        # Passing it through the encoder
        encoding = self.encoder(inputs)

        # Max pooling
        # Global Max Pooling (as per
        # https://github.com/keras-team/keras/blob
        # /7a39b6c62d43c25472b2c2476bd2a8983ae4f682/keras/layers/pooling.py
        # #L559) for 'channels_first'
        pool = F.max_pool1d(encoding, kernel_size=encoding.shape[2]).squeeze(2)

        softmax = self.softmax(pool)

        return softmax

    def load_pretrained_weights(self, args):
        state_dict_path = os.path.join(args['saved_model'])

        print('Loading the pre-trained weights')
        checkpoint = torch.load(state_dict_path, map_location=args['device'])
        pretrained_checkpoint = checkpoint['model_state_dict']

        model_dict = self.state_dict()

        # What weights are *not* copied
        missing = \
            {k: v for k, v in pretrained_checkpoint.items() if
             k not in model_dict}
        print("The weights from saved model not in classifier are: {}".format(
            missing.keys()))

        missing = \
            {k: v for k, v in model_dict.items() if
             k not in pretrained_checkpoint}
        print("The weights from classifier not in the saved model are: {}"
              .format(missing.keys()))

        self.load_state_dict(pretrained_checkpoint, False)

        return

    def freeze_encoder_layers(self):
        """
        To set only the softmax to be trainable
        :return: None, just setting the encoder part as frozen
        """
        # First setting the model to eval
        self.encoder.eval()

        # Then setting the requires_grad to False
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        return
