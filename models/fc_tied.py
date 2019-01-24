import torch
import torch.nn as nn
import torch.nn.init as weight_init


class FCNet(torch.nn.Module):
    def __init__(self, **kwargs):
        super(FCNet, self).__init__()
        self.input_size = kwargs.get("input_size", 3)
        layer1_size = kwargs.get("layer1_size", 512)
        layer2_size = kwargs.get("layer2_size", 512)
        layer3_size = kwargs.get("layer3_size", 1024)
        self.tie_arch = kwargs.get("tie_arch", False)
        self.batchnorm = kwargs.get("batchnorm", False)
        encoder_layers_sizes = kwargs.get("encoder_layers_sizes", [512, 512, 1024])
        # Decoder is the mirrored encoder
        decoder_layers_sizes = list(reversed([self.input_size] + encoder_layers_sizes[:-1]))

        self.input_dropout = torch.nn.Dropout(p=kwargs.get("input_dropout_coef", 0.0))

        # Generic generation of encoder layers:
        encoder_layers_lst = self._create_layers(encoder_layers_sizes, prev_layer_size=self.input_size,
                                                 bias=not self.batchnorm)
        self.encoder_layers = nn.ModuleList(encoder_layers_lst)

        # Dropout
        self.dropout = torch.nn.Dropout(p=kwargs.get("dropout_coef", 0.8))

        # Generic generation of encoder layers:
        self.encoder_weights_reversed = None
        decoder_layers_lst = self._create_layers(decoder_layers_sizes, prev_layer_size=encoder_layers_sizes[-1],
                                                 bias=not self.batchnorm)
        self.decoder_layers = nn.ModuleList(decoder_layers_lst)

        if self.tie_arch:
            # Create a list of encoder weights (we will use them during forward)
            self.encoder_weights_reversed = []
            for layer in self.encoder_layers:
                if type(layer) == nn.Linear:
                    self.encoder_weights_reversed.insert(0, layer.weight)
            # Create decoder bias
            self.decode_bias = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(layer_size))
                                                       for layer_size in decoder_layers_sizes])


        # return
        # ########################################################################
        # # If BatchNorm is used, dont add bias (added as the beta parameter after the batchnorm)
        # self.layer1 = torch.nn.Linear(self.input_size, layer1_size, bias=not self.batchnorm)
        # weight_init.xavier_uniform_(self.layer1.weight)
        # if self.layer1.bias is not None:
        #     self.layer1.bias.zero_()
        # self.layer1_bn = torch.nn.Sequential([])
        # if self.batchnorm:
        #     self.layer1_bn = torch.nn.BatchNorm1d(layer1_size)
        #
        # self.layer2 = torch.nn.Linear(layer1_size, layer2_size, )
        # weight_init.xavier_uniform_(self.layer2.weight)
        # self.layer2.bias.zero_()
        # self.layer3 = torch.nn.Linear(layer2_size, layer3_size)
        # weight_init.xavier_uniform_(self.layer3.weight)
        # self.layer3.bias.zero_()
        #
        #
        #
        # self.activation = torch.nn.SELU(inplace=False)
        #
        # if not self.tie_arch:
        #     self.layer4 = torch.nn.Linear(layer3_size, layer2_size)
        #     weight_init.xavier_uniform_(self.layer4.weight)
        #     self.layer4.bias.zero_()
        #
        #     self.layer5 = torch.nn.Linear(layer2_size, layer1_size)
        #     weight_init.xavier_uniform_(self.layer5.weight)
        #     self.layer5.bias.zero_()
        #
        #     self.layer6 = torch.nn.Linear(layer1_size, self.input_size)
        #     weight_init.xavier_uniform_(self.layer6.weight)
        #     self.layer6.bias.zero_()
        # else:
        #     self.decode_bias = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(layer_size))
        #                                                for layer_size in [layer2_size, layer1_size, self.input_size]])

    def _create_layers(self, coder_layers_sizes, prev_layer_size, bias=True):
        coder_layers_lst = []
        for layer_size in coder_layers_sizes:
            # Add fully-connected layer:
            #    (*) If BatchNorm is used, dont add bias (added as the beta parameter after the batchnorm)
            coder_layers_lst.append(nn.Linear(prev_layer_size, layer_size, bias=bias))
            # Init weights:
            weight_init.xavier_uniform_(coder_layers_lst[-1].weight)
            # Init bias (if exists):
            if coder_layers_lst[-1].bias is not None:
                coder_layers_lst[-1].bias.zero_()
            # Add batchnorm:
            if self.batchnorm:
                coder_layers_lst.append(nn.BatchNorm1d(layer_size))
            # Add activation:
            coder_layers_lst.append(nn.SELU(inplace=False))
            prev_layer_size = layer_size
        return coder_layers_lst

    def forward(self, x):
        out = x.view(-1, self.input_size)

        out = self.input_dropout(out)

        for layer in self.encoder_layers:
            out = layer(out)

        out = self.dropout(out)
        if self.tie_arch:
            linear_idx = 0
            for layer in self.decoder_layers:
                if type(layer) == nn.Linear:
                    if layer.bias is None:
                        out = torch.nn.functional.linear(input=out, weight=self.encoder_weights_reversed[linear_idx].t())
                    else:
                        out = torch.nn.functional.linear(input=out, weight=self.encoder_weights_reversed[linear_idx].t(), bias=self.decode_bias[linear_idx])
                    linear_idx += 1
                else:
                    out = layer(out)
        else:
            for layer in self.decoder_layers:
                out = layer(out)

        return out
        # # Encoder
        # out = self.activation(self.layer1(x))
        # out = self.activation(self.layer2(out))
        # out = self.activation(self.layer3(out))
        #
        # out = self.dropout(out)
        #
        # # Decoder
        # if not self.tie_arch:
        #     out = self.activation(self.layer4(out))
        #     out = self.activation(self.layer5(out))
        #     out = self.activation(self.layer6(out))
        # else:
        #     out = self.activation(
        #         torch.nn.functional.linear(input=out, weight=self.layer3.weight.t(), bias=self.decode_bias[0]))
        #     out = self.activation(
        #         torch.nn.functional.linear(input=out, weight=self.layer2.weight.t(), bias=self.decode_bias[1]))
        #     out = self.activation(
        #         torch.nn.functional.linear(input=out, weight=self.layer1.weight.t(), bias=self.decode_bias[2]))
        #
        # return out


def fc3months(input_size, input_dropout_coef=0.0):
    return FCNet(input_size=input_size, encoder_layers_sizes=[128, 256, 256], dropout_coef=0.65,
                 input_dropout_coef=input_dropout_coef)
    # return FCNet(input_size=input_size, layer1_size=128, layer2_size=256, layer3_size=256, dropout_coef=0.65)


def fc6months(input_size, input_dropout_coef=0.0):
    return FCNet(input_size=input_size, encoder_layers_sizes=[256, 256, 512], dropout_coef=0.8,
                 input_dropout_coef=input_dropout_coef)
    # return FCNet(input_size=input_size, layer1_size=256, layer2_size=256, layer3_size=512, dropout_coef=0.8)


def fc1year(input_size, input_dropout_coef=0.0):
    return FCNet(input_size=input_size, encoder_layers_sizes=[256, 256, 512], dropout_coef=0.8,
                 input_dropout_coef=input_dropout_coef)
    # return FCNet(input_size=input_size, layer1_size=256, layer2_size=256, layer3_size=512, dropout_coef=0.8)


def fc_full(input_size, input_dropout_coef=0.0):
    return FCNet(input_size=input_size, encoder_layers_sizes=[512, 512, 1024], dropout_coef=0.8,
                 input_dropout_coef=input_dropout_coef)
    # return FCNet(input_size=input_size, layer1_size=512, layer2_size=512, layer3_size=1024, dropout_coef=0.8)


def fc3months_tied(input_size, input_dropout_coef=0.0):
    return FCNet(input_size=input_size, encoder_layers_sizes=[128, 256, 256], dropout_coef=0.65, tie_arch=True,
                 input_dropout_coef=input_dropout_coef)
    # return FCNet(input_size=input_size, layer1_size=128, layer2_size=256, layer3_size=256, dropout_coef=0.65)


def fc6months_tied(input_size, input_dropout_coef=0.0):
    return FCNet(input_size=input_size, encoder_layers_sizes=[256, 256, 512], dropout_coef=0.8, tie_arch=True,
                 input_dropout_coef=input_dropout_coef)
    # return FCNet(input_size=input_size, layer1_size=256, layer2_size=256, layer3_size=512, dropout_coef=0.8)


def fc1year_tied(input_size, input_dropout_coef=0.0):
    return FCNet(input_size=input_size, encoder_layers_sizes=[256, 256, 512], dropout_coef=0.8, tie_arch=True,
                 input_dropout_coef=input_dropout_coef)
    # return FCNet(input_size=input_size, layer1_size=256, layer2_size=256, layer3_size=512, dropout_coef=0.8)


def fc_full_tied(input_size, input_dropout_coef=0.0):
    return FCNet(input_size=input_size, encoder_layers_sizes=[512, 512, 1024], dropout_coef=0.8, tie_arch=True,
                 input_dropout_coef=input_dropout_coef)
    # return FCNet(input_size=input_size, layer1_size=512, layer2_size=512, layer3_size=1024, dropout_coef=0.8)

def fc3months_bn(input_size, input_dropout_coef=0.0):
    return FCNet(input_size=input_size, encoder_layers_sizes=[128, 256, 256], dropout_coef=0.65, batchnorm=True,
                 input_dropout_coef=input_dropout_coef)
    # return FCNet(input_size=input_size, layer1_size=128, layer2_size=256, layer3_size=256, dropout_coef=0.65)


def fc6months_bn(input_size, input_dropout_coef=0.0):
    return FCNet(input_size=input_size, encoder_layers_sizes=[256, 256, 512], dropout_coef=0.8, batchnorm=True,
                 input_dropout_coef=input_dropout_coef)
    # return FCNet(input_size=input_size, layer1_size=256, layer2_size=256, layer3_size=512, dropout_coef=0.8)


def fc1year_bn(input_size, input_dropout_coef=0.0):
    return FCNet(input_size=input_size, encoder_layers_sizes=[256, 256, 512], dropout_coef=0.8, batchnorm=True,
                 input_dropout_coef=input_dropout_coef)
    # return FCNet(input_size=input_size, layer1_size=256, layer2_size=256, layer3_size=512, dropout_coef=0.8)


def fc_full_bn(input_size, input_dropout_coef=0.0):
    return FCNet(input_size=input_size, encoder_layers_sizes=[512, 512, 1024], dropout_coef=0.8, batchnorm=True,
                 input_dropout_coef=input_dropout_coef)
    # return FCNet(input_size=input_size, layer1_size=512, layer2_size=512, layer3_size=1024, dropout_coef=0.8)
