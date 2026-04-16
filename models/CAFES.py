from torch import nn


class BasicConvResBlock(nn.Module):
    def __init__(self, input_dim, n_filters, kernel_size=3, padding=1, stride=1, shortcut=False, downsample=None, dropout_rate=0, use_SEBlock=False):
        super(BasicConvResBlock, self).__init__()
        self.downsample = downsample
        self.shortcut = shortcut

        self.conv1 = nn.Conv1d(input_dim, n_filters, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_filters)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn2 = nn.BatchNorm1d(n_filters)
        self.se = SEBlock(n_filters) if use_SEBlock else nn.Identity()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.shortcut:
            if self.downsample is not None:
                residual = self.downsample(x)
            out += residual

        out = self.relu(out)
        return out




class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv1d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(x)
        return x * w


class CAFES(nn.Module):
    def __init__(self, n_conv_neurons, n_fc_neurons=1024, depth=9, n_classes=2, shortcut=False, dropout_rate=0, use_SEBlock=False, n_input_channels=6, use_first_bn=True):
        super(CAFES, self).__init__()

        layers = []
        fc_layers = []

        self.dropout_rate = dropout_rate
        self.use_SEBlock = use_SEBlock
        self.use_first_bn = use_first_bn

        if self.use_first_bn:
            layers.append(nn.BatchNorm1d(n_input_channels))
        layers.append(nn.Conv1d(n_input_channels, n_conv_neurons[0], kernel_size=19, padding=5, stride=3))
        layers.append(nn.BatchNorm1d(n_conv_neurons[0]))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool1d(2, padding=1, stride=2))
        layers.append(nn.Conv1d(n_conv_neurons[0], n_conv_neurons[1], kernel_size=3, padding=1))

        # layers.append(nn.Conv1d(1, n_conv_neurons[1], kernel_size=3, padding=1))

        if depth == 9:
            n_conv_block_1, n_conv_block_2, n_conv_block_3, n_conv_block_4 = 1, 1, 1, 1
        elif depth == 17:
            n_conv_block_1, n_conv_block_2, n_conv_block_3, n_conv_block_4 = 2, 2, 2, 2
        elif depth == 29:
            n_conv_block_1, n_conv_block_2, n_conv_block_3, n_conv_block_4 = 5, 5, 2, 2
        elif depth == 49:
            n_conv_block_1, n_conv_block_2, n_conv_block_3, n_conv_block_4 = 8, 8, 5, 3

        layers.append(
            BasicConvResBlock(input_dim=n_conv_neurons[1], n_filters=n_conv_neurons[1], kernel_size=3, padding=1,
                              shortcut=shortcut, dropout_rate=self.dropout_rate, use_SEBlock=self.use_SEBlock))
        for _ in range(n_conv_block_1 - 1):
            layers.append(
                BasicConvResBlock(input_dim=n_conv_neurons[1], n_filters=n_conv_neurons[1], kernel_size=3, padding=1,
                                  shortcut=shortcut, dropout_rate=self.dropout_rate, use_SEBlock=self.use_SEBlock))
        layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))  # l = initial length / 2

        ds = nn.Sequential(nn.Conv1d(n_conv_neurons[1], n_conv_neurons[2], kernel_size=1, stride=1, bias=False),
                           nn.BatchNorm1d(n_conv_neurons[2]))
        layers.append(
            BasicConvResBlock(input_dim=n_conv_neurons[1], n_filters=n_conv_neurons[2], kernel_size=3, padding=1,
                              shortcut=shortcut, downsample=ds, dropout_rate=self.dropout_rate, use_SEBlock=self.use_SEBlock))
        for _ in range(n_conv_block_2 - 1):
            layers.append(
                BasicConvResBlock(input_dim=n_conv_neurons[2], n_filters=n_conv_neurons[2], kernel_size=3, padding=1,
                                  shortcut=shortcut, dropout_rate=self.dropout_rate, use_SEBlock=self.use_SEBlock))
        layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))  # l = initial length / 4

        ds = nn.Sequential(nn.Conv1d(n_conv_neurons[2], n_conv_neurons[3], kernel_size=1, stride=1, bias=False),
                           nn.BatchNorm1d(n_conv_neurons[3]))
        layers.append(
            BasicConvResBlock(input_dim=n_conv_neurons[2], n_filters=n_conv_neurons[3], kernel_size=3, padding=1,
                              shortcut=shortcut, downsample=ds, dropout_rate=self.dropout_rate, use_SEBlock=self.use_SEBlock))
        for _ in range(n_conv_block_3 - 1):
            layers.append(
                BasicConvResBlock(input_dim=n_conv_neurons[3], n_filters=n_conv_neurons[3], kernel_size=3, padding=1,
                                  shortcut=shortcut, dropout_rate=self.dropout_rate, use_SEBlock=self.use_SEBlock))
        layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))  # l = initial length / 8

        ds = nn.Sequential(nn.Conv1d(n_conv_neurons[3], n_conv_neurons[4], kernel_size=1, stride=1, bias=False),
                           nn.BatchNorm1d(n_conv_neurons[4]))
        layers.append(
            BasicConvResBlock(input_dim=n_conv_neurons[3], n_filters=n_conv_neurons[4], kernel_size=3, padding=1,
                              shortcut=shortcut, downsample=ds, dropout_rate=self.dropout_rate, use_SEBlock=self.use_SEBlock))
        for _ in range(n_conv_block_4 - 1):
            layers.append(
                BasicConvResBlock(input_dim=n_conv_neurons[4], n_filters=n_conv_neurons[4], kernel_size=3, padding=1,
                                  shortcut=shortcut, dropout_rate=self.dropout_rate, use_SEBlock=self.use_SEBlock))

        layers.append(nn.AdaptiveMaxPool1d(8))
        fc_layers.extend([nn.Linear(8 * n_conv_neurons[4], n_fc_neurons), nn.ReLU()])
        # layers.append(nn.MaxPool1d(kernel_size=8, stride=2, padding=0))
        # fc_layers.extend([nn.Linear(61*512, n_fc_neurons), nn.ReLU()])

        fc_layers.extend([nn.Linear(n_fc_neurons, n_fc_neurons), nn.ReLU()])
        fc_layers.extend([nn.Linear(n_fc_neurons, n_classes)])

        self.layers = nn.Sequential(*layers)
        self.fc_layers = nn.Sequential(*fc_layers)

        self.__init_weights()

    def __init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x = x.unsqueeze(1)
        # x = self.input_norm(x)
        out = self.layers(x)
        out = out.view(out.size(0), -1)
        out = self.fc_layers(out)
        return out


