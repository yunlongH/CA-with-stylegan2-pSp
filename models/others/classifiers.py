import torch
import torch.nn as nn
import torch.nn.functional as F

# class Classifier256(nn.Module):
#     def __init__(self, input_channels=3, num_classes=1):
#         super(Classifier256, self).__init__()
        
#         # Convolutional layers with increasing feature maps and batch normalization
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU()
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU()
#         )
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU()
#         )
#         self.conv4 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU()
#         )
#         self.conv5 = nn.Sequential(
#             nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU()
#         )
        
#         # Adaptive average pooling to handle the fixed output size (8x8 after convolutions)
#         self.avg_pool = nn.AdaptiveAvgPool2d((4, 4))  # 4x4 output after pooling
        
#         # Fully connected layers with dropout for regularization
#         self.fc1 = nn.Linear(512 * 4 * 4, 512)
#         self.dropout = nn.Dropout(0.5)
#         self.fc2 = nn.Linear(512, num_classes)

#     def forward(self, x):
#         # Apply convolutional layers
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.conv4(x)
#         x = self.conv5(x)
        
#         # Adaptive average pooling to reduce spatial dimensions
#         x = self.avg_pool(x)
        
#         # Flatten for fully connected layers
#         x = x.view(x.size(0), -1)
        
#         # Fully connected layers with dropout
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)  # Sigmoid for binary classification
        
#         return x

class SimpleClassifier(nn.Module):
    def __init__(self, args):
        super(SimpleClassifier, self).__init__()
        input_channels = args.input_nc
        self.conv_layers = nn.Sequential(
            # Convolutional Block 1
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 32 x 128 x 128

            # Convolutional Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 64 x 64 x 64

            # Convolutional Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 128 x 32 x 32

            # Convolutional Block 4
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 256 x 16 x 16
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 16 * 16, 512),  # Flattening
            nn.ReLU(),
            nn.Dropout(0.5),  # Prevent overfitting
            nn.Linear(512, 1),  # Binary classification (output 1 value)
            #nn.Sigmoid()  # Probability output
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc_layers(x)
        return x


class Discriminator_256(nn.Module):
    def __init__(self, args):
        super(Discriminator_256, self).__init__()
        self.args = args
        ndf = 64

        # Additional convolutional layers to handle 256x256 input
        self.conv1 = nn.Conv2d(self.args.input_nc, ndf, 4, 2, 1, bias=False)  # Output: 128x128
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)  # Output: 64x64
        self.bn2 = nn.BatchNorm2d(ndf * 2)

        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)  # Output: 32x32
        self.bn3 = nn.BatchNorm2d(ndf * 4)

        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)  # Output: 16x16
        self.bn4 = nn.BatchNorm2d(ndf * 8)

        self.conv5 = nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False)  # Output: 8x8
        self.bn5 = nn.BatchNorm2d(ndf * 16)

        self.conv6 = nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias=False)  # Output: 4x4
        self.bn6 = nn.BatchNorm2d(ndf * 32)

        self.class_conv = nn.Conv2d(ndf * 32, 1, 4, 1, 0, bias=False)  # Final classifier

        #self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # Forward pass through the layers
        out = self.lrelu(self.conv1(input))  # 128x128
        out = self.lrelu(self.bn2(self.conv2(out)))  # 64x64
        out = self.lrelu(self.bn3(self.conv3(out)))  # 32x32
        out = self.lrelu(self.bn4(self.conv4(out)))  # 16x16
        out = self.lrelu(self.bn5(self.conv5(out)))  # 8x8
        out = self.lrelu(self.bn6(self.conv6(out)))  # 4x4

        classe = self.class_conv(out)  # Final output layer
        classe = classe.view(classe.shape[0], -1)  # Flatten for classification
        #classe = self.sigmoid(classe)  # Apply sigmoid activation

        return classe
    


class Discriminator_256(nn.Module):
    def __init__(self, args):
        super(Discriminator_256, self).__init__()
        self.args = args
        ndf = 64

        # Additional convolutional layers to handle 256x256 input
        self.conv1 = nn.Conv2d(self.args.input_nc, ndf, 4, 2, 1, bias=False)  # Output: 128x128
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)  # Output: 64x64
        self.bn2 = nn.BatchNorm2d(ndf * 2)

        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)  # Output: 32x32
        self.bn3 = nn.BatchNorm2d(ndf * 4)

        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)  # Output: 16x16
        self.bn4 = nn.BatchNorm2d(ndf * 8)

        self.conv5 = nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False)  # Output: 8x8
        self.bn5 = nn.BatchNorm2d(ndf * 16)

        self.conv6 = nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias=False)  # Output: 4x4
        self.bn6 = nn.BatchNorm2d(ndf * 32)

        self.class_conv = nn.Conv2d(ndf * 32, 1, 4, 1, 0, bias=False)  # Final classifier

        #self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # Forward pass through the layers
        out = self.lrelu(self.conv1(input))  # 128x128
        out = self.lrelu(self.bn2(self.conv2(out)))  # 64x64
        out = self.lrelu(self.bn3(self.conv3(out)))  # 32x32
        out = self.lrelu(self.bn4(self.conv4(out)))  # 16x16
        out = self.lrelu(self.bn5(self.conv5(out)))  # 8x8
        out = self.lrelu(self.bn6(self.conv6(out)))  # 4x4

        classe = self.class_conv(out)  # Final output layer
        classe = classe.view(classe.shape[0], -1)  # Flatten for classification
        #classe = self.sigmoid(classe)  # Apply sigmoid activation

        return classe


class Discriminator_256_dp(nn.Module):
    def __init__(self, args):
        super(Discriminator_256, self).__init__()
        self.args = args
        ndf = 64

        # Additional convolutional layers to handle 256x256 input
        self.conv1 = nn.Conv2d(self.args.input_nc, ndf, 4, 2, 1, bias=False)  # Output: 128x128
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(0.5)  # Dropout with probability 0.5

        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)  # Output: 64x64
        self.bn2 = nn.BatchNorm2d(ndf * 2)

        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)  # Output: 32x32
        self.bn3 = nn.BatchNorm2d(ndf * 4)

        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)  # Output: 16x16
        self.bn4 = nn.BatchNorm2d(ndf * 8)

        self.conv5 = nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False)  # Output: 8x8
        self.bn5 = nn.BatchNorm2d(ndf * 16)

        self.conv6 = nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias=False)  # Output: 4x4
        self.bn6 = nn.BatchNorm2d(ndf * 32)

        self.class_conv = nn.Conv2d(ndf * 32, 1, 4, 1, 0, bias=False)  # Final classifier

    def forward(self, input):
        # Forward pass through the layers
        out = self.lrelu(self.conv1(input))  # 128x128
        out = self.dropout(out)  # Apply dropout
        out = self.lrelu(self.bn2(self.conv2(out)))  # 64x64
        out = self.dropout(out)  # Apply dropout
        out = self.lrelu(self.bn3(self.conv3(out)))  # 32x32
        out = self.dropout(out)  # Apply dropout
        out = self.lrelu(self.bn4(self.conv4(out)))  # 16x16
        out = self.dropout(out)  # Apply dropout
        out = self.lrelu(self.bn5(self.conv5(out)))  # 8x8
        out = self.dropout(out)  # Apply dropout
        out = self.lrelu(self.bn6(self.conv6(out)))  # 4x4
        out = self.dropout(out)  # Apply dropout

        classe = self.class_conv(out)  # Final output layer
        classe = classe.view(classe.shape[0], -1)  # Flatten for classification

        return classe

class Discriminator_256IN(nn.Module):
    def __init__(self, args):
        super(Discriminator_256IN, self).__init__()
        self.args = args
        ndf = 64

        self.conv1 = nn.Conv2d(self.args.input_nc, ndf, 4, 2, 1, bias=False)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.bn2 = nn.InstanceNorm2d(ndf * 2, affine=True)

        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.bn3 = nn.InstanceNorm2d(ndf * 4, affine=True)

        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
        self.bn4 = nn.InstanceNorm2d(ndf * 8, affine=True)

        self.conv5 = nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False)
        self.bn5 = nn.InstanceNorm2d(ndf * 16, affine=True)

        self.conv6 = nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias=False)
        self.bn6 = nn.InstanceNorm2d(ndf * 32, affine=True)

        self.class_conv = nn.Conv2d(ndf * 32, 1, 4, 1, 0, bias=False)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        out = self.lrelu(self.conv1(input))
        out = self.lrelu(self.bn2(self.conv2(out)))
        out = self.lrelu(self.bn3(self.conv3(out)))
        out = self.lrelu(self.bn4(self.conv4(out)))
        out = self.lrelu(self.bn5(self.conv5(out)))
        out = self.lrelu(self.bn6(self.conv6(out)))
        classe = self.class_conv(out)
        classe = classe.view(classe.shape[0], -1)
        #classe = self.sigmoid(classe)
        return classe


class Discriminator_256LN(nn.Module):
    def __init__(self, args):
        super(Discriminator_256LN, self).__init__()
        self.args = args
        ndf = 64

        self.conv1 = nn.Conv2d(self.args.input_nc, ndf, 4, 2, 1, bias=False)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.ln2 = nn.LayerNorm([ndf * 2, 64, 64])  # Adjusted for 64x64 output

        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.ln3 = nn.LayerNorm([ndf * 4, 32, 32])  # Adjusted for 32x32 output

        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
        self.ln4 = nn.LayerNorm([ndf * 8, 16, 16])  # Adjusted for 16x16 output

        self.conv5 = nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False)
        self.ln5 = nn.LayerNorm([ndf * 16, 8, 8])  # Adjusted for 8x8 output

        self.conv6 = nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias=False)
        self.ln6 = nn.LayerNorm([ndf * 32, 4, 4])  # Adjusted for 4x4 output

        self.class_conv = nn.Conv2d(ndf * 32, 1, 4, 1, 0, bias=False)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        out = self.lrelu(self.conv1(input))
        out = self.lrelu(self.ln2(self.conv2(out)))
        out = self.lrelu(self.ln3(self.conv3(out)))
        out = self.lrelu(self.ln4(self.conv4(out)))
        out = self.lrelu(self.ln5(self.conv5(out)))
        out = self.lrelu(self.ln6(self.conv6(out)))
        classe = self.class_conv(out)
        classe = classe.view(classe.shape[0], -1)
        #classe = self.sigmoid(classe)
        return classe



class Discriminator_256LN_dp(nn.Module):
    def __init__(self, args):
        """
        Discriminator with LayerNorm and Dropout regularization.
        
        Args:
        - args: Arguments containing model hyperparameters (e.g., input_nc).
        - dropout_prob: Dropout probability (default: 0.5).
        """
        super(Discriminator_256LN_dp, self).__init__()
        self.args = args
        ndf = 64

        self.conv1 = nn.Conv2d(self.args.input_nc, ndf, 4, 2, 1, bias=False)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(0.5)  # Define Dropout layer

        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.ln2 = nn.LayerNorm([ndf * 2, 64, 64])  # Adjusted for 64x64 output

        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.ln3 = nn.LayerNorm([ndf * 4, 32, 32])  # Adjusted for 32x32 output

        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
        self.ln4 = nn.LayerNorm([ndf * 8, 16, 16])  # Adjusted for 16x16 output

        self.conv5 = nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False)
        self.ln5 = nn.LayerNorm([ndf * 16, 8, 8])  # Adjusted for 8x8 output

        self.conv6 = nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias=False)
        self.ln6 = nn.LayerNorm([ndf * 32, 4, 4])  # Adjusted for 4x4 output

        self.class_conv = nn.Conv2d(ndf * 32, 1, 4, 1, 0, bias=False)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        out = self.lrelu(self.conv1(input))
        out = self.dropout(out)  # Apply dropout after the first layer

        out = self.lrelu(self.ln2(self.conv2(out)))
        out = self.dropout(out)  # Apply dropout after the second layer

        out = self.lrelu(self.ln3(self.conv3(out)))
        out = self.dropout(out)  # Apply dropout after the third layer

        out = self.lrelu(self.ln4(self.conv4(out)))
        out = self.dropout(out)  # Apply dropout after the fourth layer

        out = self.lrelu(self.ln5(self.conv5(out)))
        out = self.dropout(out)  # Apply dropout after the fifth layer

        out = self.lrelu(self.ln6(self.conv6(out)))

        classe = self.class_conv(out)
        classe = classe.view(classe.shape[0], -1)
        return classe


class Discriminator_256IN_dp(nn.Module):
    def __init__(self, args):
        super(Discriminator_256IN_dp, self).__init__()
        self.args = args
        ndf = 64

        self.conv1 = nn.Conv2d(self.args.input_nc, ndf, 4, 2, 1, bias=False)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(0.5)  # Dropout with 50% probability

        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.bn2 = nn.InstanceNorm2d(ndf * 2, affine=True)

        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.bn3 = nn.InstanceNorm2d(ndf * 4, affine=True)

        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
        self.bn4 = nn.InstanceNorm2d(ndf * 8, affine=True)

        self.conv5 = nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False)
        self.bn5 = nn.InstanceNorm2d(ndf * 16, affine=True)

        self.conv6 = nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias=False)
        self.bn6 = nn.InstanceNorm2d(ndf * 32, affine=True)

        self.class_conv = nn.Conv2d(ndf * 32, 1, 4, 1, 0, bias=False)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        out = self.lrelu(self.conv1(input))
        out = self.dropout(out)  # Apply dropout after the first layer
        out = self.lrelu(self.bn2(self.conv2(out)))
        out = self.dropout(out)  # Apply dropout after the second layer
        out = self.lrelu(self.bn3(self.conv3(out)))
        out = self.dropout(out)  # Apply dropout after the third layer
        out = self.lrelu(self.bn4(self.conv4(out)))
        out = self.dropout(out)  # Apply dropout after the fourth layer
        out = self.lrelu(self.bn5(self.conv5(out)))
        out = self.dropout(out)  # Apply dropout after the fifth layer
        out = self.lrelu(self.bn6(self.conv6(out)))
        classe = self.class_conv(out)
        classe = classe.view(classe.shape[0], -1)
        return classe
