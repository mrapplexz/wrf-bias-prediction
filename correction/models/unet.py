import torch
from torch import nn
from transformers.models.bert.modeling_bert import BertEncoder
from transformers.models.bert.configuration_bert import BertConfig


from correction.models.unet_parts import DoubleConv, Down, Up, OutConv


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # self.spatial_map_encode = nn.Parameter(128, 105, 140)

        chan_factor = 2

        # self.spatial_map_encode = nn.Parameter(torch.empty([1, 128, 52, 70]))
        # self.spatial_map_decode = nn.Parameter(torch.empty([1, 128, 52, 70]))

        # nn.init.normal_(self.spatial_map_encode)
        # nn.init.normal_(self.spatial_map_decode)

        self.patch_encoding = nn.Embedding(13 * 17, 512)
        self.pos_encoding = nn.Embedding(10, 512)

        self.causal_bro = BertEncoder(BertConfig(
            hidden_size=512,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=512 * 4
        ))

        self.inc = (DoubleConv(n_channels, 64 // chan_factor))
        self.down1 = (Down(64 // chan_factor, 128 // chan_factor))
        self.down2 = (Down(128 // chan_factor, 256 // chan_factor))
        self.down3 = (Down(256 // chan_factor, 512 // chan_factor))
        self.down4 = (Down(512 // chan_factor, 1024 // chan_factor))
        self.up1 = (Up(1024 // chan_factor, 512 // chan_factor))
        self.up2 = (Up(512 // chan_factor, 256 // chan_factor))
        self.up3 = (Up(256 // chan_factor, 128 // chan_factor))
        self.up4 = (Up(128 // chan_factor, 64 // chan_factor))
        self.outc = (OutConv(64 // chan_factor, n_classes))

    def forward(self, x):
        orig_shape = x.shape
        x = x.view(-1, x.shape[-3], x.shape[-2], x.shape[-1])
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        # x3 = x3 + self.spatial_map_encode
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x5 = x5.view(orig_shape[0], orig_shape[1], 512, 13 * 17)
        x5 = x5.permute(1, 0, 3, 2)
        x5_patch = self.patch_encoding(torch.arange(0, x5.shape[2], device=x5.device))
        x5_pos = self.pos_encoding(torch.arange(0, x5.shape[1], device=x5.device))
        x5 = x5 + x5_patch.view(1, 1, 13 * 17, 512) + x5_pos.view(1, orig_shape[0], 1, 512)
        x5 = x5.reshape(x5.shape[0], -1, 512)
        x5 = self.causal_bro(x5).last_hidden_state
        x5 = x5.reshape(x5.shape[0] * orig_shape[0], 13, 17, 512)
        x5 = x5.permute(0, 3, 1, 2)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        # x = x + self.spatial_map_decode
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
