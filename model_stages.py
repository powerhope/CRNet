#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
# from base_oc_block import BaseOC_Module


from nets.stdcnet import STDCNet1446, STDCNet813
# from modules.bn import InPlaceABNSync as BatchNorm2d
BatchNorm2d = nn.BatchNorm2d

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        # self.bn = BatchNorm2d(out_chan)
        self.bn = BatchNorm2d(out_chan)
        self.relu = nn.ReLU()
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
                    
    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params
    
# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=4):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
           
#         self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 4, 1, bias=False),
#                                nn.ReLU(),
#                                nn.Conv2d(in_planes // 4, in_planes, 1, bias=False))
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.fc(self.avg_pool(x))
#         max_out = self.fc(self.max_pool(x))
#         out = avg_out + max_out
#         return self.sigmoid(out)
    
#     def get_params(self):
#         wd_params, nowd_params = [], []
#         for name, module in self.named_modules():
#             if isinstance(module, (nn.Linear, nn.Conv2d)):
#                 wd_params.append(module.weight)
#                 if not module.bias is None:
#                     nowd_params.append(module.bias)
#             elif isinstance(module, BatchNorm2d):
#                 nowd_params += list(module.parameters())
#         return wd_params, nowd_params

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class _SelfAttentionBlock(nn.Module):
    '''
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    '''
    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1):
        super(_SelfAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels == None:
            self.out_channels = in_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            BatchNorm2d(self.key_channels),
        )
        self.f_query = self.f_key
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
            kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels,
            kernel_size=1, stride=1, padding=0)
        nn.init.constant(self.W.weight, 0)
        nn.init.constant(self.W.bias, 0)


    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        value = self.f_value(x).view(batch_size, self.value_channels, -1)
        value = value.permute(0, 2, 1)
        query = self.f_query(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_key(x).view(batch_size, self.key_channels, -1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *x.size()[2:])
        context = self.W(context)
        if self.scale > 1:
#             if torch_ver == '0.4':
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=True)
#             elif torch_ver == '0.3':
#                 context = F.upsample(input=context, size=(h, w), mode='bilinear')
        return context


class SelfAttentionBlock2D(_SelfAttentionBlock):
    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1):
        super(SelfAttentionBlock2D, self).__init__(in_channels,
                                                    key_channels,
                                                    value_channels,
                                                    out_channels,
                                                    scale)


class BaseOC_Module(nn.Module):
    """
    Implementation of the BaseOC module
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: we choose 0.05 as the default value.
        size: you can apply multiple sizes. Here we only use one size.
    Return:
        features fused with Object context information.
    """
    def __init__(self, in_channels, out_channels, key_channels, value_channels, dropout, sizes=([1])):
        super(BaseOC_Module, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(in_channels, out_channels, key_channels, value_channels, size) for size in sizes])
        in_channels_change = in_channels+out_channels
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(in_channels_change, out_channels, kernel_size=1,stride=1,padding=0),
            BatchNorm2d(out_channels),
            nn.Dropout2d(dropout)
            )

    def _make_stage(self, in_channels, output_channels, key_channels, value_channels, size):
        return SelfAttentionBlock2D(in_channels,
                                    key_channels,
                                    value_channels,
                                    output_channels, 
                                    size)
        
    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]
        context = priors[0]
        for i in range(1, len(priors)):
            context += priors[i]
        output = self.conv_bn_dropout(torch.cat([context, feats], 1))
        return output


# class BaseOC_Context_Module(nn.Module):
#     """
#     Output only the context features.
#     Parameters:
#         in_features / out_features: the channels of the input / output feature maps.
#         dropout: specify the dropout ratio
#         fusion: We provide two different fusion method, "concat" or "add"
#         size: we find that directly learn the attention weights on even 1/8 feature maps is hard.
#     Return:
#         features after "concat" or "add"
#     """
#     def __init__(self, in_channels, out_channels, key_channels, value_channels, dropout, sizes=([1])):
#         super(BaseOC_Context_Module, self).__init__()
#         self.stages = []
#         self.stages = nn.ModuleList([self._make_stage(in_channels, out_channels, key_channels, value_channels, size) for size in sizes])
#         in_channels_change = in_channels+out_channels
#         self.conv_bn_dropout = nn.Sequential(
#             nn.Conv2d(in_channels_change, out_channels, kernel_size=1, padding=0),
#             BatchNorm2d(out_channels),
#             )

#     def _make_stage(self, in_channels, output_channels, key_channels, value_channels, size):
#         return SelfAttentionBlock2D(in_channels,
#                                     key_channels,
#                                     value_channels,
#                                     output_channels, 
#                                     size)
        
#     def forward(self, feats):
#         priors = [stage(feats) for stage in self.stages]
#         context = priors[0]
#         for i in range(1, len(priors)):
#             context += priors[i]
#         output = self.conv_bn_dropout(context)
#         return output



class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
#         self.chnal = ChannelAttention(mid_chan)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
#         x = self.chnal(x)
        x = self.conv_out(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params
    
# class BiSeNetOutput_gt(nn.Module):
#     def __init__(self, in_chan, mid_chan, n_classes, *args, **kwargs):
#         super(BiSeNetOutput_gt, self).__init__()
#         self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
# #         self.chnal = ChannelAttention(mid_chan)
#         self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)
#         self.init_weight()

#     def forward(self, x):
#         x = self.conv(x)
# #         x = self.chnal(x)
#         x = self.conv_out(x)
#         return x

#     def init_weight(self):
#         for ly in self.children():
#             if isinstance(ly, nn.Conv2d):
#                 nn.init.kaiming_normal_(ly.weight, a=1)
#                 if not ly.bias is None: nn.init.constant_(ly.bias, 0)

#     def get_params(self):
#         wd_params, nowd_params = [], []
#         for name, module in self.named_modules():
#             if isinstance(module, (nn.Linear, nn.Conv2d)):
#                 wd_params.append(module.weight)
#                 if not module.bias is None:
#                     nowd_params.append(module.bias)
#             elif isinstance(module, BatchNorm2d):
#                 nowd_params += list(module.parameters())
#         return wd_params, nowd_params


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size= 1, bias=False)
        # self.bn_atten = BatchNorm2d(out_chan)
        self.bn_atten = BatchNorm2d(out_chan)

        self.sigmoid_atten = nn.Sigmoid()
        self.init_weight()

    def forward(self, x):
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class ContextPath(nn.Module):
    def __init__(self, backbone='CatNetSmall', pretrain_model='', use_conv_last=False, *args, **kwargs):
        super(ContextPath, self).__init__()
        
        self.backbone_name = backbone
        if backbone == 'STDCNet1446':
            self.backbone = STDCNet1446(pretrain_model=pretrain_model, use_conv_last=use_conv_last)
#             self.arm16 = AttentionRefinementModule(512, 128)
#             self.arm8 = AttentionRefinementModule(256, 128)
            self.arm16 = BaseOC_Module(in_channels=512, out_channels=128, key_channels=256, value_channels=256, 
            dropout=0.05, sizes=([1]))
            inplanes = 1024
            if use_conv_last:
                inplanes = 1024
#             self.arm32 = AttentionRefinementModule(inplanes, 128)
            self.arm32 = BaseOC_Module(in_channels=inplanes, out_channels=128, key_channels=256, value_channels=256, 
            dropout=0.05, sizes=([1]))
            self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
            self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
            self.conv_avg = ConvBNReLU(inplanes, 128, ks=1, stride=1, padding=0)
            self.sa = SpatialAttention()

        elif backbone == 'STDCNet813':
            self.backbone = STDCNet813(pretrain_model=pretrain_model, use_conv_last=use_conv_last)
#             self.arm16 = AttentionRefinementModule(512, 128)
#             self.arm8 = BaseOC_Module(in_channels=256, out_channels=256, key_channels=256, value_channels=256, 
#             dropout=0.05, sizes=([1]))
#             self.arm8 = AttentionRefinementModule(256, 128)
            self.arm16 = BaseOC_Module(in_channels=512, out_channels=128, key_channels=256, value_channels=256, 
            dropout=0.05, sizes=([1]))
            inplanes = 1024
            if use_conv_last:
                inplanes = 1024
#             self.arm32 = AttentionRefinementModule(inplanes, 128)
            self.arm32 = BaseOC_Module(in_channels=inplanes, out_channels=128, key_channels=256, value_channels=256, 
            dropout=0.05, sizes=([1]))
            self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
            self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
            self.conv_avg = ConvBNReLU(inplanes, 128, ks=1, stride=1, padding=0)
        else:
            print("backbone is not in backbone lists")
            exit(0)

        self.init_weight()

    def forward(self, x):
        H0, W0 = x.size()[2:]

        feat2, feat4, feat8, feat16, feat32 = self.backbone(x)
        feat32 = self.sa(feat32)*feat32
        feat16 = self.sa(feat16)*feat16
        feat8 = self.sa(feat8)*feat8
        H8, W8 = feat8.size()[2:]
        H16, W16 = feat16.size()[2:]
        H32, W32 = feat32.size()[2:]
        
        avg = F.avg_pool2d(feat32, feat32.size()[2:])

        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, (H32, W32), mode='nearest')
#         feat8 = self.arm8(feat8)

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = F.interpolate(feat32_sum, (H16, W16), mode='nearest')
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, (H8, W8), mode='nearest')
        feat16_up = self.conv_head16(feat16_up)
        
        return feat2, feat4, feat8, feat16, feat16_up, feat32_up # x8, x16

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()

#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)
    
class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan,
                out_chan//4,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.conv2 = nn.Conv2d(out_chan//4,
                out_chan,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
#         self.atten = SpatialAttention()
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
#         atten = self.atten(feat)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class BiSeNet(nn.Module):
    def __init__(self, backbone, n_classes, pretrain_model='', use_boundary_2=False, use_boundary_4=False, use_boundary_8=False, use_boundary_16=False, use_conv_last=False, heat_map=False, *args, **kwargs):
        super(BiSeNet, self).__init__()
        
        self.use_boundary_2 = use_boundary_2
        self.use_boundary_4 = use_boundary_4
        self.use_boundary_8 = use_boundary_8
        self.use_boundary_16 = use_boundary_16
        # self.heat_map = heat_map
        self.cp = ContextPath(backbone, pretrain_model, use_conv_last=use_conv_last)
        
        
        
        if backbone == 'STDCNet1446':
            conv_out_inplanes = 128
            sp2_inplanes = 32
            sp4_inplanes = 64
            sp8_inplanes = 256
            sp16_inplanes = 512
            inplane = sp8_inplanes + conv_out_inplanes

        elif backbone == 'STDCNet813':
            conv_out_inplanes = 128
            sp2_inplanes = 32
            sp4_inplanes = 64
            sp8_inplanes = 256
            sp16_inplanes = 512
            inplane = sp8_inplanes + conv_out_inplanes

        else:
            print("backbone is not in backbone lists")
            exit(0)

        self.ffm = FeatureFusionModule(inplane, 256)
        self.conv_out = BiSeNetOutput(256, 256, n_classes)
        
        self.conv_out16 = BiSeNetOutput(conv_out_inplanes, 64, n_classes)
#         self.conv_out16_2 = BiSeNetOutput(96, 64, n_classes)
        
        self.conv_out32 = BiSeNetOutput(conv_out_inplanes, 64,n_classes)
#         self.conv_out32_2 = BiSeNetOutput(96, 64, n_classes)

        self.conv_out_sp16 = BiSeNetOutput(sp16_inplanes, 64, 1)
#         self.conv_out_sp16_2 = BiSeNetOutput(128, 64, 1)
        
        self.conv_out_sp8 = BiSeNetOutput(sp8_inplanes, 64, 1)
        self.conv_out_sp4 = BiSeNetOutput(sp4_inplanes, 64, 1)
        self.conv_out_sp2 = BiSeNetOutput(sp2_inplanes, 64, 1)
        
#         self.chnal_8 = ChannelAttention(sp8_inplanes)
#         self.chnal_16 = ChannelAttention(sp16_inplanes)
#         self.chnal_f = ChannelAttention(256)
#         self.conv_out_ajchan2 = ConvBNReLU(sp2_inplanes, n_classes, ks=1, stride=1, padding=0)
        self.conv_out_ajchan4 = ConvBNReLU(sp4_inplanes, n_classes, ks=1, stride=1, padding=0)
#         self.conv_out_ajchan2 = BiSeNetOutput(sp2_inplanes,32,n_classes)
#         self.conv_out_ajchan4 = BiSeNetOutput(sp4_inplanes,32,n_classes)
        self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
        
        feat_res2, feat_res4, feat_res8, feat_res16, feat_cp8, feat_cp16 = self.cp(x)

        feat_out_sp2 = self.conv_out_sp2(feat_res2)

        feat_out_sp4 = self.conv_out_sp4(feat_res4)
  
        feat_out_sp8 = self.conv_out_sp8(feat_res8)

        feat_out_sp16 = self.conv_out_sp16(feat_res16)

        feat_fuse = self.ffm(feat_res8, feat_cp8)

        feat_out = self.conv_out(feat_fuse)
        
        feat_out16 = self.conv_out16(feat_cp8)
        feat_out32 = self.conv_out32(feat_cp16)

#         feat_out = F.interpolate(feat_out, (H, W), mode='bilinear', align_corners=True)
#         feat_out16 = F.interpolate(feat_out16, (H, W), mode='bilinear', align_corners=True)
#         feat_out32 = F.interpolate(feat_out32, (H, W), mode='bilinear', align_corners=True)
        feat_out = F.interpolate(feat_out, (H//4, W//4), mode='bilinear', align_corners=True)
        feat_out = feat_out+self.conv_out_ajchan4(feat_res4)
#         feat_out = F.interpolate(feat_out, (H//2, W//2), mode='bilinear', align_corners=True)
#         feat_out = feat_out+self.conv_out_ajchan2(feat_res2)
        feat_out = F.interpolate(feat_out, (H, W), mode='bilinear', align_corners=True)
#         print("after_upsample:feat_out:")
#         print(feat_out.size())
        
        feat_out16 = F.interpolate(feat_out16, (H//4, W//4), mode='bilinear', align_corners=True)
        feat_out16 = feat_out16+self.conv_out_ajchan4(feat_res4)
#         feat_out16 = F.interpolate(feat_out16, (H//2, W//2), mode='bilinear', align_corners=True)
#         feat_out16 = feat_out16+self.conv_out_ajchan2(feat_res2)
        feat_out16 = F.interpolate(feat_out16, (H, W), mode='bilinear', align_corners=True)
#         print("after_upsample:feat_out16:")
#         print(feat_out.size())
        
        feat_out32 = F.interpolate(feat_out32, (H//4, W//4), mode='bilinear', align_corners=True)
        feat_out32 = feat_out32+self.conv_out_ajchan4(feat_res4)
#         feat_out32 = F.interpolate(feat_out32, (H//2, W//2), mode='bilinear', align_corners=True)
#         feat_out32 = feat_out32+self.conv_out_ajchan2(feat_res2)
        feat_out32 = F.interpolate(feat_out32, (H, W), mode='bilinear', align_corners=True)
#         print("after_upsample:feat_out32:")
#         print(feat_out.size())


        if self.use_boundary_2 and self.use_boundary_4 and self.use_boundary_8:
            return feat_out, feat_out16, feat_out32, feat_out_sp2, feat_out_sp4, feat_out_sp8
        
        if (not self.use_boundary_2) and self.use_boundary_4 and self.use_boundary_8:
            return feat_out, feat_out16, feat_out32, feat_out_sp4, feat_out_sp8

        if (not self.use_boundary_2) and (not self.use_boundary_4) and self.use_boundary_8:
            return feat_out, feat_out16, feat_out32, feat_out_sp8
        
        if (not self.use_boundary_2) and (not self.use_boundary_4) and (not self.use_boundary_8):
            return feat_out, feat_out16, feat_out32

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            child_wd_params, child_nowd_params = child.get_params()
            if isinstance(child, (FeatureFusionModule, BiSeNetOutput)):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params


if __name__ == "__main__":
    
    net = BiSeNet('STDCNet813', 19)
    net.cuda()
    net.eval()
    in_ten = torch.randn(1, 3, 768, 1536).cuda()
    out, out16, out32 = net(in_ten)
    print(out.shape)
    torch.save(net.state_dict(), 'STDCNet813.pth')

    
