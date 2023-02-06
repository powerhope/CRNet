# import torch
# import torch.nn as nn
# from torch.nn import init
# import math



# class ConvX(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel=3, stride=1):
#         super(ConvX, self).__init__()
#         self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=kernel//2, bias=False)
#         self.bn = nn.BatchNorm2d(out_planes)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         out = self.relu(self.bn(self.conv(x)))
#         return out


# class AddBottleneck(nn.Module):
#     def __init__(self, in_planes, out_planes, block_num=3, stride=1):
#         super(AddBottleneck, self).__init__()
#         assert block_num > 1, print("block number should be larger than 1.")
#         self.conv_list = nn.ModuleList()
#         self.stride = stride
#         if stride == 2:
#             self.avd_layer = nn.Sequential(
#                 nn.Conv2d(out_planes//2, out_planes//2, kernel_size=3, stride=2, padding=1, groups=out_planes//2, bias=False),
#                 nn.BatchNorm2d(out_planes//2),
#             )
#             self.skip = nn.Sequential(
#                 nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=2, padding=1, groups=in_planes, bias=False),
#                 nn.BatchNorm2d(in_planes),
#                 nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
#                 nn.BatchNorm2d(out_planes),
#             )
#             stride = 1

#         for idx in range(block_num):
#             if idx == 0:
#                 self.conv_list.append(ConvX(in_planes, out_planes//2, kernel=1))
#             elif idx == 1 and block_num == 2:
#                 self.conv_list.append(ConvX(out_planes//2, out_planes//2, stride=stride))
#             elif idx == 1 and block_num > 2:
#                 self.conv_list.append(ConvX(out_planes//2, out_planes//4, stride=stride))
#             elif idx < block_num - 1:
#                 self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx+1))))
#             else:
#                 self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx))))
            
#     def forward(self, x):
#         out_list = []
#         out = x

#         for idx, conv in enumerate(self.conv_list):
#             if idx == 0 and self.stride == 2:
#                 out = self.avd_layer(conv(out))
#             else:
#                 out = conv(out)
#             out_list.append(out)

#         if self.stride == 2:
#             x = self.skip(x)

#         return torch.cat(out_list, dim=1) + x



# class CatBottleneck(nn.Module):
#     def __init__(self, in_planes, out_planes, block_num=3, stride=1):
#         super(CatBottleneck, self).__init__()
#         assert block_num > 1, print("block number should be larger than 1.")
#         self.conv_list = nn.ModuleList()
#         self.stride = stride
#         if stride == 2:
#             self.avd_layer = nn.Sequential(
#                 nn.Conv2d(out_planes//2, out_planes//2, kernel_size=3, stride=2, padding=1, groups=out_planes//2, bias=False),
#                 nn.BatchNorm2d(out_planes//2),
#             )
#             self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
#             stride = 1

#         for idx in range(block_num):
#             if idx == 0:
#                 self.conv_list.append(ConvX(in_planes, out_planes//2, kernel=1))
#             elif idx == 1 and block_num == 2:
#                 self.conv_list.append(ConvX(out_planes//2, out_planes//2, stride=stride))
#             elif idx == 1 and block_num > 2:
#                 self.conv_list.append(ConvX(out_planes//2, out_planes//4, stride=stride))
#             elif idx < block_num - 1:
#                 self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx+1))))
#             else:
#                 self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx))))
            
#     def forward(self, x):
#         out_list = []
#         out1 = self.conv_list[0](x)

#         for idx, conv in enumerate(self.conv_list[1:]):
#             if idx == 0:
#                 if self.stride == 2:
#                     out = conv(self.avd_layer(out1))
#                 else:
#                     out = conv(out1)
#             else:
#                 out = conv(out)
#             out_list.append(out)

#         if self.stride == 2:
#             out1 = self.skip(out1)
#         out_list.insert(0, out1)

#         out = torch.cat(out_list, dim=1)
#         return out

# #STDC2Net
# class STDCNet1446(nn.Module):
#     def __init__(self, base=64, layers=[4,5,3], block_num=4, type="cat", num_classes=1000, dropout=0.20, pretrain_model='', use_conv_last=False):
#         super(STDCNet1446, self).__init__()
#         if type == "cat":
#             block = CatBottleneck
#         elif type == "add":
#             block = AddBottleneck
#         self.use_conv_last = use_conv_last
#         self.features = self._make_layers(base, layers, block_num, block)
#         self.conv_last = ConvX(base*16, max(1024, base*16), 1, 1)
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(max(1024, base*16), max(1024, base*16), bias=False)
#         self.bn = nn.BatchNorm1d(max(1024, base*16))
#         self.relu = nn.ReLU(inplace=True)
#         self.dropout = nn.Dropout(p=dropout)
#         self.linear = nn.Linear(max(1024, base*16), num_classes, bias=False)

#         self.x2 = nn.Sequential(self.features[:1])
#         self.x4 = nn.Sequential(self.features[1:2])
#         self.x8 = nn.Sequential(self.features[2:6])
#         self.x16 = nn.Sequential(self.features[6:11])
#         self.x32 = nn.Sequential(self.features[11:])

#         if pretrain_model:
#             print('use pretrain model {}'.format(pretrain_model))
#             self.init_weight(pretrain_model)
#         else:
#             self.init_params()

#     def init_weight(self, pretrain_model):
        
#         state_dict = torch.load(pretrain_model)["state_dict"]
#         self_state_dict = self.state_dict()
#         for k, v in state_dict.items():
#             self_state_dict.update({k: v})
#         self.load_state_dict(self_state_dict)

#     def init_params(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)

#     def _make_layers(self, base, layers, block_num, block):
#         features = []
#         features += [ConvX(3, base//2, 3, 2)]
#         features += [ConvX(base//2, base, 3, 2)]

#         for i, layer in enumerate(layers):
#             for j in range(layer):
#                 if i == 0 and j == 0:
#                     features.append(block(base, base*4, block_num, 2))
#                 elif j == 0:
#                     features.append(block(base*int(math.pow(2,i+1)), base*int(math.pow(2,i+2)), block_num, 2))
#                 else:
#                     features.append(block(base*int(math.pow(2,i+2)), base*int(math.pow(2,i+2)), block_num, 1))

#         return nn.Sequential(*features)

#     def forward(self, x):
#         feat2 = self.x2(x)
#         feat4 = self.x4(feat2)
#         feat8 = self.x8(feat4)
#         feat16 = self.x16(feat8)
#         feat32 = self.x32(feat16)
#         if self.use_conv_last:
#            feat32 = self.conv_last(feat32)

#         return feat2, feat4, feat8, feat16, feat32

#     def forward_impl(self, x):
#         out = self.features(x)
#         out = self.conv_last(out).pow(2)
#         out = self.gap(out).flatten(1)
#         out = self.fc(out)
#         # out = self.bn(out)
#         out = self.relu(out)
#         # out = self.relu(self.bn(self.fc(out)))
#         out = self.dropout(out)
#         out = self.linear(out)
#         return out

# # STDC1Net
# class STDCNet813(nn.Module):
#     def __init__(self, base=64, layers=[2,2,2], block_num=4, type="cat", num_classes=1000, dropout=0.20, pretrain_model='', use_conv_last=False):
#         super(STDCNet813, self).__init__()
#         if type == "cat":
#             block = CatBottleneck
#         elif type == "add":
#             block = AddBottleneck
#         self.use_conv_last = use_conv_last
#         self.features = self._make_layers(base, layers, block_num, block)
#         self.conv_last = ConvX(base*16, max(1024, base*16), 1, 1)
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(max(1024, base*16), max(1024, base*16), bias=False)
#         self.bn = nn.BatchNorm1d(max(1024, base*16))
#         self.relu = nn.ReLU(inplace=True)
#         self.dropout = nn.Dropout(p=dropout)
#         self.linear = nn.Linear(max(1024, base*16), num_classes, bias=False)

#         self.x2 = nn.Sequential(self.features[:1])
#         self.x4 = nn.Sequential(self.features[1:2])
#         self.x8 = nn.Sequential(self.features[2:4])
#         self.x16 = nn.Sequential(self.features[4:6])
#         self.x32 = nn.Sequential(self.features[6:])

#         if pretrain_model:
#             print('use pretrain model {}'.format(pretrain_model))
#             self.init_weight(pretrain_model)
#         else:
#             self.init_params()

#     def init_weight(self, pretrain_model):
        
#         state_dict = torch.load(pretrain_model)["state_dict"]
#         self_state_dict = self.state_dict()
#         for k, v in state_dict.items():
#             self_state_dict.update({k: v})
#         self.load_state_dict(self_state_dict)

#     def init_params(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)

#     def _make_layers(self, base, layers, block_num, block):
#         features = []
#         features += [ConvX(3, base//2, 3, 2)]
#         features += [ConvX(base//2, base, 3, 2)]

#         for i, layer in enumerate(layers):
#             for j in range(layer):
#                 if i == 0 and j == 0:
#                     features.append(block(base, base*4, block_num, 2))
#                 elif j == 0:
#                     features.append(block(base*int(math.pow(2,i+1)), base*int(math.pow(2,i+2)), block_num, 2))
#                 else:
#                     features.append(block(base*int(math.pow(2,i+2)), base*int(math.pow(2,i+2)), block_num, 1))

#         return nn.Sequential(*features)

#     def forward(self, x):
#         feat2 = self.x2(x)
#         feat4 = self.x4(feat2)
#         feat8 = self.x8(feat4)
#         feat16 = self.x16(feat8)
#         feat32 = self.x32(feat16)
#         if self.use_conv_last:
#            feat32 = self.conv_last(feat32)

#         return feat2, feat4, feat8, feat16, feat32

#     def forward_impl(self, x):
#         out = self.features(x)
#         out = self.conv_last(out).pow(2)
#         out = self.gap(out).flatten(1)
#         out = self.fc(out)
#         # out = self.bn(out)
#         out = self.relu(out)
#         # out = self.relu(self.bn(self.fc(out)))
#         out = self.dropout(out)
#         out = self.linear(out)
#         return out

# if __name__ == "__main__":
#     model = STDCNet813(num_classes=1000, dropout=0.00, block_num=4)
#     model.eval()
#     x = torch.randn(1,3,224,224)
#     y = model(x)
#     torch.save(model.state_dict(), 'cat.pth')
#     print(y.size())
##########################################################################################################修改的第一次结构
# import torch
# import torch.nn as nn
# from torch.nn import init
# import math
# BatchNorm2d = nn.BatchNorm2d

# # class depthwise_separable_conv(nn.Module):
# #     def init(self, nin, nout):
# #         super(depthwise_separable_conv, self).init()
# #         self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin)
# #         self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)
# #     def forward(self, x):
# #         out = self.depthwise(x)
# #         out = self.pointwise(out)
# #         return out
# # class ConvBNReLU(nn.Module):
# #     def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
# #         super(ConvBNReLU, self).__init__()
# #         self.conv = nn.Conv2d(in_chan,
# #                 out_chan,
# #                 kernel_size = ks,
# #                 stride = stride,
# #                 padding = padding,
# #                 bias = False)
# #         # self.bn = BatchNorm2d(out_chan)
# #         self.bn = BatchNorm2d(out_chan)
# #         self.relu = nn.ReLU()
# # #         self.init_weight()

# #     def forward(self, x):
# #         x = self.conv(x)
# #         x = self.bn(x)
# #         x = self.relu(x)
# #         return x
# class ConvX(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel=3, stride=1):
#         super(ConvX, self).__init__()
#         self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=kernel//2, bias=False)
# #         self.depthwise = nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1, groups=in_planes)
# #         self.pointwise = nn.Conv2d(in_planes, out_planes, kernel_size=1)
#         self.bn = nn.BatchNorm2d(out_planes)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         out = self.relu(self.bn(self.conv(x)))
# #         out = self.relu(self.bn(self.pointwise(self.depthwise(x))))
#         return out


# class AddBottleneck(nn.Module):
#     def __init__(self, in_planes, out_planes, block_num=3, stride=1):
#         super(AddBottleneck, self).__init__()
#         assert block_num > 1, print("block number should be larger than 1.")
#         self.conv_list = nn.ModuleList()
#         self.stride = stride
#         if stride == 2:
#             self.avd_layer = nn.Sequential(
#                 nn.Conv2d(out_planes//2, out_planes//2, kernel_size=3, stride=2, padding=1, groups=out_planes//2, bias=False),
#                 nn.BatchNorm2d(out_planes//2),
#             )
#             self.skip = nn.Sequential(
#                 nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=2, padding=1, groups=in_planes, bias=False),
#                 nn.BatchNorm2d(in_planes),
#                 nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
#                 nn.BatchNorm2d(out_planes),
#             )
#             stride = 1

#         for idx in range(block_num):
#             if idx == 0:
#                 self.conv_list.append(ConvX(in_planes, out_planes//2, kernel=1))
#             elif idx == 1 and block_num == 2:
#                 self.conv_list.append(ConvX(out_planes//2, out_planes//2, stride=stride))
#             elif idx == 1 and block_num > 2:
#                 self.conv_list.append(ConvX(out_planes//2, out_planes//4, stride=stride))
#             elif idx < block_num - 1:
#                 self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx+1))))
#             else:
#                 self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx))))
            
#     def forward(self, x):
#         out_list = []
#         out = x

#         for idx, conv in enumerate(self.conv_list):
#             if idx == 0 and self.stride == 2:
#                 out = self.avd_layer(conv(out))
#             else:
#                 out = conv(out)
#             out_list.append(out)

#         if self.stride == 2:
#             x = self.skip(x)

#         return torch.cat(out_list, dim=1) + x



# class CatBottleneck(nn.Module):
#     def __init__(self, in_planes, out_planes, block_num=3, stride=1):
#         super(CatBottleneck, self).__init__()
#         assert block_num > 1, print("block number should be larger than 1.")
#         self.conv_list = nn.ModuleList()
#         self.stride = stride
#         if stride == 2:
#             self.avd_layer = nn.Sequential(
#                 nn.Conv2d(out_planes//2, out_planes//2, kernel_size=3, stride=2, padding=1, groups=out_planes//2, bias=False),
#                 nn.BatchNorm2d(out_planes//2),
#             )
#             self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
#             stride = 1

#         for idx in range(block_num):
#             if idx == 0:
#                 self.conv_list.append(ConvX(in_planes, out_planes//2, kernel=1))
#             elif idx == 1 and block_num == 2:
#                 self.conv_list.append(ConvX(out_planes//2, out_planes//2, stride=stride))
#             elif idx == 1 and block_num > 2:
#                 self.conv_list.append(ConvX(out_planes//2, out_planes//4, stride=stride))
#             elif idx < block_num - 1:
#                 self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx+1))))
#             else:
#                 self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx))))
            
#     def forward(self, x):
#         out_list = []
#         out1 = self.conv_list[0](x)

#         for idx, conv in enumerate(self.conv_list[1:]):
#             if idx == 0:
#                 if self.stride == 2:
#                     out = conv(self.avd_layer(out1))
#                 else:
#                     out = conv(out1)
#             else:
#                 out = conv(out)
#             out_list.append(out)

#         if self.stride == 2:
#             out1 = self.skip(out1)
#         out_list.insert(0, out1)

#         out = torch.cat(out_list, dim=1)
#         return out

# #STDC2Net
# class STDCNet1446(nn.Module):
#     def __init__(self, base=64, layers=[4,5,3], block_num=4, type="cat", num_classes=1000, dropout=0.20, pretrain_model='', use_conv_last=False):
#         super(STDCNet1446, self).__init__()
#         if type == "cat":
#             block = CatBottleneck
#         elif type == "add":
#             block = AddBottleneck
#         self.use_conv_last = use_conv_last
#         self.features = self._make_layers(base, layers, block_num, block)
#         self.conv_last = ConvX(base*16, max(1024, base*16), 1, 1)
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(max(1024, base*16), max(1024, base*16), bias=False)
#         self.bn = nn.BatchNorm1d(max(1024, base*16))
#         self.relu = nn.ReLU(inplace=True)
#         self.dropout = nn.Dropout(p=dropout)
#         self.linear = nn.Linear(max(1024, base*16), num_classes, bias=False)
#         self.trans4 = ConvX(base,base*2,3,2)
#         self.trans8 = ConvX(base*2,base*4,3,2)
#         self.trans16 = ConvX(base*4,base*8,3,2)
# #         self.concatfuse8 = ConvX(base*4, base*4)
# #         self.concatfuse16 = ConvX(base*8, base*8)
# #         self.concatfuse32 = ConvX(base*16, base*16)

#         self.x2 = nn.Sequential(self.features[:1])
#         self.x4 = nn.Sequential(self.features[1:2])
#         self.x8 = nn.Sequential(self.features[2:6])
#         self.x16 = nn.Sequential(self.features[6:11])
#         self.x32 = nn.Sequential(self.features[11:])

#         if pretrain_model:
# #             print('use pretrain model {}'.format(pretrain_model))
# #             self.init_weight(pretrain_model)
# #         else:
#             self.init_params()

#     def init_weight(self, pretrain_model):
        
#         state_dict = torch.load(pretrain_model)["state_dict"]
#         self_state_dict = self.state_dict()
#         for k, v in state_dict.items():
#             self_state_dict.update({k: v})
#         self.load_state_dict(self_state_dict)

#     def init_params(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)

#     def _make_layers(self, base, layers, block_num, block):
#         features = []
#         features += [ConvX(3, base//2, 3, 2)]
#         features += [ConvX(base//2, base, 3, 2)]

#         for i, layer in enumerate(layers):
#             for j in range(layer):
#                 if i == 0 and j == 0:
#                     features.append(block(base, base*2, block_num, 2))
#                 elif j == 0:
#                     features.append(block(base*int(math.pow(2,i)), base*int(math.pow(2,i+1)), block_num, 2))
#                 else:
#                     features.append(block(base*int(math.pow(2,i+1)), base*int(math.pow(2,i+1)), block_num, 1))

#         return nn.Sequential(*features)

#     def forward(self, x):
#         feat2 = self.x2(x)
#         feat4 = self.x4(feat2)
#         print("feat4:")
#         print(feat4.size())
        
#         trans4 = self.trans4(feat4)
        
#         feat8_temp = self.x8(feat4)
#         feat8 = torch.cat((trans4,feat8_temp),dim=1)
# #         feat8 = self.concatfuse8(feat8)
#         print("feat8:")
#         print(feat8.size())
        
#         trans8 = self.trans8(feat8_temp)
        
#         feat16_temp = self.x16(feat8_temp)
#         feat16 = torch.cat((trans8,feat16_temp),dim=1)
# #         feat16 = self.concatfuse16(feat16)
#         print("feat16:")
#         print(feat16.size())
        
#         trans16 = self.trans16(feat16_temp)
        
#         feat32_temp = self.x32(feat16_temp)
#         feat32 = torch.cat((trans16,feat32_temp),dim=1)
# #         feat32 = self.concatfuse32(feat32)
#         print("feat32:")
#         print(feat32.size())
        
#         if self.use_conv_last:
#            feat32 = self.conv_last(feat32)

#         return feat2, feat4, feat8, feat16, feat32

#     def forward_impl(self, x):
#         out = self.features(x)
#         out = self.conv_last(out).pow(2)
#         out = self.gap(out).flatten(1)
#         out = self.fc(out)
#         # out = self.bn(out)
#         out = self.relu(out)
#         # out = self.relu(self.bn(self.fc(out)))
#         out = self.dropout(out)
#         out = self.linear(out)
#         return out

# # STDC1Net
# class STDCNet813(nn.Module):
#     def __init__(self, base=64, layers=[2,2,2], block_num=4, type="cat", num_classes=1000, dropout=0.20, pretrain_model='', use_conv_last=False):
#         super(STDCNet813, self).__init__()
#         if type == "cat":
#             block = CatBottleneck
#         elif type == "add":
#             block = AddBottleneck
#         self.use_conv_last = use_conv_last
#         self.features = self._make_layers(base, layers, block_num, block)
#         self.conv_last = ConvX(base*16, max(1024, base*16), 1, 1)
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(max(1024, base*16), max(1024, base*16), bias=False)
#         self.bn = nn.BatchNorm1d(max(1024, base*16))
#         self.relu = nn.ReLU(inplace=True)
#         self.dropout = nn.Dropout(p=dropout)
#         self.linear = nn.Linear(max(1024, base*16), num_classes, bias=False)

#         self.x2 = nn.Sequential(self.features[:1])
#         self.x4 = nn.Sequential(self.features[1:2])
#         self.x8 = nn.Sequential(self.features[2:4])
#         self.x16 = nn.Sequential(self.features[4:6])
#         self.x32 = nn.Sequential(self.features[6:])

#         if pretrain_model:
#             print('use pretrain model {}'.format(pretrain_model))
#             self.init_weight(pretrain_model)
#         else:
#             self.init_params()

#     def init_weight(self, pretrain_model):
        
#         state_dict = torch.load(pretrain_model)["state_dict"]
#         self_state_dict = self.state_dict()
#         for k, v in state_dict.items():
#             self_state_dict.update({k: v})
#         self.load_state_dict(self_state_dict)

#     def init_params(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)

#     def _make_layers(self, base, layers, block_num, block):
#         features = []
#         features += [ConvX(3, base//2, 3, 2)]
#         features += [ConvX(base//2, base, 3, 2)]

#         for i, layer in enumerate(layers):
#             for j in range(layer):
#                 if i == 0 and j == 0:
#                     features.append(block(base, base*4, block_num, 2))
#                 elif j == 0:
#                     features.append(block(base*int(math.pow(2,i+1)), base*int(math.pow(2,i+2)), block_num, 2))
#                 else:
#                     features.append(block(base*int(math.pow(2,i+2)), base*int(math.pow(2,i+2)), block_num, 1))

#         return nn.Sequential(*features)

#     def forward(self, x):
#         feat2 = self.x2(x)
#         feat4 = self.x4(feat2)
#         feat8 = self.x8(feat4)
#         feat16 = self.x16(feat8)
#         feat32 = self.x32(feat16)
#         if self.use_conv_last:
#            feat32 = self.conv_last(feat32)

#         return feat2, feat4, feat8, feat16, feat32

#     def forward_impl(self, x):
#         out = self.features(x)
#         out = self.conv_last(out).pow(2)
#         out = self.gap(out).flatten(1)
#         out = self.fc(out)
#         # out = self.bn(out)
#         out = self.relu(out)
#         # out = self.relu(self.bn(self.fc(out)))
#         out = self.dropout(out)
#         out = self.linear(out)
#         return out

# if __name__ == "__main__":
#     model = STDCNet813(num_classes=1000, dropout=0.00, block_num=4)
#     model.eval()
#     x = torch.randn(1,3,224,224)
#     y = model(x)
#     torch.save(model.state_dict(), 'cat.pth')
#     print(y.size())
############################################################################################################第三次实验。加入channalattention
# import torch
# import torch.nn as nn
# from torch.nn import init
# import math
# BatchNorm2d = nn.BatchNorm2d

# # class depthwise_separable_conv(nn.Module):
# #     def init(self, nin, nout):
# #         super(depthwise_separable_conv, self).init()
# #         self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin)
# #         self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)
# #     def forward(self, x):
# #         out = self.depthwise(x)
# #         out = self.pointwise(out)
# #         return out
# # class ConvBNReLU(nn.Module):
# #     def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
# #         super(ConvBNReLU, self).__init__()
# #         self.conv = nn.Conv2d(in_chan,
# #                 out_chan,
# #                 kernel_size = ks,
# #                 stride = stride,
# #                 padding = padding,
# #                 bias = False)
# #         # self.bn = BatchNorm2d(out_chan)
# #         self.bn = BatchNorm2d(out_chan)
# #         self.relu = nn.ReLU()
# # #         self.init_weight()

# #     def forward(self, x):
# #         x = self.conv(x)
# #         x = self.bn(x)
# #         x = self.relu(x)
# #         return x
# class ConvX(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel=3, stride=1):
#         super(ConvX, self).__init__()
#         self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=kernel//2, bias=False)
# #         self.depthwise = nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1, groups=in_planes)
# #         self.pointwise = nn.Conv2d(in_planes, out_planes, kernel_size=1)
#         self.bn = nn.BatchNorm2d(out_planes)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         out = self.relu(self.bn(self.conv(x)))
# #         out = self.relu(self.bn(self.pointwise(self.depthwise(x))))
#         return out


# class AddBottleneck(nn.Module):
#     def __init__(self, in_planes, out_planes, block_num=3, stride=1):
#         super(AddBottleneck, self).__init__()
#         assert block_num > 1, print("block number should be larger than 1.")
#         self.conv_list = nn.ModuleList()
#         self.stride = stride
#         if stride == 2:
#             self.avd_layer = nn.Sequential(
#                 nn.Conv2d(out_planes//2, out_planes//2, kernel_size=3, stride=2, padding=1, groups=out_planes//2, bias=False),
#                 nn.BatchNorm2d(out_planes//2),
#             )
#             self.skip = nn.Sequential(
#                 nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=2, padding=1, groups=in_planes, bias=False),
#                 nn.BatchNorm2d(in_planes),
#                 nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
#                 nn.BatchNorm2d(out_planes),
#             )
#             stride = 1

#         for idx in range(block_num):
#             if idx == 0:
#                 self.conv_list.append(ConvX(in_planes, out_planes//2, kernel=1))
#             elif idx == 1 and block_num == 2:
#                 self.conv_list.append(ConvX(out_planes//2, out_planes//2, stride=stride))
#             elif idx == 1 and block_num > 2:
#                 self.conv_list.append(ConvX(out_planes//2, out_planes//4, stride=stride))
#             elif idx < block_num - 1:
#                 self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx+1))))
#             else:
#                 self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx))))
            
#     def forward(self, x):
#         out_list = []
#         out = x

#         for idx, conv in enumerate(self.conv_list):
#             if idx == 0 and self.stride == 2:
#                 out = self.avd_layer(conv(out))
#             else:
#                 out = conv(out)
#             out_list.append(out)

#         if self.stride == 2:
#             x = self.skip(x)

#         return torch.cat(out_list, dim=1) + x

# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
           
#         self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
#                                nn.ReLU(),
#                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.fc(self.avg_pool(x))
#         max_out = self.fc(self.max_pool(x))
#         out = avg_out + max_out
#         return self.sigmoid(out)

# class CatBottleneck(nn.Module):
#     def __init__(self, in_planes, out_planes, block_num=3, stride=1):
#         super(CatBottleneck, self).__init__()
#         assert block_num > 1, print("block number should be larger than 1.")
#         self.conv_list = nn.ModuleList()
#         self.stride = stride
# #         self.atenchannal2 = ChannelAttention()
#         if stride == 2:
#             self.avd_layer = nn.Sequential(
#                 nn.Conv2d(out_planes//2, out_planes//2, kernel_size=3, stride=2, padding=1, groups=out_planes//2, bias=False),
#                 nn.BatchNorm2d(out_planes//2),
#             )
#             self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
#             stride = 1

#         for idx in range(block_num):
#             if idx == 0:
#                 self.conv_list.append(ConvX(in_planes, out_planes//2, kernel=1))
#                 self.conv_list.append(ChannelAttention(out_planes//2))
#             elif idx == 1 and block_num == 2:
#                 self.conv_list.append(ConvX(out_planes//2, out_planes//2, stride=stride))
#                 self.conv_list.append(ChannelAttention(out_planes//2))
#             elif idx == 1 and block_num > 2:
#                 self.conv_list.append(ConvX(out_planes//2, out_planes//4, stride=stride))
#                 self.conv_list.append(ChannelAttention(out_planes//4))
#             elif idx < block_num - 1:
#                 self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx+1))))
#                 self.conv_list.append(ChannelAttention(out_planes//int(math.pow(2, idx+1))))
#             else:
#                 self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx))))
#                 self.conv_list.append(ChannelAttention(out_planes//int(math.pow(2, idx))))
                                      
            
#     def forward(self, x):
#         out_list = []
#         out1 = self.conv_list[0](x)
#         out1 = (self.conv_list[1](out1))*out1
# #         print("out1:")
# #         print(out1.size())

#         for idx, conv in enumerate(self.conv_list[2::2]):
#             if idx == 0:
#                 if self.stride == 2:
#                     out = conv(self.avd_layer(out1))
# #                     print("out2:")
# #                     print(out.size())
#                     out = (self.conv_list[2*idx+3](out))*out
# #                     print("out2:")
# #                     print(out.size())
#                 else:
#                     out = conv(out1)
#                     out = (self.conv_list[2*idx+3](out))*out
# #                     print("out2:")
# #                     print(out.size())
#             else:
#                 out = conv(out)
#                 out = (self.conv_list[2*idx+3](out))*out
# #                 print("out:")
# #                 print(out.size())
#             out_list.append(out)

#         if self.stride == 2:
#             out1 = self.skip(out1)
#         out_list.insert(0, out1)

#         out = torch.cat(out_list, dim=1)
#         return out

# #STDC2Net
# class STDCNet1446(nn.Module):
#     def __init__(self, base=64, layers=[4,5,3], block_num=4, type="cat", num_classes=1000, dropout=0.20, pretrain_model='', use_conv_last=False):
#         super(STDCNet1446, self).__init__()
#         if type == "cat":
#             block = CatBottleneck
#         elif type == "add":
#             block = AddBottleneck
#         self.use_conv_last = use_conv_last
#         self.features = self._make_layers(base, layers, block_num, block)
#         self.conv_last = ConvX(base*16, max(1024, base*16), 1, 1)
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(max(1024, base*16), max(1024, base*16), bias=False)
#         self.bn = nn.BatchNorm1d(max(1024, base*16))
#         self.relu = nn.ReLU(inplace=True)
#         self.dropout = nn.Dropout(p=dropout)
#         self.linear = nn.Linear(max(1024, base*16), num_classes, bias=False)
#         self.trans4 = ConvX(base,base*2,3,2)
#         self.trans8 = ConvX(base*2,base*4,3,2)
#         self.trans16 = ConvX(base*4,base*8,3,2)
# #         self.concatfuse8 = ConvX(base*4, base*4)
# #         self.concatfuse16 = ConvX(base*8, base*8)
# #         self.concatfuse32 = ConvX(base*16, base*16)

#         self.x2 = nn.Sequential(self.features[:1])
#         self.x4 = nn.Sequential(self.features[1:2])
#         self.x8 = nn.Sequential(self.features[2:6])
#         self.x16 = nn.Sequential(self.features[6:11])
#         self.x32 = nn.Sequential(self.features[11:])

#         if pretrain_model:
# #             print('use pretrain model {}'.format(pretrain_model))
# #             self.init_weight(pretrain_model)
# #         else:
#             self.init_params()

#     def init_weight(self, pretrain_model):
        
#         state_dict = torch.load(pretrain_model)["state_dict"]
#         self_state_dict = self.state_dict()
#         for k, v in state_dict.items():
#             self_state_dict.update({k: v})
#         self.load_state_dict(self_state_dict)

#     def init_params(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)

#     def _make_layers(self, base, layers, block_num, block):
#         features = []
#         features += [ConvX(3, base//2, 3, 2)]
#         features += [ConvX(base//2, base, 3, 2)]

#         for i, layer in enumerate(layers):
#             for j in range(layer):
#                 if i == 0 and j == 0:
#                     features.append(block(base, base*2, block_num, 2))
#                 elif j == 0:
#                     features.append(block(base*int(math.pow(2,i)), base*int(math.pow(2,i+1)), block_num, 2))
#                 else:
#                     features.append(block(base*int(math.pow(2,i+1)), base*int(math.pow(2,i+1)), block_num, 1))

#         return nn.Sequential(*features)

#     def forward(self, x):
#         feat2 = self.x2(x)
#         feat4 = self.x4(feat2)
# #         print("feat4:")
# #         print(feat4.size())
        
#         trans4 = self.trans4(feat4)
        
#         feat8_temp = self.x8(feat4)
#         feat8 = torch.cat((trans4,feat8_temp),dim=1)
# #         feat8 = self.concatfuse8(feat8)
# #         print("feat8:")
# #         print(feat8.size())
        
#         trans8 = self.trans8(feat8_temp)
        
#         feat16_temp = self.x16(feat8_temp)
#         feat16 = torch.cat((trans8,feat16_temp),dim=1)
# #         feat16 = self.concatfuse16(feat16)
# #         print("feat16:")
# #         print(feat16.size())
        
#         trans16 = self.trans16(feat16_temp)
        
#         feat32_temp = self.x32(feat16_temp)
#         feat32 = torch.cat((trans16,feat32_temp),dim=1)
# #         feat32 = self.concatfuse32(feat32)
# #         print("feat32:")
# #         print(feat32.size())
        
#         if self.use_conv_last:
#            feat32 = self.conv_last(feat32)

#         return feat2, feat4, feat8, feat16, feat32

#     def forward_impl(self, x):
#         out = self.features(x)
#         out = self.conv_last(out).pow(2)
#         out = self.gap(out).flatten(1)
#         out = self.fc(out)
#         # out = self.bn(out)
#         out = self.relu(out)
#         # out = self.relu(self.bn(self.fc(out)))
#         out = self.dropout(out)
#         out = self.linear(out)
#         return out

# # STDC1Net
# class STDCNet813(nn.Module):
#     def __init__(self, base=64, layers=[2,2,2], block_num=4, type="cat", num_classes=1000, dropout=0.20, pretrain_model='', use_conv_last=False):
#         super(STDCNet813, self).__init__()
#         if type == "cat":
#             block = CatBottleneck
#         elif type == "add":
#             block = AddBottleneck
#         self.use_conv_last = use_conv_last
#         self.features = self._make_layers(base, layers, block_num, block)
#         self.conv_last = ConvX(base*16, max(1024, base*16), 1, 1)
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(max(1024, base*16), max(1024, base*16), bias=False)
#         self.bn = nn.BatchNorm1d(max(1024, base*16))
#         self.relu = nn.ReLU(inplace=True)
#         self.dropout = nn.Dropout(p=dropout)
#         self.linear = nn.Linear(max(1024, base*16), num_classes, bias=False)

#         self.x2 = nn.Sequential(self.features[:1])
#         self.x4 = nn.Sequential(self.features[1:2])
#         self.x8 = nn.Sequential(self.features[2:4])
#         self.x16 = nn.Sequential(self.features[4:6])
#         self.x32 = nn.Sequential(self.features[6:])

#         if pretrain_model:
#             print('use pretrain model {}'.format(pretrain_model))
#             self.init_weight(pretrain_model)
#         else:
#             self.init_params()

#     def init_weight(self, pretrain_model):
        
#         state_dict = torch.load(pretrain_model)["state_dict"]
#         self_state_dict = self.state_dict()
#         for k, v in state_dict.items():
#             self_state_dict.update({k: v})
#         self.load_state_dict(self_state_dict)

#     def init_params(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)

#     def _make_layers(self, base, layers, block_num, block):
#         features = []
#         features += [ConvX(3, base//2, 3, 2)]
#         features += [ConvX(base//2, base, 3, 2)]

#         for i, layer in enumerate(layers):
#             for j in range(layer):
#                 if i == 0 and j == 0:
#                     features.append(block(base, base*4, block_num, 2))
#                 elif j == 0:
#                     features.append(block(base*int(math.pow(2,i+1)), base*int(math.pow(2,i+2)), block_num, 2))
#                 else:
#                     features.append(block(base*int(math.pow(2,i+2)), base*int(math.pow(2,i+2)), block_num, 1))

#         return nn.Sequential(*features)

#     def forward(self, x):
#         feat2 = self.x2(x)
#         feat4 = self.x4(feat2)
#         feat8 = self.x8(feat4)
#         feat16 = self.x16(feat8)
#         feat32 = self.x32(feat16)
#         if self.use_conv_last:
#            feat32 = self.conv_last(feat32)

#         return feat2, feat4, feat8, feat16, feat32

#     def forward_impl(self, x):
#         out = self.features(x)
#         out = self.conv_last(out).pow(2)
#         out = self.gap(out).flatten(1)
#         out = self.fc(out)
#         # out = self.bn(out)
#         out = self.relu(out)
#         # out = self.relu(self.bn(self.fc(out)))
#         out = self.dropout(out)
#         out = self.linear(out)
#         return out

# if __name__ == "__main__":
#     model = STDCNet813(num_classes=1000, dropout=0.00, block_num=4)
#     model.eval()
#     x = torch.randn(1,3,224,224)
#     y = model(x)
#     torch.save(model.state_dict(), 'cat.pth')
#     print(y.size())
#######################################################################################第四次实验
# import torch
# import torch.nn as nn
# from torch.nn import init
# import math
# BatchNorm2d = nn.BatchNorm2d

# # class depthwise_separable_conv(nn.Module):
# #     def init(self, nin, nout):
# #         super(depthwise_separable_conv, self).init()
# #         self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin)
# #         self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)
# #     def forward(self, x):
# #         out = self.depthwise(x)
# #         out = self.pointwise(out)
# #         return out
# # class ConvBNReLU(nn.Module):
# #     def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
# #         super(ConvBNReLU, self).__init__()
# #         self.conv = nn.Conv2d(in_chan,
# #                 out_chan,
# #                 kernel_size = ks,
# #                 stride = stride,
# #                 padding = padding,
# #                 bias = False)
# #         # self.bn = BatchNorm2d(out_chan)
# #         self.bn = BatchNorm2d(out_chan)
# #         self.relu = nn.ReLU()
# # #         self.init_weight()

# #     def forward(self, x):
# #         x = self.conv(x)
# #         x = self.bn(x)
# #         x = self.relu(x)
# #         return x
# class ConvX(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel=3, stride=1):
#         super(ConvX, self).__init__()
#         self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=kernel//2, bias=False)
# #         self.depthwise = nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1, groups=in_planes)
# #         self.pointwise = nn.Conv2d(in_planes, out_planes, kernel_size=1)
#         self.bn = nn.BatchNorm2d(out_planes)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         out = self.relu(self.bn(self.conv(x)))
# #         out = self.relu(self.bn(self.pointwise(self.depthwise(x))))
#         return out


# class AddBottleneck(nn.Module):
#     def __init__(self, in_planes, out_planes, block_num=3, stride=1):
#         super(AddBottleneck, self).__init__()
#         assert block_num > 1, print("block number should be larger than 1.")
#         self.conv_list = nn.ModuleList()
#         self.stride = stride
#         if stride == 2:
#             self.avd_layer = nn.Sequential(
#                 nn.Conv2d(out_planes//2, out_planes//2, kernel_size=3, stride=2, padding=1, groups=out_planes//2, bias=False),
#                 nn.BatchNorm2d(out_planes//2),
#             )
#             self.skip = nn.Sequential(
#                 nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=2, padding=1, groups=in_planes, bias=False),
#                 nn.BatchNorm2d(in_planes),
#                 nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
#                 nn.BatchNorm2d(out_planes),
#             )
#             stride = 1

#         for idx in range(block_num):
#             if idx == 0:
#                 self.conv_list.append(ConvX(in_planes, out_planes//2, kernel=1))
#             elif idx == 1 and block_num == 2:
#                 self.conv_list.append(ConvX(out_planes//2, out_planes//2, stride=stride))
#             elif idx == 1 and block_num > 2:
#                 self.conv_list.append(ConvX(out_planes//2, out_planes//4, stride=stride))
#             elif idx < block_num - 1:
#                 self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx+1))))
#             else:
#                 self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx))))
            
#     def forward(self, x):
#         out_list = []
#         out = x

#         for idx, conv in enumerate(self.conv_list):
#             if idx == 0 and self.stride == 2:
#                 out = self.avd_layer(conv(out))
#             else:
#                 out = conv(out)
#             out_list.append(out)

#         if self.stride == 2:
#             x = self.skip(x)

#         return torch.cat(out_list, dim=1) + x

# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
           
#         self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
#                                nn.ReLU(),
#                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.fc(self.avg_pool(x))
#         max_out = self.fc(self.max_pool(x))
#         out = avg_out + max_out
#         return self.sigmoid(out)

# class CatBottleneck(nn.Module):
#     def __init__(self, in_planes, out_planes, block_num=3, stride=1):
#         super(CatBottleneck, self).__init__()
#         assert block_num > 1, print("block number should be larger than 1.")
#         self.conv_list = nn.ModuleList()
#         self.stride = stride
# #         self.atenchannal2 = ChannelAttention()
#         if stride == 2:
#             self.avd_layer = nn.Sequential(
#                 nn.Conv2d(out_planes//2, out_planes//2, kernel_size=3, stride=2, padding=1, groups=out_planes//2, bias=False),
#                 nn.BatchNorm2d(out_planes//2),
#             )
#             self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
#             stride = 1

#         for idx in range(block_num):
#             if idx == 0:
#                 self.conv_list.append(ConvX(in_planes, out_planes//2, kernel=1))
#                 self.conv_list.append(ChannelAttention(out_planes//2))
#             elif idx == 1 and block_num == 2:
#                 self.conv_list.append(ConvX(out_planes//2, out_planes//2, stride=stride))
# #                 self.conv_list.append(ChannelAttention(out_planes//2))
#             elif idx == 1 and block_num > 2:
#                 self.conv_list.append(ConvX(out_planes//2, out_planes//4, stride=stride))
# #                 self.conv_list.append(ChannelAttention(out_planes//4))
#             elif idx < block_num - 1:
#                 self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx+1))))
# #                 self.conv_list.append(ChannelAttention(out_planes//int(math.pow(2, idx+1))))
#             else:
#                 self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx))))
# #                 self.conv_list.append(ChannelAttention(out_planes//int(math.pow(2, idx))))
                                      
            
#     def forward(self, x):
#         out_list = []
#         out1 = self.conv_list[0](x)
#         out1 = (self.conv_list[1](out1))*out1
# #         print("out1:")
# #         print(out1.size())

#         for idx, conv in enumerate(self.conv_list[2:]):
#             if idx == 0:
#                 if self.stride == 2:
#                     out = conv(self.avd_layer(out1))
# #                     print("out2:")
# #                     print(out.size())
# #                     out = (self.conv_list[2*idx+3](out))*out
# #                     print("out2:")
# #                     print(out.size())
#                 else:
#                     out = conv(out1)
# #                     out = (self.conv_list[2*idx+3](out))*out
# #                     print("out2:")
# #                     print(out.size())
#             else:
#                 out = conv(out)
# #                 out = (self.conv_list[2*idx+3](out))*out
# #                 print("out:")
# #                 print(out.size())
#             out_list.append(out)

#         if self.stride == 2:
#             out1 = self.skip(out1)
#         out_list.insert(0, out1)

#         out = torch.cat(out_list, dim=1)
#         return out

# #STDC2Net
# class STDCNet1446(nn.Module):
#     def __init__(self, base=64, layers=[4,5,3], block_num=4, type="cat", num_classes=1000, dropout=0.20, pretrain_model='', use_conv_last=False):
#         super(STDCNet1446, self).__init__()
#         if type == "cat":
#             block = CatBottleneck
#         elif type == "add":
#             block = AddBottleneck
#         self.use_conv_last = use_conv_last
#         self.features = self._make_layers(base, layers, block_num, block)
#         self.conv_last = ConvX(base*16, max(1024, base*16), 1, 1)
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(max(1024, base*16), max(1024, base*16), bias=False)
#         self.bn = nn.BatchNorm1d(max(1024, base*16))
#         self.relu = nn.ReLU(inplace=True)
#         self.dropout = nn.Dropout(p=dropout)
#         self.linear = nn.Linear(max(1024, base*16), num_classes, bias=False)
#         self.trans4 = ConvX(base,base*2,3,2)
#         self.trans8 = ConvX(base*2,base*4,3,2)
#         self.trans16 = ConvX(base*4,base*8,3,2)
# #         self.concatfuse8 = ConvX(base*4, base*4)
# #         self.concatfuse16 = ConvX(base*8, base*8)
# #         self.concatfuse32 = ConvX(base*16, base*16)

#         self.x2 = nn.Sequential(self.features[:1])
#         self.x4 = nn.Sequential(self.features[1:2])
#         self.x8 = nn.Sequential(self.features[2:6])
#         self.x16 = nn.Sequential(self.features[6:11])
#         self.x32 = nn.Sequential(self.features[11:])

#         if pretrain_model:
# #             print('use pretrain model {}'.format(pretrain_model))
# #             self.init_weight(pretrain_model)
# #         else:
#             self.init_params()

#     def init_weight(self, pretrain_model):
        
#         state_dict = torch.load(pretrain_model)["state_dict"]
#         self_state_dict = self.state_dict()
#         for k, v in state_dict.items():
#             self_state_dict.update({k: v})
#         self.load_state_dict(self_state_dict)

#     def init_params(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)

#     def _make_layers(self, base, layers, block_num, block):
#         features = []
#         features += [ConvX(3, base//2, 3, 2)]
#         features += [ConvX(base//2, base, 3, 2)]

#         for i, layer in enumerate(layers):
#             for j in range(layer):
#                 if i == 0 and j == 0:
#                     features.append(block(base, base*2, block_num, 2))
#                 elif j == 0:
#                     features.append(block(base*int(math.pow(2,i)), base*int(math.pow(2,i+1)), block_num, 2))
#                 else:
#                     features.append(block(base*int(math.pow(2,i+1)), base*int(math.pow(2,i+1)), block_num, 1))

#         return nn.Sequential(*features)

#     def forward(self, x):
#         feat2 = self.x2(x)
#         feat4 = self.x4(feat2)
# #         print("feat4:")
# #         print(feat4.size())
        
#         trans4 = self.trans4(feat4)
        
#         feat8_temp = self.x8(feat4)
#         feat8 = torch.cat((trans4,feat8_temp),dim=1)
# #         feat8 = self.concatfuse8(feat8)
# #         print("feat8:")
# #         print(feat8.size())
        
#         trans8 = self.trans8(feat8_temp)
        
#         feat16_temp = self.x16(feat8_temp)
#         feat16 = torch.cat((trans8,feat16_temp),dim=1)
# #         feat16 = self.concatfuse16(feat16)
# #         print("feat16:")
# #         print(feat16.size())
        
#         trans16 = self.trans16(feat16_temp)
        
#         feat32_temp = self.x32(feat16_temp)
#         feat32 = torch.cat((trans16,feat32_temp),dim=1)
# #         feat32 = self.concatfuse32(feat32)
# #         print("feat32:")
# #         print(feat32.size())
        
#         if self.use_conv_last:
#            feat32 = self.conv_last(feat32)

#         return feat2, feat4, feat8, feat16, feat32

#     def forward_impl(self, x):
#         out = self.features(x)
#         out = self.conv_last(out).pow(2)
#         out = self.gap(out).flatten(1)
#         out = self.fc(out)
#         # out = self.bn(out)
#         out = self.relu(out)
#         # out = self.relu(self.bn(self.fc(out)))
#         out = self.dropout(out)
#         out = self.linear(out)
#         return out

# # STDC1Net
# class STDCNet813(nn.Module):
#     def __init__(self, base=64, layers=[2,2,2], block_num=4, type="cat", num_classes=1000, dropout=0.20, pretrain_model='', use_conv_last=False):
#         super(STDCNet813, self).__init__()
#         if type == "cat":
#             block = CatBottleneck
#         elif type == "add":
#             block = AddBottleneck
#         self.use_conv_last = use_conv_last
#         self.features = self._make_layers(base, layers, block_num, block)
#         self.conv_last = ConvX(base*16, max(1024, base*16), 1, 1)
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(max(1024, base*16), max(1024, base*16), bias=False)
#         self.bn = nn.BatchNorm1d(max(1024, base*16))
#         self.relu = nn.ReLU(inplace=True)
#         self.dropout = nn.Dropout(p=dropout)
#         self.linear = nn.Linear(max(1024, base*16), num_classes, bias=False)

#         self.x2 = nn.Sequential(self.features[:1])
#         self.x4 = nn.Sequential(self.features[1:2])
#         self.x8 = nn.Sequential(self.features[2:4])
#         self.x16 = nn.Sequential(self.features[4:6])
#         self.x32 = nn.Sequential(self.features[6:])

#         if pretrain_model:
#             print('use pretrain model {}'.format(pretrain_model))
#             self.init_weight(pretrain_model)
#         else:
#             self.init_params()

#     def init_weight(self, pretrain_model):
        
#         state_dict = torch.load(pretrain_model)["state_dict"]
#         self_state_dict = self.state_dict()
#         for k, v in state_dict.items():
#             self_state_dict.update({k: v})
#         self.load_state_dict(self_state_dict)

#     def init_params(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)

#     def _make_layers(self, base, layers, block_num, block):
#         features = []
#         features += [ConvX(3, base//2, 3, 2)]
#         features += [ConvX(base//2, base, 3, 2)]

#         for i, layer in enumerate(layers):
#             for j in range(layer):
#                 if i == 0 and j == 0:
#                     features.append(block(base, base*4, block_num, 2))
#                 elif j == 0:
#                     features.append(block(base*int(math.pow(2,i+1)), base*int(math.pow(2,i+2)), block_num, 2))
#                 else:
#                     features.append(block(base*int(math.pow(2,i+2)), base*int(math.pow(2,i+2)), block_num, 1))

#         return nn.Sequential(*features)

#     def forward(self, x):
#         feat2 = self.x2(x)
#         feat4 = self.x4(feat2)
#         feat8 = self.x8(feat4)
#         feat16 = self.x16(feat8)
#         feat32 = self.x32(feat16)
#         if self.use_conv_last:
#            feat32 = self.conv_last(feat32)

#         return feat2, feat4, feat8, feat16, feat32

#     def forward_impl(self, x):
#         out = self.features(x)
#         out = self.conv_last(out).pow(2)
#         out = self.gap(out).flatten(1)
#         out = self.fc(out)
#         # out = self.bn(out)
#         out = self.relu(out)
#         # out = self.relu(self.bn(self.fc(out)))
#         out = self.dropout(out)
#         out = self.linear(out)
#         return out

# if __name__ == "__main__":
#     model = STDCNet813(num_classes=1000, dropout=0.00, block_num=4)
#     model.eval()
#     x = torch.randn(1,3,224,224)
#     y = model(x)
#     torch.save(model.state_dict(), 'cat.pth')
#     print(y.size())
######################################################################################第五次实验
# import torch
# import torch.nn as nn
# from torch.nn import init
# import math
# BatchNorm2d = nn.BatchNorm2d

# # class depthwise_separable_conv(nn.Module):
# #     def init(self, nin, nout):
# #         super(depthwise_separable_conv, self).init()
# #         self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin)
# #         self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)
# #     def forward(self, x):
# #         out = self.depthwise(x)
# #         out = self.pointwise(out)
# #         return out
# # class ConvBNReLU(nn.Module):
# #     def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
# #         super(ConvBNReLU, self).__init__()
# #         self.conv = nn.Conv2d(in_chan,
# #                 out_chan,
# #                 kernel_size = ks,
# #                 stride = stride,
# #                 padding = padding,
# #                 bias = False)
# #         # self.bn = BatchNorm2d(out_chan)
# #         self.bn = BatchNorm2d(out_chan)
# #         self.relu = nn.ReLU()
# # #         self.init_weight()

# #     def forward(self, x):
# #         x = self.conv(x)
# #         x = self.bn(x)
# #         x = self.relu(x)
# #         return x
# class ConvX(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel=3, stride=1):
#         super(ConvX, self).__init__()
#         self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=kernel//2, bias=False)
# #         self.depthwise = nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1, groups=in_planes)
# #         self.pointwise = nn.Conv2d(in_planes, out_planes, kernel_size=1)
#         self.bn = nn.BatchNorm2d(out_planes)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         out = self.relu(self.bn(self.conv(x)))
# #         out = self.relu(self.bn(self.pointwise(self.depthwise(x))))
#         return out


# class AddBottleneck(nn.Module):
#     def __init__(self, in_planes, out_planes, block_num=3, stride=1):
#         super(AddBottleneck, self).__init__()
#         assert block_num > 1, print("block number should be larger than 1.")
#         self.conv_list = nn.ModuleList()
#         self.stride = stride
#         if stride == 2:
#             self.avd_layer = nn.Sequential(
#                 nn.Conv2d(out_planes//2, out_planes//2, kernel_size=3, stride=2, padding=1, groups=out_planes//2, bias=False),
#                 nn.BatchNorm2d(out_planes//2),
#             )
#             self.skip = nn.Sequential(
#                 nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=2, padding=1, groups=in_planes, bias=False),
#                 nn.BatchNorm2d(in_planes),
#                 nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
#                 nn.BatchNorm2d(out_planes),
#             )
#             stride = 1

#         for idx in range(block_num):
#             if idx == 0:
#                 self.conv_list.append(ConvX(in_planes, out_planes//2, kernel=1))
#             elif idx == 1 and block_num == 2:
#                 self.conv_list.append(ConvX(out_planes//2, out_planes//2, stride=stride))
#             elif idx == 1 and block_num > 2:
#                 self.conv_list.append(ConvX(out_planes//2, out_planes//4, stride=stride))
#             elif idx < block_num - 1:
#                 self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx+1))))
#             else:
#                 self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx))))
            
#     def forward(self, x):
#         out_list = []
#         out = x

#         for idx, conv in enumerate(self.conv_list):
#             if idx == 0 and self.stride == 2:
#                 out = self.avd_layer(conv(out))
#             else:
#                 out = conv(out)
#             out_list.append(out)

#         if self.stride == 2:
#             x = self.skip(x)

#         return torch.cat(out_list, dim=1) + x

# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
           
#         self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
#                                nn.ReLU(),
#                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.fc(self.avg_pool(x))
#         max_out = self.fc(self.max_pool(x))
#         out = avg_out + max_out
#         return self.sigmoid(out)

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

# class CatBottleneck(nn.Module):
#     def __init__(self, in_planes, out_planes, block_num=3, stride=1):
#         super(CatBottleneck, self).__init__()
#         assert block_num > 1, print("block number should be larger than 1.")
#         self.conv_list = nn.ModuleList()
#         self.stride = stride
# #         self.atenchannal2 = ChannelAttention()
#         if stride == 2:
#             self.avd_layer = nn.Sequential(
#                 nn.Conv2d(out_planes//2, out_planes//2, kernel_size=3, stride=2, padding=1, groups=out_planes//2, bias=False),
#                 nn.BatchNorm2d(out_planes//2),
#             )
#             self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
#             stride = 1

#         for idx in range(block_num):
#             if idx == 0:
#                 self.conv_list.append(ConvX(in_planes, out_planes//2, kernel=1))
# #                 self.conv_list.append(ChannelAttention(out_planes//2))
#             elif idx == 1 and block_num == 2:
#                 self.conv_list.append(ConvX(out_planes//2, out_planes//2, stride=stride))
# #                 self.conv_list.append(ChannelAttention(out_planes//2))
#             elif idx == 1 and block_num > 2:
#                 self.conv_list.append(ConvX(out_planes//2, out_planes//4, stride=stride))
# #                 self.conv_list.append(ChannelAttention(out_planes//4))
#             elif idx < block_num - 1:
#                 self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx+1))))
# #                 self.conv_list.append(ChannelAttention(out_planes//int(math.pow(2, idx+1))))
#             else:
#                 self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx))))
# #                 self.conv_list.append(ChannelAttention(out_planes//int(math.pow(2, idx))))
                                      
            
#     def forward(self, x):
#         out_list = []
#         out1 = self.conv_list[0](x)
# #         out1 = (self.conv_list[1](out1))*out1
# #         print("out1:")
# #         print(out1.size())

#         for idx, conv in enumerate(self.conv_list[1:]):
#             if idx == 0:
#                 if self.stride == 2:
#                     out = conv(self.avd_layer(out1))
# #                     print("out2:")
# #                     print(out.size())
# #                     out = (self.conv_list[2*idx+3](out))*out
# #                     print("out2:")
# #                     print(out.size())
#                 else:
#                     out = conv(out1)
# #                     out = (self.conv_list[2*idx+3](out))*out
# #                     print("out2:")
# #                     print(out.size())
#             else:
#                 out = conv(out)
# #                 out = (self.conv_list[2*idx+3](out))*out
# #                 print("out:")
# #                 print(out.size())
#             out_list.append(out)

#         if self.stride == 2:
#             out1 = self.skip(out1)
#         out_list.insert(0, out1)

#         out = torch.cat(out_list, dim=1)
#         return out

# #STDC2Net
# class STDCNet1446(nn.Module):
#     def __init__(self, base=64, layers=[5,6,4], block_num=4, type="cat", num_classes=1000, dropout=0.20, pretrain_model='', use_conv_last=False):
#         super(STDCNet1446, self).__init__()
#         if type == "cat":
#             block = CatBottleneck
#         elif type == "add":
#             block = AddBottleneck
#         self.use_conv_last = use_conv_last
#         self.features = self._make_layers(base, layers, block_num, block)
#         self.conv_last = ConvX(base*16, max(1024, base*16), 1, 1)
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(max(1024, base*16), max(1024, base*16), bias=False)
#         self.bn = nn.BatchNorm1d(max(1024, base*16))
#         self.relu = nn.ReLU(inplace=True)
#         self.dropout = nn.Dropout(p=dropout)
#         self.linear = nn.Linear(max(1024, base*16), num_classes, bias=False)
# #         self.trans4 = ConvX(base,base*2,3,2)
# #         self.trans8 = ConvX(base*2,base*4,3,2)
# #         self.trans16 = ConvX(base*4,base*8,3,2)
# #         self.concatfuse8 = ConvX(base*4, base*4)
# #         self.concatfuse16 = ConvX(base*8, base*8)
# #         self.concatfuse32 = ConvX(base*16, base*16)

#         self.x2 = nn.Sequential(self.features[:1])
#         self.x4 = nn.Sequential(self.features[1:2])
#         self.x8 = nn.Sequential(self.features[2:6])
#         self.t256 = nn.Sequential(self.features[6])
#         self.x16 = nn.Sequential(self.features[7:12])
#         self.t512 = nn.Sequential(self.features[12])
#         self.x32 = nn.Sequential(self.features[13:16])
#         self.t1024 = nn.Sequential(self.features[16])

#         if pretrain_model:
# #             print('use pretrain model {}'.format(pretrain_model))
# #             self.init_weight(pretrain_model)
# #         else:
#             self.init_params()

#     def init_weight(self, pretrain_model):
        
#         state_dict = torch.load(pretrain_model)["state_dict"]
#         self_state_dict = self.state_dict()
#         for k, v in state_dict.items():
#             self_state_dict.update({k: v})
#         self.load_state_dict(self_state_dict)

#     def init_params(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)

#     def _make_layers(self, base, layers, block_num, block):
#         features = []
#         features += [ConvX(3, base//2, 3, 2)]
#         features += [ConvX(base//2, base, 3, 2)]

#         for i, layer in enumerate(layers):
#             for j in range(layer):
#                 if i == 0 and j == 0:
#                     features.append(block(base, base*2, block_num, 2))
#                 elif j == 0:
#                     features.append(block(base*int(math.pow(2,i)), base*int(math.pow(2,i+1)), block_num, 2))
#                 elif j+1==layer:
#                     features.append(block(base*int(math.pow(2,i+1)),base*int(math.pow(2,i+2))))
#                 else:
#                     features.append(block(base*int(math.pow(2,i+1)), base*int(math.pow(2,i+1)), block_num, 1))
# #             features.append(block(base*int(math.pow(2,i+1)),base*int(math.pow(2,i+2))))

#         return nn.Sequential(*features)

#     def forward(self, x):
#         feat2 = self.x2(x)
#         feat4 = self.x4(feat2)
# #         print("feat4:")
# #         print(feat4.size())
        
# #         trans4 = self.trans4(feat4)
        
# #         feat8_temp = self.x8(feat4)
#         feat8tmp = self.x8(feat4)
#         feat8 = self.t256(feat8tmp)
# #         feat8 = torch.cat((trans4,feat8_temp),dim=1)
# #         feat8 = self.concatfuse8(feat8)
# #         print("feat8:")
# #         print(feat8.size())
        
# #         trans8 = self.trans8(feat8_temp)
        
#         feat16tmp = self.x16(feat8tmp)
#         feat16 = self.t512(feat16tmp)
# #         feat16_temp = self.x16(feat8_temp)
# #         feat16 = torch.cat((trans8,feat16_temp),dim=1)
# #         feat16 = self.concatfuse16(feat16)
# #         print("feat16:")
# #         print(feat16.size())
        
# #         trans16 = self.trans16(feat16_temp)
        
#         feat32tmp = self.x32(feat16tmp)
#         feat32 = self.t1024(feat32tmp)
# #         feat32 = torch.cat((trans16,feat32_temp),dim=1)
# #         feat32 = self.concatfuse32(feat32)
# #         print("feat32:")
# #         print(feat32.size())
        
#         if self.use_conv_last:
#            feat32 = self.conv_last(feat32)

#         return feat2, feat4, feat8, feat16, feat32

#     def forward_impl(self, x):
#         out = self.features(x)
#         out = self.conv_last(out).pow(2)
#         out = self.gap(out).flatten(1)
#         out = self.fc(out)
#         # out = self.bn(out)
#         out = self.relu(out)
#         # out = self.relu(self.bn(self.fc(out)))
#         out = self.dropout(out)
#         out = self.linear(out)
#         return out

# # STDC1Net
# class STDCNet813(nn.Module):
#     def __init__(self, base=64, layers=[2,2,2], block_num=4, type="cat", num_classes=1000, dropout=0.20, pretrain_model='', use_conv_last=False):
#         super(STDCNet813, self).__init__()
#         if type == "cat":
#             block = CatBottleneck
#         elif type == "add":
#             block = AddBottleneck
#         self.use_conv_last = use_conv_last
#         self.features = self._make_layers(base, layers, block_num, block)
#         self.conv_last = ConvX(base*16, max(1024, base*16), 1, 1)
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(max(1024, base*16), max(1024, base*16), bias=False)
#         self.bn = nn.BatchNorm1d(max(1024, base*16))
#         self.relu = nn.ReLU(inplace=True)
#         self.dropout = nn.Dropout(p=dropout)
#         self.linear = nn.Linear(max(1024, base*16), num_classes, bias=False)

#         self.x2 = nn.Sequential(self.features[:1])
#         self.x4 = nn.Sequential(self.features[1:2])
#         self.x8 = nn.Sequential(self.features[2:4])
#         self.x16 = nn.Sequential(self.features[4:6])
#         self.x32 = nn.Sequential(self.features[6:])

#         if pretrain_model:
#             print('use pretrain model {}'.format(pretrain_model))
#             self.init_weight(pretrain_model)
#         else:
#             self.init_params()

#     def init_weight(self, pretrain_model):
        
#         state_dict = torch.load(pretrain_model)["state_dict"]
#         self_state_dict = self.state_dict()
#         for k, v in state_dict.items():
#             self_state_dict.update({k: v})
#         self.load_state_dict(self_state_dict)

#     def init_params(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)

#     def _make_layers(self, base, layers, block_num, block):
#         features = []
#         features += [ConvX(3, base//2, 3, 2)]
#         features += [ConvX(base//2, base, 3, 2)]

#         for i, layer in enumerate(layers):
#             for j in range(layer):
#                 if i == 0 and j == 0:
#                     features.append(block(base, base*4, block_num, 2))
#                 elif j == 0:
#                     features.append(block(base*int(math.pow(2,i+1)), base*int(math.pow(2,i+2)), block_num, 2))
#                 else:
#                     features.append(block(base*int(math.pow(2,i+2)), base*int(math.pow(2,i+2)), block_num, 1))

#         return nn.Sequential(*features)

#     def forward(self, x):
#         feat2 = self.x2(x)
#         feat4 = self.x4(feat2)
#         feat8 = self.x8(feat4)
#         feat16 = self.x16(feat8)
#         feat32 = self.x32(feat16)
#         if self.use_conv_last:
#            feat32 = self.conv_last(feat32)

#         return feat2, feat4, feat8, feat16, feat32

#     def forward_impl(self, x):
#         out = self.features(x)
#         out = self.conv_last(out).pow(2)
#         out = self.gap(out).flatten(1)
#         out = self.fc(out)
#         # out = self.bn(out)
#         out = self.relu(out)
#         # out = self.relu(self.bn(self.fc(out)))
#         out = self.dropout(out)
#         out = self.linear(out)
#         return out

# if __name__ == "__main__":
#     model = STDCNet813(num_classes=1000, dropout=0.00, block_num=4)
#     model.eval()
#     x = torch.randn(1,3,224,224)
#     y = model(x)
#     torch.save(model.state_dict(), 'cat.pth')
#     print(y.size())
#####################################################################################################第七次实验
# import torch
# import torch.nn as nn
# from torch.nn import init
# import math
# BatchNorm2d = nn.BatchNorm2d

# # class depthwise_separable_conv(nn.Module):
# #     def init(self, nin, nout):
# #         super(depthwise_separable_conv, self).init()
# #         self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin)
# #         self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)
# #     def forward(self, x):
# #         out = self.depthwise(x)
# #         out = self.pointwise(out)
# #         return out
# # class ConvBNReLU(nn.Module):
# #     def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
# #         super(ConvBNReLU, self).__init__()
# #         self.conv = nn.Conv2d(in_chan,
# #                 out_chan,
# #                 kernel_size = ks,
# #                 stride = stride,
# #                 padding = padding,
# #                 bias = False)
# #         # self.bn = BatchNorm2d(out_chan)
# #         self.bn = BatchNorm2d(out_chan)
# #         self.relu = nn.ReLU()
# # #         self.init_weight()

# #     def forward(self, x):
# #         x = self.conv(x)
# #         x = self.bn(x)
# #         x = self.relu(x)
# #         return x
# class ConvX(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel=3, stride=1):
#         super(ConvX, self).__init__()
#         self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=kernel//2, bias=False)
# #         self.depthwise = nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1, groups=in_planes)
# #         self.pointwise = nn.Conv2d(in_planes, out_planes, kernel_size=1)
#         self.bn = nn.BatchNorm2d(out_planes)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         out = self.relu(self.bn(self.conv(x)))
# #         out = self.relu(self.bn(self.pointwise(self.depthwise(x))))
#         return out


# class AddBottleneck(nn.Module):
#     def __init__(self, in_planes, out_planes, block_num=3, stride=1):
#         super(AddBottleneck, self).__init__()
#         assert block_num > 1, print("block number should be larger than 1.")
#         self.conv_list = nn.ModuleList()
#         self.stride = stride
#         if stride == 2:
#             self.avd_layer = nn.Sequential(
#                 nn.Conv2d(out_planes//2, out_planes//2, kernel_size=3, stride=2, padding=1, groups=out_planes//2, bias=False),
#                 nn.BatchNorm2d(out_planes//2),
#             )
#             self.skip = nn.Sequential(
#                 nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=2, padding=1, groups=in_planes, bias=False),
#                 nn.BatchNorm2d(in_planes),
#                 nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
#                 nn.BatchNorm2d(out_planes),
#             )
#             stride = 1

#         for idx in range(block_num):
#             if idx == 0:
#                 self.conv_list.append(ConvX(in_planes, out_planes//2, kernel=1))
#             elif idx == 1 and block_num == 2:
#                 self.conv_list.append(ConvX(out_planes//2, out_planes//2, stride=stride))
#             elif idx == 1 and block_num > 2:
#                 self.conv_list.append(ConvX(out_planes//2, out_planes//4, stride=stride))
#             elif idx < block_num - 1:
#                 self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx+1))))
#             else:
#                 self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx))))
            
#     def forward(self, x):
#         out_list = []
#         out = x

#         for idx, conv in enumerate(self.conv_list):
#             if idx == 0 and self.stride == 2:
#                 out = self.avd_layer(conv(out))
#             else:
#                 out = conv(out)
#             out_list.append(out)

#         if self.stride == 2:
#             x = self.skip(x)

#         return torch.cat(out_list, dim=1) + x

# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
           
#         self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
#                                nn.ReLU(),
#                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.fc(self.avg_pool(x))
#         max_out = self.fc(self.max_pool(x))
#         out = avg_out + max_out
#         return self.sigmoid(out)

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

# class CatBottleneck(nn.Module):
#     def __init__(self, in_planes, out_planes, block_num=3, stride=1):
#         super(CatBottleneck, self).__init__()
#         assert block_num > 1, print("block number should be larger than 1.")
#         self.conv_list = nn.ModuleList()
#         self.stride = stride
#         self.atenchannal2 = SpatialAttention()
#         if stride == 2:
#             self.avd_layer = nn.Sequential(
#                 nn.Conv2d(out_planes//2, out_planes//2, kernel_size=3, stride=2, padding=1, groups=out_planes//2, bias=False),
#                 nn.BatchNorm2d(out_planes//2),
#             )
#             self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
#             stride = 1

#         for idx in range(block_num):
#             if idx == 0:
#                 self.conv_list.append(ConvX(in_planes, out_planes//2, kernel=1))
# #                 self.conv_list.append(ChannelAttention(out_planes//2))
#             elif idx == 1 and block_num == 2:
#                 self.conv_list.append(ConvX(out_planes//2, out_planes//2, stride=stride))
# #                 self.conv_list.append(ChannelAttention(out_planes//2))
#             elif idx == 1 and block_num > 2:
#                 self.conv_list.append(ConvX(out_planes//2, out_planes//4, stride=stride))
# #                 self.conv_list.append(ChannelAttention(out_planes//4))
#             elif idx < block_num - 1:
#                 self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx+1))))
# #                 self.conv_list.append(ChannelAttention(out_planes//int(math.pow(2, idx+1))))
#             else:
#                 self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx))))
# #                 self.conv_list.append(ChannelAttention(out_planes//int(math.pow(2, idx))))
                                      
            
#     def forward(self, x):
#         out_list = []
#         out1 = self.conv_list[0](x)
# #         out1 = self.atenchannal2(out1)*out1
# #         print("out1:")
# #         print(out1.size())

#         for idx, conv in enumerate(self.conv_list[1:]):
#             if idx == 0:
#                 if self.stride == 2:
#                     out = conv(self.avd_layer(out1))
# #                     print("out2:")
# #                     print(out.size())
# #                     out = (self.conv_list[2*idx+3](out))*out
# #                     print("out2:")
# #                     print(out.size())
#                 else:
#                     out = conv(out1)
# #                     out = (self.conv_list[2*idx+3](out))*out
# #                     print("out2:")
# #                     print(out.size())
#             else:
#                 out = conv(out)
# #                 out = (self.conv_list[2*idx+3](out))*out
# #                 print("out:")
# #                 print(out.size())
#             out_list.append(out)

#         if self.stride == 2:
#             out1 = self.skip(out1)
#         out_list.insert(0, out1)

#         out = torch.cat(out_list, dim=1)
#         out = self.atenchannal2(out)*out
#         return out

# #STDC2Net
# class STDCNet1446(nn.Module):
#     def __init__(self, base=64, layers=[5,6,4], block_num=4, type="cat", num_classes=1000, dropout=0.20, pretrain_model='', use_conv_last=False):
#         super(STDCNet1446, self).__init__()
#         if type == "cat":
#             block = CatBottleneck
#         elif type == "add":
#             block = AddBottleneck
#         self.use_conv_last = use_conv_last
#         self.features = self._make_layers(base, layers, block_num, block)
#         self.conv_last = ConvX(base*16, max(1024, base*16), 1, 1)
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(max(1024, base*16), max(1024, base*16), bias=False)
#         self.bn = nn.BatchNorm1d(max(1024, base*16))
#         self.relu = nn.ReLU(inplace=True)
#         self.dropout = nn.Dropout(p=dropout)
#         self.linear = nn.Linear(max(1024, base*16), num_classes, bias=False)
# #         self.trans4 = ConvX(base,base*2,3,2)
# #         self.trans8 = ConvX(base*2,base*4,3,2)
# #         self.trans16 = ConvX(base*4,base*8,3,2)
# #         self.concatfuse8 = ConvX(base*4, base*4)
# #         self.concatfuse16 = ConvX(base*8, base*8)
# #         self.concatfuse32 = ConvX(base*16, base*16)

#         self.x2 = nn.Sequential(self.features[:1])
#         self.x4 = nn.Sequential(self.features[1:2])
#         self.x8 = nn.Sequential(self.features[2:6])
#         self.t256 = nn.Sequential(self.features[6])
#         self.x16 = nn.Sequential(self.features[7:12])
#         self.t512 = nn.Sequential(self.features[12])
#         self.x32 = nn.Sequential(self.features[13:16])
#         self.t1024 = nn.Sequential(self.features[16])
        

#         if pretrain_model:
# #             print('use pretrain model {}'.format(pretrain_model))
# #             self.init_weight(pretrain_model)
# #         else:
#             self.init_params()

#     def init_weight(self, pretrain_model):
        
#         state_dict = torch.load(pretrain_model)["state_dict"]
#         self_state_dict = self.state_dict()
#         for k, v in state_dict.items():
#             self_state_dict.update({k: v})
#         self.load_state_dict(self_state_dict)

#     def init_params(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)

#     def _make_layers(self, base, layers, block_num, block):
#         features = []
#         features += [ConvX(3, base//2, 3, 2)]
#         features += [ConvX(base//2, base, 3, 2)]

#         for i, layer in enumerate(layers):
#             for j in range(layer):
#                 if i == 0 and j == 0:
#                     features.append(block(base, base*2, block_num, 2))
#                 elif j == 0:
#                     features.append(block(base*int(math.pow(2,i)), base*int(math.pow(2,i+1)), block_num, 2))
#                 elif j+1==layer:
#                     features.append(block(base*int(math.pow(2,i+1)),base*int(math.pow(2,i+2))))
#                 else:
#                     features.append(block(base*int(math.pow(2,i+1)), base*int(math.pow(2,i+1)), block_num, 1))
# #             features.append(block(base*int(math.pow(2,i+1)),base*int(math.pow(2,i+2))))

#         return nn.Sequential(*features)

#     def forward(self, x):
#         feat2 = self.x2(x)
#         feat4 = self.x4(feat2)
# #         print("feat4:")
# #         print(feat4.size())
        
# #         trans4 = self.trans4(feat4)
        
# #         feat8_temp = self.x8(feat4)
#         feat8tmp = self.x8(feat4)
#         feat8 = self.t256(feat8tmp)
# #         feat8 = torch.cat((trans4,feat8_temp),dim=1)
# #         feat8 = self.concatfuse8(feat8)
# #         print("feat8:")
# #         print(feat8.size())
        
# #         trans8 = self.trans8(feat8_temp)
        
#         feat16tmp = self.x16(feat8tmp)
#         feat16 = self.t512(feat16tmp)
# #         feat16_temp = self.x16(feat8_temp)
# #         feat16 = torch.cat((trans8,feat16_temp),dim=1)
# #         feat16 = self.concatfuse16(feat16)
# #         print("feat16:")
# #         print(feat16.size())
        
# #         trans16 = self.trans16(feat16_temp)
        
#         feat32tmp = self.x32(feat16tmp)
#         feat32 = self.t1024(feat32tmp)
# #         feat32 = torch.cat((trans16,feat32_temp),dim=1)
# #         feat32 = self.concatfuse32(feat32)
# #         print("feat32:")
# #         print(feat32.size())
        
#         if self.use_conv_last:
#            feat32 = self.conv_last(feat32)

#         return feat2, feat4, feat8, feat16, feat32

#     def forward_impl(self, x):
#         out = self.features(x)
#         out = self.conv_last(out).pow(2)
#         out = self.gap(out).flatten(1)
#         out = self.fc(out)
#         # out = self.bn(out)
#         out = self.relu(out)
#         # out = self.relu(self.bn(self.fc(out)))
#         out = self.dropout(out)
#         out = self.linear(out)
#         return out

# # STDC1Net
# class STDCNet813(nn.Module):
#     def __init__(self, base=64, layers=[2,2,2], block_num=4, type="cat", num_classes=1000, dropout=0.20, pretrain_model='', use_conv_last=False):
#         super(STDCNet813, self).__init__()
#         if type == "cat":
#             block = CatBottleneck
#         elif type == "add":
#             block = AddBottleneck
#         self.use_conv_last = use_conv_last
#         self.features = self._make_layers(base, layers, block_num, block)
#         self.conv_last = ConvX(base*16, max(1024, base*16), 1, 1)
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(max(1024, base*16), max(1024, base*16), bias=False)
#         self.bn = nn.BatchNorm1d(max(1024, base*16))
#         self.relu = nn.ReLU(inplace=True)
#         self.dropout = nn.Dropout(p=dropout)
#         self.linear = nn.Linear(max(1024, base*16), num_classes, bias=False)

#         self.x2 = nn.Sequential(self.features[:1])
#         self.x4 = nn.Sequential(self.features[1:2])
#         self.x8 = nn.Sequential(self.features[2:4])
#         self.x16 = nn.Sequential(self.features[4:6])
#         self.x32 = nn.Sequential(self.features[6:])

#         if pretrain_model:
#             print('use pretrain model {}'.format(pretrain_model))
#             self.init_weight(pretrain_model)
#         else:
#             self.init_params()

#     def init_weight(self, pretrain_model):
        
#         state_dict = torch.load(pretrain_model)["state_dict"]
#         self_state_dict = self.state_dict()
#         for k, v in state_dict.items():
#             self_state_dict.update({k: v})
#         self.load_state_dict(self_state_dict)

#     def init_params(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)

#     def _make_layers(self, base, layers, block_num, block):
#         features = []
#         features += [ConvX(3, base//2, 3, 2)]
#         features += [ConvX(base//2, base, 3, 2)]

#         for i, layer in enumerate(layers):
#             for j in range(layer):
#                 if i == 0 and j == 0:
#                     features.append(block(base, base*4, block_num, 2))
#                 elif j == 0:
#                     features.append(block(base*int(math.pow(2,i+1)), base*int(math.pow(2,i+2)), block_num, 2))
#                 else:
#                     features.append(block(base*int(math.pow(2,i+2)), base*int(math.pow(2,i+2)), block_num, 1))

#         return nn.Sequential(*features)

#     def forward(self, x):
#         feat2 = self.x2(x)
#         feat4 = self.x4(feat2)
#         feat8 = self.x8(feat4)
#         feat16 = self.x16(feat8)
#         feat32 = self.x32(feat16)
#         if self.use_conv_last:
#            feat32 = self.conv_last(feat32)

#         return feat2, feat4, feat8, feat16, feat32

#     def forward_impl(self, x):
#         out = self.features(x)
#         out = self.conv_last(out).pow(2)
#         out = self.gap(out).flatten(1)
#         out = self.fc(out)
#         # out = self.bn(out)
#         out = self.relu(out)
#         # out = self.relu(self.bn(self.fc(out)))
#         out = self.dropout(out)
#         out = self.linear(out)
#         return out

# if __name__ == "__main__":
#     model = STDCNet813(num_classes=1000, dropout=0.00, block_num=4)
#     model.eval()
#     x = torch.randn(1,3,224,224)
#     y = model(x)
#     torch.save(model.state_dict(), 'cat.pth')
#     print(y.size())
#############################################################################第八次实验
# import torch
# import torch.nn as nn
# from torch.nn import init
# import math
# BatchNorm2d = nn.BatchNorm2d

# # class depthwise_separable_conv(nn.Module):
# #     def init(self, nin, nout):
# #         super(depthwise_separable_conv, self).init()
# #         self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin)
# #         self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)
# #     def forward(self, x):
# #         out = self.depthwise(x)
# #         out = self.pointwise(out)
# #         return out
# # class ConvBNReLU(nn.Module):
# #     def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
# #         super(ConvBNReLU, self).__init__()
# #         self.conv = nn.Conv2d(in_chan,
# #                 out_chan,
# #                 kernel_size = ks,
# #                 stride = stride,
# #                 padding = padding,
# #                 bias = False)
# #         # self.bn = BatchNorm2d(out_chan)
# #         self.bn = BatchNorm2d(out_chan)
# #         self.relu = nn.ReLU()
# # #         self.init_weight()

# #     def forward(self, x):
# #         x = self.conv(x)
# #         x = self.bn(x)
# #         x = self.relu(x)
# #         return x
# class ConvX(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel=3, stride=1):
#         super(ConvX, self).__init__()
#         self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=kernel//2, bias=False)
# #         self.depthwise = nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1, groups=in_planes)
# #         self.pointwise = nn.Conv2d(in_planes, out_planes, kernel_size=1)
#         self.bn = nn.BatchNorm2d(out_planes)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         out = self.relu(self.bn(self.conv(x)))
# #         out = self.relu(self.bn(self.pointwise(self.depthwise(x))))
#         return out


# class AddBottleneck(nn.Module):
#     def __init__(self, in_planes, out_planes, block_num=3, stride=1):
#         super(AddBottleneck, self).__init__()
#         assert block_num > 1, print("block number should be larger than 1.")
#         self.conv_list = nn.ModuleList()
#         self.stride = stride
#         if stride == 2:
#             self.avd_layer = nn.Sequential(
#                 nn.Conv2d(out_planes//2, out_planes//2, kernel_size=3, stride=2, padding=1, groups=out_planes//2, bias=False),
#                 nn.BatchNorm2d(out_planes//2),
#             )
#             self.skip = nn.Sequential(
#                 nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=2, padding=1, groups=in_planes, bias=False),
#                 nn.BatchNorm2d(in_planes),
#                 nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
#                 nn.BatchNorm2d(out_planes),
#             )
#             stride = 1

#         for idx in range(block_num):
#             if idx == 0:
#                 self.conv_list.append(ConvX(in_planes, out_planes//2, kernel=1))
#             elif idx == 1 and block_num == 2:
#                 self.conv_list.append(ConvX(out_planes//2, out_planes//2, stride=stride))
#             elif idx == 1 and block_num > 2:
#                 self.conv_list.append(ConvX(out_planes//2, out_planes//4, stride=stride))
#             elif idx < block_num - 1:
#                 self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx+1))))
#             else:
#                 self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx))))
            
#     def forward(self, x):
#         out_list = []
#         out = x

#         for idx, conv in enumerate(self.conv_list):
#             if idx == 0 and self.stride == 2:
#                 out = self.avd_layer(conv(out))
#             else:
#                 out = conv(out)
#             out_list.append(out)

#         if self.stride == 2:
#             x = self.skip(x)

#         return torch.cat(out_list, dim=1) + x

# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
           
#         self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
#                                nn.ReLU(),
#                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.fc(self.avg_pool(x))
#         max_out = self.fc(self.max_pool(x))
#         out = avg_out + max_out
#         return self.sigmoid(out)

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

# class CatBottleneck(nn.Module):
#     def __init__(self, in_planes, out_planes, block_num=3, stride=1):
#         super(CatBottleneck, self).__init__()
#         assert block_num > 1, print("block number should be larger than 1.")
#         self.conv_list = nn.ModuleList()
#         self.stride = stride
#         self.atenchannal2 = SpatialAttention()
#         if stride == 2:
#             self.avd_layer = nn.Sequential(
#                 nn.Conv2d(out_planes//2, out_planes//2, kernel_size=3, stride=2, padding=1, groups=out_planes//2, bias=False),
#                 nn.BatchNorm2d(out_planes//2),
#             )
#             self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
#             stride = 1
#         self.avg = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
#         self.conavg_layer = nn.Sequential(
#             nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False),
#         nn.BatchNorm2d(out_planes))
#         for idx in range(block_num):
#             if idx == 0:
#                 self.conv_list.append(ConvX(in_planes, out_planes//2, kernel=1))
# #                 self.conv_list.append(ConvX(in_planes,out_planes, kernel=1,stride=self.stride))
# #                 self.conv_list.append(ChannelAttention(out_planes//2))
#             elif idx == 1 and block_num == 2:
#                 self.conv_list.append(ConvX(out_planes//2, out_planes//2, stride=stride))

# #                 self.conv_list.append(ChannelAttention(out_planes//2))
#             elif idx == 1 and block_num > 2:
#                 self.conv_list.append(ConvX(out_planes//2, out_planes//4, stride=stride))

# #                 self.conv_list.append(ChannelAttention(out_planes//4))
#             elif idx < block_num - 1:
#                 self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx+1))))

# #                 self.conv_list.append(ChannelAttention(out_planes//int(math.pow(2, idx+1))))
#             else:
#                 self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx))))
              
# #                 self.conv_list.append(ChannelAttention(out_planes//int(math.pow(2, idx))))
                                      
            
#     def forward(self, x):
#         out_list = []
#         out1 = self.conv_list[0](x)
# #         out_atten = self.atenchannal2(out1)
#         if self.stride==2:
#             x = self.avg(x)
#         out_temp = self.conavg_layer(x)
# #         print("out_temp:")
# #         print(out_temp.size())

#         for idx, conv in enumerate(self.conv_list[1:]):
#             if idx == 0:
#                 if self.stride == 2:
#                     out = conv(self.avd_layer(out1))
# #                     print("out2:")
# #                     print(out.size())
# #                     out = self.conv_list[2*idx+3](out1)
# #                     print("out2:")
# #                     print(out.size())
#                 else:
#                     out = conv(out1)
# #                     out = self.conv_list[2*idx+3](out1)+out
# #                     print("out2:")
# #                     print(out.size())
#             else:
#                 out = conv(out)
# #                 out = atten + out
# #                 out = (self.conv_list[2*idx+3](out))+out_temp
# #                 print("out:")
# #                 print(out.size())
#             out_list.append(out)

#         if self.stride == 2:
#             out1 = self.skip(out1)
#         out_list.insert(0, out1)

#         out = torch.cat(out_list, dim=1)
# #         print("output:")
# #         print(out.size())
#         out = out+out_temp
#         out = self.atenchannal2(out)*out
# #         out = outlist[]
#         return out

# #STDC2Net
# class STDCNet1446(nn.Module):
#     def __init__(self, base=64, layers=[5,6,4], block_num=4, type="cat", num_classes=1000, dropout=0.20, pretrain_model='', use_conv_last=False):
#         super(STDCNet1446, self).__init__()
#         if type == "cat":
#             block = CatBottleneck
#         elif type == "add":
#             block = AddBottleneck
#         self.use_conv_last = use_conv_last
#         self.features = self._make_layers(base, layers, block_num, block)
#         self.conv_last = ConvX(base*16, max(1024, base*16), 1, 1)
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(max(1024, base*16), max(1024, base*16), bias=False)
#         self.bn = nn.BatchNorm1d(max(1024, base*16))
#         self.relu = nn.ReLU(inplace=True)
#         self.dropout = nn.Dropout(p=dropout)
#         self.linear = nn.Linear(max(1024, base*16), num_classes, bias=False)
# #         self.trans4 = ConvX(base,base*2,3,2)
# #         self.trans8 = ConvX(base*2,base*4,3,2)
# #         self.trans16 = ConvX(base*4,base*8,3,2)
# #         self.concatfuse8 = ConvX(base*4, base*4)
# #         self.concatfuse16 = ConvX(base*8, base*8)
# #         self.concatfuse32 = ConvX(base*16, base*16)

#         self.x2 = nn.Sequential(self.features[:1])
#         self.x4 = nn.Sequential(self.features[1:2])
#         self.x8 = nn.Sequential(self.features[2:6])
#         self.t256 = nn.Sequential(self.features[6])
#         self.x16 = nn.Sequential(self.features[7:12])
#         self.t512 = nn.Sequential(self.features[12])
#         self.x32 = nn.Sequential(self.features[13:16])
#         self.t1024 = nn.Sequential(self.features[16])
        

#         if pretrain_model:
# #             print('use pretrain model {}'.format(pretrain_model))
# #             self.init_weight(pretrain_model)
# #         else:
#             self.init_params()

#     def init_weight(self, pretrain_model):
        
#         state_dict = torch.load(pretrain_model)["state_dict"]
#         self_state_dict = self.state_dict()
#         for k, v in state_dict.items():
#             self_state_dict.update({k: v})
#         self.load_state_dict(self_state_dict)

#     def init_params(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)

#     def _make_layers(self, base, layers, block_num, block):
#         features = []
#         features += [ConvX(3, base//2, 3, 2)]
#         features += [ConvX(base//2, base, 3, 2)]

#         for i, layer in enumerate(layers):
#             for j in range(layer):
#                 if i == 0 and j == 0:
#                     features.append(block(base, base*2, block_num, 2))
#                 elif j == 0:
#                     features.append(block(base*int(math.pow(2,i)), base*int(math.pow(2,i+1)), block_num, 2))
#                 elif j+1==layer:
#                     features.append(block(base*int(math.pow(2,i+1)),base*int(math.pow(2,i+2))))
#                 else:
#                     features.append(block(base*int(math.pow(2,i+1)), base*int(math.pow(2,i+1)), block_num, 1))
# #             features.append(block(base*int(math.pow(2,i+1)),base*int(math.pow(2,i+2))))

#         return nn.Sequential(*features)

#     def forward(self, x):
#         feat2 = self.x2(x)
#         feat4 = self.x4(feat2)
# #         print("feat4:")
# #         print(feat4.size())
        
# #         trans4 = self.trans4(feat4)
        
# #         feat8_temp = self.x8(feat4)
#         feat8tmp = self.x8(feat4)
#         feat8 = self.t256(feat8tmp)
# #         feat8 = torch.cat((trans4,feat8_temp),dim=1)
# #         feat8 = self.concatfuse8(feat8)
# #         print("feat8:")
# #         print(feat8.size())
        
# #         trans8 = self.trans8(feat8_temp)
        
#         feat16tmp = self.x16(feat8tmp)
#         feat16 = self.t512(feat16tmp)
# #         feat16_temp = self.x16(feat8_temp)
# #         feat16 = torch.cat((trans8,feat16_temp),dim=1)
# #         feat16 = self.concatfuse16(feat16)
# #         print("feat16:")
# #         print(feat16.size())
        
# #         trans16 = self.trans16(feat16_temp)
        
#         feat32tmp = self.x32(feat16tmp)
#         feat32 = self.t1024(feat32tmp)
# #         feat32 = torch.cat((trans16,feat32_temp),dim=1)
# #         feat32 = self.concatfuse32(feat32)
# #         print("feat32:")
# #         print(feat32.size())
        
#         if self.use_conv_last:
#            feat32 = self.conv_last(feat32)

#         return feat2, feat4, feat8, feat16, feat32

#     def forward_impl(self, x):
#         out = self.features(x)
#         out = self.conv_last(out).pow(2)
#         out = self.gap(out).flatten(1)
#         out = self.fc(out)
#         # out = self.bn(out)
#         out = self.relu(out)
#         # out = self.relu(self.bn(self.fc(out)))
#         out = self.dropout(out)
#         out = self.linear(out)
#         return out

# # STDC1Net
# class STDCNet813(nn.Module):
#     def __init__(self, base=64, layers=[2,2,2], block_num=4, type="cat", num_classes=1000, dropout=0.20, pretrain_model='', use_conv_last=False):
#         super(STDCNet813, self).__init__()
#         if type == "cat":
#             block = CatBottleneck
#         elif type == "add":
#             block = AddBottleneck
#         self.use_conv_last = use_conv_last
#         self.features = self._make_layers(base, layers, block_num, block)
#         self.conv_last = ConvX(base*16, max(1024, base*16), 1, 1)
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(max(1024, base*16), max(1024, base*16), bias=False)
#         self.bn = nn.BatchNorm1d(max(1024, base*16))
#         self.relu = nn.ReLU(inplace=True)
#         self.dropout = nn.Dropout(p=dropout)
#         self.linear = nn.Linear(max(1024, base*16), num_classes, bias=False)

#         self.x2 = nn.Sequential(self.features[:1])
#         self.x4 = nn.Sequential(self.features[1:2])
#         self.x8 = nn.Sequential(self.features[2:4])
#         self.x16 = nn.Sequential(self.features[4:6])
#         self.x32 = nn.Sequential(self.features[6:])

#         if pretrain_model:
#             print('use pretrain model {}'.format(pretrain_model))
#             self.init_weight(pretrain_model)
#         else:
#             self.init_params()

#     def init_weight(self, pretrain_model):
        
#         state_dict = torch.load(pretrain_model)["state_dict"]
#         self_state_dict = self.state_dict()
#         for k, v in state_dict.items():
#             self_state_dict.update({k: v})
#         self.load_state_dict(self_state_dict)

#     def init_params(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)

#     def _make_layers(self, base, layers, block_num, block):
#         features = []
#         features += [ConvX(3, base//2, 3, 2)]
#         features += [ConvX(base//2, base, 3, 2)]

#         for i, layer in enumerate(layers):
#             for j in range(layer):
#                 if i == 0 and j == 0:
#                     features.append(block(base, base*4, block_num, 2))
#                 elif j == 0:
#                     features.append(block(base*int(math.pow(2,i+1)), base*int(math.pow(2,i+2)), block_num, 2))
#                 else:
#                     features.append(block(base*int(math.pow(2,i+2)), base*int(math.pow(2,i+2)), block_num, 1))

#         return nn.Sequential(*features)

#     def forward(self, x):
#         feat2 = self.x2(x)
#         feat4 = self.x4(feat2)
#         feat8 = self.x8(feat4)
#         feat16 = self.x16(feat8)
#         feat32 = self.x32(feat16)
#         if self.use_conv_last:
#            feat32 = self.conv_last(feat32)

#         return feat2, feat4, feat8, feat16, feat32

#     def forward_impl(self, x):
#         out = self.features(x)
#         out = self.conv_last(out).pow(2)
#         out = self.gap(out).flatten(1)
#         out = self.fc(out)
#         # out = self.bn(out)
#         out = self.relu(out)
#         # out = self.relu(self.bn(self.fc(out)))
#         out = self.dropout(out)
#         out = self.linear(out)
#         return out

# if __name__ == "__main__":
#     model = STDCNet813(num_classes=1000, dropout=0.00, block_num=4)
#     model.eval()
#     x = torch.randn(1,3,224,224)
#     y = model(x)
#     torch.save(model.state_dict(), 'cat.pth')
#     print(y.size())
#######################################################################################第九次实验
# import torch
# import torch.nn as nn
# from torch.nn import init
# import math
# BatchNorm2d = nn.BatchNorm2d

# # class depthwise_separable_conv(nn.Module):
# #     def init(self, nin, nout):
# #         super(depthwise_separable_conv, self).init()
# #         self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin)
# #         self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)
# #     def forward(self, x):
# #         out = self.depthwise(x)
# #         out = self.pointwise(out)
# #         return out
# # class ConvBNReLU(nn.Module):
# #     def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
# #         super(ConvBNReLU, self).__init__()
# #         self.conv = nn.Conv2d(in_chan,
# #                 out_chan,
# #                 kernel_size = ks,
# #                 stride = stride,
# #                 padding = padding,
# #                 bias = False)
# #         # self.bn = BatchNorm2d(out_chan)
# #         self.bn = BatchNorm2d(out_chan)
# #         self.relu = nn.ReLU()
# # #         self.init_weight()

# #     def forward(self, x):
# #         x = self.conv(x)
# #         x = self.bn(x)
# #         x = self.relu(x)
# #         return x
# class ConvX(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel=3, stride=1):
#         super(ConvX, self).__init__()
#         self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=kernel//2, bias=False)
# #         self.depthwise = nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1, groups=in_planes)
# #         self.pointwise = nn.Conv2d(in_planes, out_planes, kernel_size=1)
#         self.bn = nn.BatchNorm2d(out_planes)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         out = self.relu(self.bn(self.conv(x)))
# #         out = self.relu(self.bn(self.pointwise(self.depthwise(x))))
#         return out


# class AddBottleneck(nn.Module):
#     def __init__(self, in_planes, out_planes, block_num=3, stride=1):
#         super(AddBottleneck, self).__init__()
#         assert block_num > 1, print("block number should be larger than 1.")
#         self.conv_list = nn.ModuleList()
#         self.stride = stride
#         if stride == 2:
#             self.avd_layer = nn.Sequential(
#                 nn.Conv2d(out_planes//2, out_planes//2, kernel_size=3, stride=2, padding=1, groups=out_planes//2, bias=False),
#                 nn.BatchNorm2d(out_planes//2),
#             )
#             self.skip = nn.Sequential(
#                 nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=2, padding=1, groups=in_planes, bias=False),
#                 nn.BatchNorm2d(in_planes),
#                 nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
#                 nn.BatchNorm2d(out_planes),
#             )
#             stride = 1

#         for idx in range(block_num):
#             if idx == 0:
#                 self.conv_list.append(ConvX(in_planes, out_planes//2, kernel=1))
#             elif idx == 1 and block_num == 2:
#                 self.conv_list.append(ConvX(out_planes//2, out_planes//2, stride=stride))
#             elif idx == 1 and block_num > 2:
#                 self.conv_list.append(ConvX(out_planes//2, out_planes//4, stride=stride))
#             elif idx < block_num - 1:
#                 self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx+1))))
#             else:
#                 self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx))))
            
#     def forward(self, x):
#         out_list = []
#         out = x

#         for idx, conv in enumerate(self.conv_list):
#             if idx == 0 and self.stride == 2:
#                 out = self.avd_layer(conv(out))
#             else:
#                 out = conv(out)
#             out_list.append(out)

#         if self.stride == 2:
#             x = self.skip(x)

#         return torch.cat(out_list, dim=1) + x

# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
           
#         self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
#                                nn.ReLU(),
#                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.fc(self.avg_pool(x))
#         max_out = self.fc(self.max_pool(x))
#         out = avg_out + max_out
#         return self.sigmoid(out)

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

# class CatBottleneck(nn.Module):
#     def __init__(self, in_planes, out_planes, block_num=3, stride=1):
#         super(CatBottleneck, self).__init__()
#         assert block_num > 1, print("block number should be larger than 1.")
#         self.conv_list = nn.ModuleList()
#         self.stride = stride
#         self.atenchannal2 = SpatialAttention()
#         if stride == 2:
#             self.avd_layer = nn.Sequential(
#                 nn.Conv2d(out_planes//2, out_planes//2, kernel_size=3, stride=2, padding=1, groups=out_planes//2, bias=False),
#                 nn.BatchNorm2d(out_planes//2),
#             )
#             self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
#             stride = 1
#         self.avg = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
# #         self.conavg_layer = nn.Sequential(
# #             nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=1, bias=False),
# #         nn.BatchNorm2d(out_planes))
        

#         for idx in range(block_num):
#             if idx == 0:
#                 self.conv_list.append(ConvX(in_planes, out_planes//2, kernel=1))
# #                 self.conv_list.append()
# #                 self.conv_list.append(ChannelAttention(out_planes//2))
#             elif idx == 1 and block_num == 2:
#                 self.conv_list.append(ConvX(out_planes//2, out_planes//2, stride=stride))

# #                 self.conv_list.append(ChannelAttention(out_planes//2))
#             elif idx == 1 and block_num > 2:
#                 self.conv_list.append(ConvX(out_planes//2, out_planes//4, stride=stride))

# #                 self.conv_list.append(ChannelAttention(out_planes//4))
#             elif idx < block_num - 1:
#                 self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx+1))))

# #                 self.conv_list.append(ChannelAttention(out_planes//int(math.pow(2, idx+1))))
#             else:
#                 self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx))))
              
# #                 self.conv_list.append(ChannelAttention(out_planes//int(math.pow(2, idx))))
                                      
            
#     def forward(self, x):
#         out_list = []
#         out1 = self.conv_list[0](x)
# #         out_atten = self.atenchannal2(out1)
# #         if self.stride==2:
# #             x = self.avg(x)
# #         out_temp = self.conavg_layer(x)
        
        
# #         print("out_temp:")
# #         print(out_temp.size())

#         for idx, conv in enumerate(self.conv_list[1:]):
#             if idx == 0:
#                 if self.stride == 2:
#                     out = conv(self.avd_layer(out1))
# #                     print("out2:")
# #                     print(out.size())
# #                     out = self.conv_list[2*idx+3](out1)
# #                     print("out2:")
# #                     print(out.size())
#                 else:
#                     out = conv(out1)
# #                     out = self.conv_list[2*idx+3](out1)+out
# #                     print("out2:")
# #                     print(out.size())
#             else:
#                 out = conv(out)
# #                 out = atten + out
# #                 out = (self.conv_list[2*idx+3](out))+out_temp
# #                 print("out:")
# #                 print(out.size())
#             out_list.append(out)

#         if self.stride == 2:
#             out1 = self.skip(out1)
#         out_list.insert(0, out1)

#         out = torch.cat(out_list, dim=1)
# #         print("output:")
# #         print(out.size())
# #         out = out+out_temp
#         out = self.atenchannal2(out)*out
# #         out = outlist[]
#         return out

# #STDC2Net
# class STDCNet1446(nn.Module):
#     def __init__(self, base=64, layers=[6,7,5], block_num=4, type="cat", num_classes=1000, dropout=0.20, pretrain_model='', use_conv_last=False):
#         super(STDCNet1446, self).__init__()
#         if type == "cat":
#             block = CatBottleneck
#         elif type == "add":
#             block = AddBottleneck
#         self.use_conv_last = use_conv_last
#         self.features = self._make_layers(base, layers, block_num, block)
#         self.conv_last = ConvX(base*16, max(1024, base*16), 1, 1)
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(max(1024, base*16), max(1024, base*16), bias=False)
#         self.bn = nn.BatchNorm1d(max(1024, base*16))
#         self.relu = nn.ReLU(inplace=True)
#         self.dropout = nn.Dropout(p=dropout)
#         self.linear = nn.Linear(max(1024, base*16), num_classes, bias=False)
# #         self.trans4 = ConvX(base,base*2,3,2)
# #         self.trans8 = ConvX(base*2,base*4,3,2)
# #         self.trans16 = ConvX(base*4,base*8,3,2)
# #         self.concatfuse8 = ConvX(base*4, base*4)
# #         self.concatfuse16 = ConvX(base*8, base*8)
# #         self.concatfuse32 = ConvX(base*16, base*16)

#         self.x2 = nn.Sequential(self.features[:1])
#         self.x4 = nn.Sequential(self.features[1:2])
#         self.x8 = nn.Sequential(self.features[2:6])
#         self.t256 = nn.Sequential(self.features[6])
#         self.addlayer7 = nn.Sequential(self.features[7])
#         self.x16 = nn.Sequential(self.features[8:13])
#         self.t512 = nn.Sequential(self.features[13])
#         self.addlayer14 = nn.Sequential(self.features[14])
#         self.x32 = nn.Sequential(self.features[15:18])
#         self.t1024 = nn.Sequential(self.features[18])
#         self.addlayer19 = nn.Sequential(self.features[19])
        

#         if pretrain_model:
# #             print('use pretrain model {}'.format(pretrain_model))
# #             self.init_weight(pretrain_model)
# #         else:
#             self.init_params()

#     def init_weight(self, pretrain_model):
        
#         state_dict = torch.load(pretrain_model)["state_dict"]
#         self_state_dict = self.state_dict()
#         for k, v in state_dict.items():
#             self_state_dict.update({k: v})
#         self.load_state_dict(self_state_dict)

#     def init_params(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)

#     def _make_layers(self, base, layers, block_num, block):
#         features = []
#         features += [ConvX(3, base//2, 3, 2)]
#         features += [ConvX(base//2, base, 3, 2)]

#         for i, layer in enumerate(layers):
#             for j in range(layer):
#                 if i == 0 and j == 0:
#                     features.append(block(base, base*2, block_num, 2))
#                 elif j == 0:
#                     features.append(block(base*int(math.pow(2,i)), base*int(math.pow(2,i+1)), block_num, 2))
#                 elif j+2==layer:
#                     features.append(block(base*int(math.pow(2,i+1)),base*int(math.pow(2,i+2))))
#                 elif j+2>layer:
#                     features.append(block(base*int(math.pow(2,i+2)),base*int(math.pow(2,i+2))))
#                 else:
#                     features.append(block(base*int(math.pow(2,i+1)), base*int(math.pow(2,i+1)), block_num, 1))
# #             features.append(block(base*int(math.pow(2,i+1)),base*int(math.pow(2,i+2))))

#         return nn.Sequential(*features)

#     def forward(self, x):
#         feat2 = self.x2(x)
#         feat4 = self.x4(feat2)
# #         print("feat4:")
# #         print(feat4.size())
        
# #         trans4 = self.trans4(feat4)
        
# #         feat8_temp = self.x8(feat4)
#         feat8tmp = self.x8(feat4)
#         feat8 = self.addlayer7(self.t256(feat8tmp))
# #         feat8 = torch.cat((trans4,feat8_temp),dim=1)
# #         feat8 = self.concatfuse8(feat8)
# #         print("feat8:")
# #         print(feat8.size())
        
# #         trans8 = self.trans8(feat8_temp)
        
#         feat16tmp = self.x16(feat8tmp)
# #         print("feat16tmp:")
# #         print(feat16tmp.size())
#         feat16 = self.addlayer14(self.t512(feat16tmp))
# #         feat16_temp = self.x16(feat8_temp)
# #         feat16 = torch.cat((trans8,feat16_temp),dim=1)
# #         feat16 = self.concatfuse16(feat16)
# #         print("feat16:")
# #         print(feat16.size())
        
# #         trans16 = self.trans16(feat16_temp)
        
#         feat32tmp = self.x32(feat16tmp)
#         feat32 = self.addlayer19(self.t1024(feat32tmp))
# #         feat32 = torch.cat((trans16,feat32_temp),dim=1)
# #         feat32 = self.concatfuse32(feat32)
# #         print("feat32:")
# #         print(feat32.size())
        
#         if self.use_conv_last:
#            feat32 = self.conv_last(feat32)

#         return feat2, feat4, feat8, feat16, feat32

#     def forward_impl(self, x):
#         out = self.features(x)
#         out = self.conv_last(out).pow(2)
#         out = self.gap(out).flatten(1)
#         out = self.fc(out)
#         # out = self.bn(out)
#         out = self.relu(out)
#         # out = self.relu(self.bn(self.fc(out)))
#         out = self.dropout(out)
#         out = self.linear(out)
#         return out

# # STDC1Net
# class STDCNet813(nn.Module):
#     def __init__(self, base=64, layers=[2,2,2], block_num=4, type="cat", num_classes=1000, dropout=0.20, pretrain_model='', use_conv_last=False):
#         super(STDCNet813, self).__init__()
#         if type == "cat":
#             block = CatBottleneck
#         elif type == "add":
#             block = AddBottleneck
#         self.use_conv_last = use_conv_last
#         self.features = self._make_layers(base, layers, block_num, block)
#         self.conv_last = ConvX(base*16, max(1024, base*16), 1, 1)
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(max(1024, base*16), max(1024, base*16), bias=False)
#         self.bn = nn.BatchNorm1d(max(1024, base*16))
#         self.relu = nn.ReLU(inplace=True)
#         self.dropout = nn.Dropout(p=dropout)
#         self.linear = nn.Linear(max(1024, base*16), num_classes, bias=False)

#         self.x2 = nn.Sequential(self.features[:1])
#         self.x4 = nn.Sequential(self.features[1:2])
#         self.x8 = nn.Sequential(self.features[2:4])
#         self.x16 = nn.Sequential(self.features[4:6])
#         self.x32 = nn.Sequential(self.features[6:])

#         if pretrain_model:
#             print('use pretrain model {}'.format(pretrain_model))
#             self.init_weight(pretrain_model)
#         else:
#             self.init_params()

#     def init_weight(self, pretrain_model):
        
#         state_dict = torch.load(pretrain_model)["state_dict"]
#         self_state_dict = self.state_dict()
#         for k, v in state_dict.items():
#             self_state_dict.update({k: v})
#         self.load_state_dict(self_state_dict)

#     def init_params(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)

#     def _make_layers(self, base, layers, block_num, block):
#         features = []
#         features += [ConvX(3, base//2, 3, 2)]
#         features += [ConvX(base//2, base, 3, 2)]

#         for i, layer in enumerate(layers):
#             for j in range(layer):
#                 if i == 0 and j == 0:
#                     features.append(block(base, base*4, block_num, 2))
#                 elif j == 0:
#                     features.append(block(base*int(math.pow(2,i+1)), base*int(math.pow(2,i+2)), block_num, 2))
#                 else:
#                     features.append(block(base*int(math.pow(2,i+2)), base*int(math.pow(2,i+2)), block_num, 1))

#         return nn.Sequential(*features)

#     def forward(self, x):
#         feat2 = self.x2(x)
#         feat4 = self.x4(feat2)
#         feat8 = self.x8(feat4)
#         feat16 = self.x16(feat8)
#         feat32 = self.x32(feat16)
#         if self.use_conv_last:
#            feat32 = self.conv_last(feat32)

#         return feat2, feat4, feat8, feat16, feat32

#     def forward_impl(self, x):
#         out = self.features(x)
#         out = self.conv_last(out).pow(2)
#         out = self.gap(out).flatten(1)
#         out = self.fc(out)
#         # out = self.bn(out)
#         out = self.relu(out)
#         # out = self.relu(self.bn(self.fc(out)))
#         out = self.dropout(out)
#         out = self.linear(out)
#         return out

# if __name__ == "__main__":
#     model = STDCNet813(num_classes=1000, dropout=0.00, block_num=4)
#     model.eval()
#     x = torch.randn(1,3,224,224)
#     y = model(x)
#     torch.save(model.state_dict(), 'cat.pth')
#     print(y.size())
#############################################################################################77.8的实验
import torch
import torch.nn as nn
from torch.nn import init
import math
BatchNorm2d = nn.BatchNorm2d

# class depthwise_separable_conv(nn.Module):
#     def init(self, nin, nout):
#         super(depthwise_separable_conv, self).init()
#         self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin)
#         self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)
#     def forward(self, x):
#         out = self.depthwise(x)
#         out = self.pointwise(out)
#         return out
# class ConvBNReLU(nn.Module):
#     def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
#         super(ConvBNReLU, self).__init__()
#         self.conv = nn.Conv2d(in_chan,
#                 out_chan,
#                 kernel_size = ks,
#                 stride = stride,
#                 padding = padding,
#                 bias = False)
#         # self.bn = BatchNorm2d(out_chan)
#         self.bn = BatchNorm2d(out_chan)
#         self.relu = nn.ReLU()
# #         self.init_weight()

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         return x
class ConvX(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1):
        super(ConvX, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=kernel//2, bias=False)
#         self.depthwise = nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1, groups=in_planes)
#         self.pointwise = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
#         out = self.relu(self.bn(self.pointwise(self.depthwise(x))))
        return out


class AddBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super(AddBottleneck, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(out_planes//2, out_planes//2, kernel_size=3, stride=2, padding=1, groups=out_planes//2, bias=False),
                nn.BatchNorm2d(out_planes//2),
            )
            self.skip = nn.Sequential(
                nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=2, padding=1, groups=in_planes, bias=False),
                nn.BatchNorm2d(in_planes),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_planes),
            )
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(ConvX(in_planes, out_planes//2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(ConvX(out_planes//2, out_planes//2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(ConvX(out_planes//2, out_planes//4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx+1))))
            else:
                self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx))))
            
    def forward(self, x):
        out_list = []
        out = x

        for idx, conv in enumerate(self.conv_list):
            if idx == 0 and self.stride == 2:
                out = self.avd_layer(conv(out))
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            x = self.skip(x)

        return torch.cat(out_list, dim=1) + x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

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
class channal_shuffle(nn.Module):
    def __init__(self,groups=4):
        super(channal_shuffle, self).__init__()
        self.groups = groups
    def forward(self, x):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // self.groups
        # grouping, 通道分组
        # b, num_channels, h, w =======>  b, groups, channels_per_group, h, w
        x = x.view(batchsize, self.groups, channels_per_group, height, width)
        # channel shuffle, 通道洗牌
        x = torch.transpose(x, 1, 2).contiguous()
        # x.shape=(batchsize, channels_per_group, groups, height, width)
        # flatten
        x = x.view(batchsize, -1, height, width)
        return x
class CatBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super(CatBottleneck, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
#         self.atenchannal1 = ChannelAttention()
#         self.atenchannal2 = SpatialAttention(1024)
        self.channal_shuffle = channal_shuffle(4)
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(out_planes//2, out_planes//2, kernel_size=3, stride=2, padding=1, groups=out_planes//2, bias=False),
                nn.BatchNorm2d(out_planes//2),
            )
            self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            stride = 1
        self.avg = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
#         self.conavg_layer = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0,bias=False)
#         self.conavg_layer_BN = nn.BatchNorm2d(out_planes)

#         self.conavg_layer = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0,bias=False),
#                                           nn.BatchNorm2d(out_planes))

        self.conavg_layer = nn.Sequential(nn.Conv2d(in_planes, out_planes//4, kernel_size=1, stride=1, padding=0,bias=False),
                                          nn.BatchNorm2d(out_planes//4),
#                                           nn.ReLU(),
                                          nn.Conv2d(out_planes//4, out_planes//4, kernel_size=3, stride=1, padding=1,bias=False),
                                         nn.BatchNorm2d(out_planes//4),
#                                           nn.ReLU(),
                                         nn.Conv2d(out_planes//4, out_planes, kernel_size=1, stride=1, padding=0,bias=False),
                                         nn.BatchNorm2d(out_planes),nn.Sigmoid())
    
#         self.channal_link = nn.Sequential(nn.Conv2d(out_planes, out_planes, kernel_size=1, stride=1, padding=0,bias=False),
#                                           nn.BatchNorm2d(out_planes))
        

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(ConvX(in_planes, out_planes//2, kernel=1))
#                 self.conv_list.append()
#                 self.conv_list.append(ChannelAttention(out_planes//2))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(ConvX(out_planes//2, out_planes//2, stride=stride))

#                 self.conv_list.append(ChannelAttention(out_planes//2))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(ConvX(out_planes//2, out_planes//4, stride=stride))

#                 self.conv_list.append(ChannelAttention(out_planes//4))
            elif idx < block_num - 1:
                self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx+1))))

#                 self.conv_list.append(ChannelAttention(out_planes//int(math.pow(2, idx+1))))
            else:
                self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx))))
              
#                 self.conv_list.append(ChannelAttention(out_planes//int(math.pow(2, idx))))
                                      
    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)
#         out_atten = self.atenchannal2(out1)
        if self.stride==2:
            x = self.avg(x)
#         out_temp = self.atenchannal2(self.conavg_layer(x))
        out_temp = self.conavg_layer(x)
#         out_temp = self.atenchannal2(out_temp)*out_temp
#         out_temp = self.conavg_layer_BN(out_temp)
        
        
#         print("out_temp:")
#         print(out_temp.size())

        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
#                     print("out2:")
#                     print(out.size())
#                     out = self.conv_list[2*idx+3](out1)
#                     print("out2:")
#                     print(out.size())
                else:
                    out = conv(out1)
#                     out = self.conv_list[2*idx+3](out1)+out
#                     print("out2:")
#                     print(out.size())
            else:
                out = conv(out)
#                 out = atten + out
#                 out = (self.conv_list[2*idx+3](out))+out_temp
#                 print("out:")
#                 print(out.size())
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)

        out = torch.cat(out_list, dim=1)
#         print("output:")
#         print(out.size())
#         out = self.channal_link(out)
        out = out+out_temp
#         out = self.channal_shuffle(out)
#         out = self.atenchannal2(out)*out
#         out = outlist[]
        return out

#STDC2Net
class STDCNet1446(nn.Module):
    def __init__(self, base=64, layers=[3,4,2], block_num=4, type="cat", num_classes=1000, dropout=0.20, pretrain_model='', use_conv_last=False):
        super(STDCNet1446, self).__init__()
        if type == "cat":
            block = CatBottleneck
        elif type == "add":
            block = AddBottleneck
        self.use_conv_last = use_conv_last
        self.features = self._make_layers(base, layers, block_num, block)
        self.conv_last = ConvX(base*16, max(1024, base*16), 1, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(max(1024, base*16), max(1024, base*16), bias=False)
        self.bn = nn.BatchNorm1d(max(1024, base*16))
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(max(1024, base*16), num_classes, bias=False)
#         self.atenchannal1 = ChannelAttention(1024)
#         self.atenchannal2 = SpatialAttention()
#         self.atenchannal2_3 = ChannelAttention(256)
#         self.atenchannal2_4 = ChannelAttention(512)
#         self.atenchannal2_5 = ChannelAttention(1024)
#         self.spatten = SpatialAttention()
#         self.trans4 = ConvX(base,base*2,3,2)
#         self.trans8 = ConvX(base*2,base*4,3,2)
#         self.trans16 = ConvX(base*4,base*8,3,2)
#         self.concatfuse8 = ConvX(base*4, base*4)
#         self.concatfuse16 = ConvX(base*8, base*8)
#         self.concatfuse32 = ConvX(base*16, base*16)

#         self.x2 = nn.Sequential(self.features[:1])
#         self.x4 = nn.Sequential(self.features[1:2])
#         self.x8 = nn.Sequential(self.features[2:6])
#         self.t256 = nn.Sequential(self.features[6])
#         self.addlayer7 = nn.Sequential(self.features[7])
#         self.x16 = nn.Sequential(self.features[8:13])
#         self.t512 = nn.Sequential(self.features[13])
#         self.addlayer14 = nn.Sequential(self.features[14])
#         self.x32 = nn.Sequential(self.features[15:18])
#         self.t1024 = nn.Sequential(self.features[18])
#         self.addlayer19 = nn.Sequential(self.features[19])

#         self.x2 = nn.Sequential(self.features[:1])
#         self.x4 = nn.Sequential(self.features[1:2])
#         self.x8 = nn.Sequential(self.features[2:7])
#         self.t256 = nn.Sequential(self.features[7])
#         self.addlayer7 = nn.Sequential(self.features[8])
#         self.x16 = nn.Sequential(self.features[9:15])
#         self.t512 = nn.Sequential(self.features[15])
#         self.addlayer14 = nn.Sequential(self.features[16])
#         self.x32 = nn.Sequential(self.features[17:21])
#         self.t1024 = nn.Sequential(self.features[21])
#         self.addlayer19 = nn.Sequential(self.features[22])

#         self.x2 = nn.Sequential(self.features[:1])
#         self.x4 = nn.Sequential(self.features[1:2])
#         self.x8 = nn.Sequential(self.features[2:6])
#         self.x16 = nn.Sequential(self.features[6:11])
#         self.x32 = nn.Sequential(self.features[11:])
        
#         self.x2 = nn.Sequential(self.features[:1])
#         self.x4 = nn.Sequential(self.features[1:2])
#         self.x8 = nn.Sequential(self.features[2:7])
#         self.x16 = nn.Sequential(self.features[7:13])
#         self.x32 = nn.Sequential(self.features[13:])

#         self.x2 = nn.Sequential(self.features[:1])
#         self.x4 = nn.Sequential(self.features[1:2])
#         self.x8 = nn.Sequential(self.features[2:6])
#         self.x16 = nn.Sequential(self.features[6:11])
#         self.x32 = nn.Sequential(self.features[11:])

        self.x2 = nn.Sequential(self.features[:1])
        self.x4 = nn.Sequential(self.features[1:2])
        self.x8 = nn.Sequential(self.features[2:5])
        self.x16 = nn.Sequential(self.features[5:9])
        self.x32 = nn.Sequential(self.features[9:])

#         self.x2 = nn.Sequential(self.features[:1])
#         self.x4 = nn.Sequential(self.features[1:2])
#         self.x8 = nn.Sequential(self.features[2:4])
#         self.x16 = nn.Sequential(self.features[4:6])
#         self.x32 = nn.Sequential(self.features[6:])
        

        if pretrain_model:
            print('use pretrain model {}'.format(pretrain_model))
            self.init_weight(pretrain_model)
        else:
            self.init_params()

    def init_weight(self, pretrain_model):
        
        state_dict = torch.load(pretrain_model)
        self_state_dict = self.state_dict()
        sub_list = list(state_dict.keys())[:296]
        for k, v in state_dict.items():
            if k in sub_list:
                k = k.replace('cp.backbone.','')
                self_state_dict.update({k: v})
        self.load_state_dict(self_state_dict)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_layers(self, base, layers, block_num, block):
        features = []
        features += [ConvX(3, base//2, 3, 2)]
        features += [ConvX(base//2, base, 3, 2)]

        for i, layer in enumerate(layers):
            for j in range(layer):
                if i == 0 and j == 0:
                    features.append(block(base, base*4, block_num, 2))
                elif j == 0:
                    features.append(block(base*int(math.pow(2,i+1)), base*int(math.pow(2,i+2)), block_num, 2))
#                 elif j+2==layer:
#                     features.append(block(base*int(math.pow(2,i+1)),base*int(math.pow(2,i+2))))
#                 elif j+2>layer:
#                     features.append(block(base*int(math.pow(2,i+2)),base*int(math.pow(2,i+2))))
                else:
                    features.append(block(base*int(math.pow(2,i+2)), base*int(math.pow(2,i+2)), block_num, 1))
#             features.append(block(base*int(math.pow(2,i+1)),base*int(math.pow(2,i+2))))

        return nn.Sequential(*features)

    def forward(self, x):
        feat2 = self.x2(x)
        feat4 = self.x4(feat2)
#         print("feat4:")
#         print(feat4.size())
        
#         trans4 = self.trans4(feat4)
        
#         feat8_temp = self.x8(feat4)
#         feat8tmp = self.x8(feat4)
        feat8 = self.x8(feat4)
#         feat8 = self.addlayer7(self.t256(feat8tmp))
#         feat8_ori = self.addlayer7(self.t256(feat8tmp))
#         feat8 = self.atenchannal2_3(feat8_ori)*feat8_ori
#         feat8 = self.spatten(feat8)*feat8
#         feat8 = feat8+feat8_ori
#         feat8 = self.t256(feat8tmp)
#         feat8 = torch.cat((trans4,feat8_temp),dim=1)
#         feat8 = self.concatfuse8(feat8)
#         print("feat8:")
#         print(feat8.size())
        
#         trans8 = self.trans8(feat8_temp)
        
#         feat16tmp = self.x16(feat8tmp)
        feat16 = self.x16(feat8)
#         print("feat16tmp:")
#         print(feat16tmp.size())
#         feat16 = self.addlayer14(self.t512(feat16tmp))
#         feat16_ori = self.addlayer14(self.t512(feat16tmp))
#         feat16 = self.atenchannal2_4(feat16_ori)*feat16_ori
#         feat16 = self.spatten(feat16)*feat16
#         feat16 = feat16+feat16_ori
#         feat16 = self.t512(feat16tmp)
#         feat16_temp = self.x16(feat8_temp)
#         feat16 = torch.cat((trans8,feat16_temp),dim=1)
#         feat16 = self.concatfuse16(feat16)
#         print("feat16:")
#         print(feat16.size())
        
#         trans16 = self.trans16(feat16_temp)
        
#         feat32tmp = self.x32(feat16tmp)
        feat32 = self.x32(feat16)
#         feat32 = self.atenchannal1(feat32)*feat32
#         feat32 = self.atenchannal2(feat32)*feat32
#         feat32 = self.addlayer19(self.t1024(feat32tmp))
#         feat32_ori = self.addlayer19(self.t1024(feat32tmp))
#         feat32 = self.atenchannal2_5(feat32_ori)*feat32_ori
#         feat32 = self.spatten(feat32)*feat32
#         feat32 = feat32+feat32_ori
#         feat32 = self.t1024(feat32tmp)
#         feat32 = torch.cat((trans16,feat32_temp),dim=1)
#         feat32 = self.concatfuse32(feat32)
#         print("feat32:")
#         print(feat32.size())
        
        if self.use_conv_last:
           feat32 = self.conv_last(feat32)

        return feat2, feat4, feat8, feat16, feat32

    def forward_impl(self, x):
        out = self.features(x)
        out = self.conv_last(out).pow(2)
        out = self.gap(out).flatten(1)
        out = self.fc(out)
        # out = self.bn(out)
        out = self.relu(out)
        # out = self.relu(self.bn(self.fc(out)))
        out = self.dropout(out)
        out = self.linear(out)
        return out

# STDC1Net
class STDCNet813(nn.Module):
    def __init__(self, base=64, layers=[2,2,2], block_num=4, type="cat", num_classes=1000, dropout=0.20, pretrain_model='', use_conv_last=False):
        super(STDCNet813, self).__init__()
        if type == "cat":
            block = CatBottleneck
        elif type == "add":
            block = AddBottleneck
        self.use_conv_last = use_conv_last
        self.features = self._make_layers(base, layers, block_num, block)
        self.conv_last = ConvX(base*16, max(1024, base*16), 1, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(max(1024, base*16), max(1024, base*16), bias=False)
        self.bn = nn.BatchNorm1d(max(1024, base*16))
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(max(1024, base*16), num_classes, bias=False)

        self.x2 = nn.Sequential(self.features[:1])
        self.x4 = nn.Sequential(self.features[1:2])
        self.x8 = nn.Sequential(self.features[2:4])
        self.x16 = nn.Sequential(self.features[4:6])
        self.x32 = nn.Sequential(self.features[6:])

        if pretrain_model:
            print('use pretrain model {}'.format(pretrain_model))
            self.init_weight(pretrain_model)
        else:
            self.init_params()

    def init_weight(self, pretrain_model):
        
        state_dict = torch.load(pretrain_model)["state_dict"]
        self_state_dict = self.state_dict()
        for k, v in state_dict.items():
            self_state_dict.update({k: v})
        self.load_state_dict(self_state_dict)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_layers(self, base, layers, block_num, block):
        features = []
        features += [ConvX(3, base//2, 3, 2)]
        features += [ConvX(base//2, base, 3, 2)]

        for i, layer in enumerate(layers):
            for j in range(layer):
                if i == 0 and j == 0:
                    features.append(block(base, base*4, block_num, 2))
                elif j == 0:
                    features.append(block(base*int(math.pow(2,i+1)), base*int(math.pow(2,i+2)), block_num, 2))
                else:
                    features.append(block(base*int(math.pow(2,i+2)), base*int(math.pow(2,i+2)), block_num, 1))

        return nn.Sequential(*features)

    def forward(self, x):
        feat2 = self.x2(x)
        feat4 = self.x4(feat2)
        feat8 = self.x8(feat4)
        feat16 = self.x16(feat8)
        feat32 = self.x32(feat16)
        if self.use_conv_last:
           feat32 = self.conv_last(feat32)

        return feat2, feat4, feat8, feat16, feat32

    def forward_impl(self, x):
        out = self.features(x)
        out = self.conv_last(out).pow(2)
        out = self.gap(out).flatten(1)
        out = self.fc(out)
        # out = self.bn(out)
        out = self.relu(out)
        # out = self.relu(self.bn(self.fc(out)))
        out = self.dropout(out)
        out = self.linear(out)
        return out

if __name__ == "__main__":
    model = STDCNet813(num_classes=1000, dropout=0.00, block_num=4)
    model.eval()
    x = torch.randn(1,3,224,224)
    y = model(x)
    torch.save(model.state_dict(), 'cat.pth')
    print(y.size())
