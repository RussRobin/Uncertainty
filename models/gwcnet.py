from __future__ import print_function
import torch.utils.data
from models.submodule import *
import math
import torch.nn.functional as F

# def pdf2disparity(pdf, bin_values):
#     """
#     # return disparity and error variance uncertainty from pdf
#     """

#     # p = pdf[0].numpy()
#     N, C, H, W = pdf.shape
#     bin_values = bin_values[None, :, None, None].repeat(N, 1, H, W).to(pdf.device)

#     # print('bin_values',bin_values.shape)
#     # print('pdf', pdf.shape)
#     mu = torch.sum(bin_values * pdf, dim=1).view(N, -1, H, W)
#     sigma2 = torch.sum(torch.square(bin_values - mu) * pdf, dim=1)
#     return mu.view(N, H, W), sigma2.view(N, H, W)

def pdf2disparity(pdf, t0s):
    """
    # return disparity and error variance uncertainty from pdf
    """

    # p = pdf[0].numpy()
    N, C, H, W = pdf.shape
    t0s = t0s[None, :, None, None].repeat(N, 1, H, W).to(pdf.device)

    mu = torch.sum(t0s * pdf, dim=1).view(N, -1, H, W)
    sigma2 = torch.sum(torch.square(t0s - mu) * pdf, dim=1)
    return mu.view(N, H, W), sigma2.view(N, H, W)

class feature_extraction(nn.Module):
    def __init__(self, concat_feature=False, concat_feature_channel=12):
        super(feature_extraction, self).__init__()
        self.concat_feature = concat_feature

        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)

        if self.concat_feature:
            self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                    bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.layer1(x)
        l2 = self.layer2(x)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        gwc_feature = torch.cat((l2, l3, l4), dim=1)

        if not self.concat_feature:
            return {"gwc_feature": gwc_feature}
        else:
            concat_feature = self.lastconv(gwc_feature)
            return {"gwc_feature": gwc_feature, "concat_feature": concat_feature}


class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))

        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        a = self.conv6(conv5)
        a = self.redir1(x)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)

        return conv6

class UncertaintyDecoder(nn.Module):
    def __init__(self, maxdisp, num_scale=4):
        super(UncertaintyDecoder, self).__init__()
        self.num_scale = num_scale
        self.maxdisp = maxdisp
        self.idx_list = self.index_combinations(self.num_scale)
        self.input_len = len(self.idx_list)
        self.fc1 = nn.Linear(self.input_len, self.input_len*2)
        self.fc2 = nn.Linear(self.input_len*2, self.input_len)
        self.fc3 = nn.Linear(self.input_len, 4)
        self.act = nn.ELU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)

    def index_combinations(self,num_scales):
        L = []
        for i in range(num_scales):
            for j in range(i + 1, num_scales):
                L.append((i, j))
        return L

    def forward(self,disp_list):
        assert len(disp_list) == self.num_scale, \
            "Expected disp predictions from each scales"
        feature_list = []
        for i,j in self.idx_list:
            disp1,disp2 = disp_list[i]/self.maxdisp,disp_list[j]/self.maxdisp  # (b,w,h)
            feature_list.append((disp1-disp2)**2)
        disp_var = torch.stack(feature_list,dim=0)  #(6,b,w,h)
        disp_var = disp_var.permute(1,2,3,0)
        out = self.fc1(disp_var.cuda())
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = out.permute(3,0,1,2)
        return out

class GwcNet(nn.Module):
    def __init__(self, maxdisp, use_concat_volume=False,est_uncert=False,ConOR=False,bin_values=None,bs=False,use_dropout=False,return_embedding=False):
        super(GwcNet, self).__init__()
        self.ConOR = ConOR
        self.bin_values = bin_values # CWX NOTE 这个实际上是t0s
        
        self.maxdisp = maxdisp
        self.use_concat_volume = use_concat_volume
        self.est_uncert = est_uncert

        self.use_dropout = use_dropout  # for MC dropout

        self.bs = bs # if true, the model is used in bootstrap

        self.num_groups = 40
        # return out3 (upsampled to img size) in conor testing mode. out3:[B,32，D/4,H/4,W/4] -> embedding:[B,32*D/4,H,W]
        self.return_embedding = return_embedding 

        if self.use_concat_volume:
            self.concat_channels = 12
            self.feature_extraction = feature_extraction(concat_feature=True,
                                                         concat_feature_channel=self.concat_channels)
        else:
            self.concat_channels = 0
            self.feature_extraction = feature_extraction(concat_feature=False)

        if self.use_dropout: # for MC dropout
            self.dropout = nn.Dropout(p=0.5)

        self.dres0 = nn.Sequential(convbn_3d(self.num_groups + self.concat_channels * 2, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classif0 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        ####################################################
        # add uncertainty decoder
        ####################################################
        if self.est_uncert:
            self.uncertdec = UncertaintyDecoder(maxdisp=self.maxdisp)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, left, right):
        features_left = self.feature_extraction(left)
        features_right = self.feature_extraction(right)
        
        if self.use_dropout: # MC Dropout
            features_left['gwc_feature'] = self.dropout(features_left['gwc_feature'])
            features_left['concat_feature'] = self.dropout(features_left['concat_feature'])
            features_right['gwc_feature'] = self.dropout(features_right['gwc_feature'])
            features_right['concat_feature'] = self.dropout(features_right['concat_feature'])


        gwc_volume = build_gwc_volume(features_left["gwc_feature"], features_right["gwc_feature"], self.maxdisp // 4,
                                      self.num_groups)
        if self.use_concat_volume:
            concat_volume = build_concat_volume(features_left["concat_feature"], features_right["concat_feature"],
                                                self.maxdisp // 4)
            volume = torch.cat((gwc_volume, concat_volume), 1)
        else:
            volume = gwc_volume

        cost0 = self.dres0(volume)
        if self.use_dropout: # MC Dropout
            cost0 = self.dropout(cost0)

        cost0 = self.dres1(cost0) + cost0
        out1 = self.dres2(cost0)
        out2 = self.dres3(out1)
        out3 = self.dres4(out2)

        if self.use_dropout: # MC Dropout
            out1 = self.dropout(out1)
            out2 = self.dropout(out2)
            out3 = self.dropout(out3)


        cost0 = self.classif0(cost0)
        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2)
        cost3 = self.classif3(out3)

        cost0 = F.upsample(cost0, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
        cost0 = torch.squeeze(cost0, 1)
        
        cost1 = F.upsample(cost1, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
        cost1 = torch.squeeze(cost1, 1)

        cost2 = F.upsample(cost2, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
        cost2 = torch.squeeze(cost2, 1)

        cost3 = F.upsample(cost3, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
        cost3 = torch.squeeze(cost3, 1)
        
        if self.ConOR: # our implementation of ConOR
            pdf0 = F.softmax(cost0, dim=1)
            pdf1 = F.softmax(cost1, dim=1)
            pdf2 = F.softmax(cost2, dim=1)
            pdf3 = F.softmax(cost3, dim=1)

            pred0, uncert0 = pdf2disparity(pdf0, self.bin_values)
            pred1, uncert1 = pdf2disparity(pdf1, self.bin_values)
            pred2, uncert2 = pdf2disparity(pdf2, self.bin_values)
            pred3, uncert3 = pdf2disparity(pdf3, self.bin_values)
            # CWX TODO
            # 训练bs的时候不需要uncert
            if self.bs:
                if self.training:
                    output = {'pdf':[pdf0, pdf1, pdf2, pdf3],
                            'disp':[pred0, pred1, pred2, pred3],
                            # 'uncert':[uncert0, uncert1, uncert2, uncert3]
                            }
                else:
                    output = {'pdf':[pdf3],
                            'disp':[pred3],
                            'cost': [cost3],
                            # 'uncert':[uncert3]
                            }
            else:
                if self.training:
                    output = {'pdf':[pdf0, pdf1, pdf2, pdf3],
                            'disp':[pred0, pred1, pred2, pred3],
                            'uncert':[uncert0, uncert1, uncert2, uncert3]
                            }
                else:
                    if self.return_embedding:
                        embedding = out3.view(out3.shape[0], -1, out3.shape[3], out3.shape[4])
                        new_shape = (embedding.shape[0], embedding.shape[1], embedding.shape[2] * 4, embedding.shape[3] * 4)
                        embedding = F.interpolate(embedding, size=(new_shape[2], new_shape[3]), mode='bilinear')
                        output = {'pdf':[pdf3],
                            'disp':[pred3],
                            'cost': [cost3],
                            'uncert':[uncert3],
                            'embedding':[embedding]
                            }
                        del embedding
                    else:
                        output = {'pdf':[pdf3],
                            'disp':[pred3],
                            'cost': [cost3],
                            'uncert':[uncert3]
                            }
            del cost0,cost1,cost2,cost3,pred0,pred1,pred2,pred3,pdf0,pdf1,pdf2,pdf3,uncert0,uncert1,uncert2,uncert3

        else:
            pred0 = F.softmax(cost0, dim=1) # SIZE(2,192,256,512)
            pred0 = disparity_regression(pred0, self.maxdisp) #size(2,256,512)

            pred1 = F.softmax(cost1, dim=1)
            pred1 = disparity_regression(pred1, self.maxdisp)

            pred2 = F.softmax(cost2, dim=1)
            pred2 = disparity_regression(pred2, self.maxdisp)

            pred3 = F.softmax(cost3, dim=1)
            pdf3 = F.softmax(cost3, dim=1)
            pred3 = disparity_regression(pred3, self.maxdisp) # disparity estimation = sum(d* sigma(-C_d))

            if self.est_uncert:
                uncert = self.uncertdec([pred0,pred1,pred2,pred3])

                if self.training:
                    output = {'disp':[pred0, pred1, pred2, pred3],
                            'uncert':[uncert[0],uncert[1],uncert[2],uncert[3]]}
                else:
                    output = {'disp': [pred3],
                            'uncert':[uncert[3]],
                            'cost': [cost3],
                            'pdf': [pdf3]
                            }

            else:
                if self.training:
                    output = {'disp': [pred0, pred1, pred2, pred3]}
                else:
                    output = {'disp': [pred3],
                            'cost': [cost3],
                            'pdf': [pdf3]}

            del cost0,cost1,cost2,cost3,pred0,pred1,pred2,pred3
        return output

def GwcNet_G(d,bootstrap=False):
    return GwcNet(d, use_concat_volume=False,est_uncert=False,ConOR=False,bs=bootstrap)

def GwcNet_GC(d,bootstrap=False):
    return GwcNet(d, use_concat_volume=True,est_uncert=False,ConOR=False,bs=bootstrap)

def GwcNet_GCS(d,bootstrap=False):
    return GwcNet(d, use_concat_volume=True,est_uncert=True,ConOR=False,bs=bootstrap)

def GwcNet_conor(d,bin_value,bootstrap=False,use_dropout=False,return_embedding=False):
    """
    # GwcNet with Constrained Ordinal Regression
    """
    return GwcNet(d, use_concat_volume=True,ConOR=True,bin_values=bin_value,bs=bootstrap,use_dropout=use_dropout,return_embedding=return_embedding)