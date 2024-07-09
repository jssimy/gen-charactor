import torch
import torch.nn as  nn
import torch.nn.functional as F
import numpy as np

DEVICE =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPS = 1e-6
mse = nn.MSELoss()

def cosine_dismat(A, B):
    A = A.view(A.shape[0], A.shape[1], -1)
    B = B.view(B.shape[0], B.shape[1], -1)

    A_norm = torch.sqrt((A**2).sum(1))
    B_norm = torch.sqrt((B**2).sum(1))

    A = (A/A_norm.unsqueeze(dim=1).expand(A.shape)).permute(0,2,1)
    B = (B/B_norm.unsqueeze(dim=1).expand(B.shape))
    dismat = 1.-torch.bmm(A, B) 

    return dismat

def calc_remd_loss(A, B):
    C = cosine_dismat(A, B)
    m1, _ = C.min(1)
    m2, _ = C.min(2)
    
    remd = torch.max(m1.mean(), m2.mean())

    return remd

def calc_ss_loss(A, B):
    MA = cosine_dismat(A, A)
    MB = cosine_dismat(B, B)
    Lself_similarity = torch.abs(MA-MB).mean() 

    return Lself_similarity

def calc_moment_loss(A, B):
    A = A.view(A.shape[0], A.shape[1], -1)
    B = B.view(B.shape[0], B.shape[1], -1)

    mu_a = torch.mean(A, 1, keepdim=True)
    mu_b = torch.mean(B, 1, keepdim=True)
    mu_d = torch.abs(mu_a - mu_b).mean()

    A_c = A - mu_a
    B_c = B - mu_b
    cov_a = torch.bmm(A_c, A_c.permute(0,2,1)) / (A.shape[2]-1)
    cov_b = torch.bmm(B_c, B_c.permute(0,2,1)) / (B.shape[2]-1)
    cov_d = torch.abs(cov_a - cov_b).mean()
    loss = mu_d + cov_d
    return loss

def calc_mse_loss(A, B):
    return mse(A, B)

def calc_histogram_loss(A, B, histogram_block):
    input_hist = histogram_block(A)
    target_hist = histogram_block(B)
    histogram_loss = (1/np.sqrt(2.0) * (torch.sqrt(torch.sum(
        torch.pow(torch.sqrt(target_hist) - torch.sqrt(input_hist), 2)))) / 
        input_hist.shape[0])

    return histogram_loss
    
# B, C, H, W; mean var on HW
def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)    
    return normalized_feat


class RGBuvHistBlock(nn.Module):
    def __init__(self, h=64, insz=150, resizing='interpolation',
               method='inverse-quadratic', sigma=0.02, intensity_scale=True,
               device='cuda'):
        """ Computes the RGB-uv histogram feature of a given image.
        Args:
        h: histogram dimension size (scalar). The default value is 64.
        insz: maximum size of the input image; if it is larger than this size, the 
            image will be resized (scalar). Default value is 150 (i.e., 150 x 150 
            pixels).
        resizing: resizing method if applicable. Options are: 'interpolation' or 
            'sampling'. Default is 'interpolation'.
        method: the method used to count the number of pixels for each bin in the 
            histogram feature. Options are: 'thresholding', 'RBF' (radial basis 
            function), or 'inverse-quadratic'. Default value is 'inverse-quadratic'.
        sigma: if the method value is 'RBF' or 'inverse-quadratic', then this is 
            the sigma parameter of the kernel function. The default value is 0.02.
        intensity_scale: boolean variable to use the intensity scale (I_y in 
            Equation 2). Default value is True.

        Methods:
        forward: accepts input image and returns its histogram feature. Note that 
            unless the method is 'thresholding', this is a differentiable function 
            and can be easily integrated with the loss function. As mentioned in the
            paper, the 'inverse-quadratic' was found more stable than 'RBF' in our 
            training.
        """
        super(RGBuvHistBlock, self).__init__()
        self.h = h
        self.insz = insz
        self.device = device
        self.resizing = resizing
        self.method = method
        self.intensity_scale = intensity_scale
        if self.method == 'thresholding':
            self.eps = 6.0 / h
        else:
            self.sigma = sigma

    def forward(self, x):
        x = torch.clamp(x, 0, 1)
        if x.shape[2] > self.insz or x.shape[3] > self.insz:
            if self.resizing == 'interpolation':
                x_sampled = F.interpolate(x, size=(self.insz, self.insz),
                                    mode='bilinear', align_corners=False)
            elif self.resizing == 'sampling':
                inds_1 = torch.LongTensor(
                np.linspace(0, x.shape[2], self.h, endpoint=False)).to(
                device=self.device)
                inds_2 = torch.LongTensor(
                np.linspace(0, x.shape[3], self.h, endpoint=False)).to(
                device=self.device)
                x_sampled = x.index_select(2, inds_1)
                x_sampled = x_sampled.index_select(3, inds_2)
            else:
                raise Exception(
                f'Wrong resizing method. It should be: interpolation or sampling. '
                f'But the given value is {self.resizing}.')
        else:
            x_sampled = x

        L = x_sampled.shape[0]  # size of mini-batch
        if x_sampled.shape[1] > 3:
            x_sampled = x_sampled[:, :3, :, :]
        X = torch.unbind(x_sampled, dim=0)
        hists = torch.zeros((x_sampled.shape[0], 3, self.h, self.h)).to(
                                                      device=self.device)
        for l in range(L):
            I = torch.t(torch.reshape(X[l], (3, -1)))
            II = torch.pow(I, 2)
            if self.intensity_scale:
                Iy = torch.unsqueeze(torch.sqrt(II[:, 0] + II[:, 1] + II[:, 2] + EPS), 
                                    dim=1)
            else:
                Iy = 1

            Iu0 = torch.unsqueeze(torch.log(I[:, 0] + EPS) - torch.log(I[:, 1] + EPS),
                                    dim=1)
            Iv0 = torch.unsqueeze(torch.log(I[:, 0] + EPS) - torch.log(I[:, 2] + EPS),
                                    dim=1)
            diff_u0 = abs(
                Iu0 - torch.unsqueeze(torch.tensor(np.linspace(-3, 3, num=self.h)),
                                    dim=0).to(self.device))
            diff_v0 = abs(
                Iv0 - torch.unsqueeze(torch.tensor(np.linspace(-3, 3, num=self.h)),
                                    dim=0).to(self.device))
            if self.method == 'thresholding':
                diff_u0 = torch.reshape(diff_u0, (-1, self.h)) <= self.eps / 2
                diff_v0 = torch.reshape(diff_v0, (-1, self.h)) <= self.eps / 2
            elif self.method == 'RBF':
                diff_u0 = torch.pow(torch.reshape(diff_u0, (-1, self.h)),
                                    2) / self.sigma ** 2
                diff_v0 = torch.pow(torch.reshape(diff_v0, (-1, self.h)),
                                    2) / self.sigma ** 2
                diff_u0 = torch.exp(-diff_u0)  # Radial basis function
                diff_v0 = torch.exp(-diff_v0)
            elif self.method == 'inverse-quadratic':
                diff_u0 = torch.pow(torch.reshape(diff_u0, (-1, self.h)),
                                    2) / self.sigma ** 2
                diff_v0 = torch.pow(torch.reshape(diff_v0, (-1, self.h)),
                                    2) / self.sigma ** 2
                diff_u0 = 1 / (1 + diff_u0)  # Inverse quadratic
                diff_v0 = 1 / (1 + diff_v0)
            else:
                raise Exception(
                f'Wrong kernel method. It should be either thresholding, RBF,' 
                f' inverse-quadratic. But the given value is {self.method}.')
            diff_u0 = diff_u0.type(torch.float32)
            diff_v0 = diff_v0.type(torch.float32)
            a = torch.t(Iy * diff_u0)
            hists[l, 0, :, :] = torch.mm(a, diff_v0)

            Iu1 = torch.unsqueeze(torch.log(I[:, 1] + EPS) - torch.log(I[:, 0] + EPS),
                                    dim=1)
            Iv1 = torch.unsqueeze(torch.log(I[:, 1] + EPS) - torch.log(I[:, 2] + EPS),
                                    dim=1)
            diff_u1 = abs(
                Iu1 - torch.unsqueeze(torch.tensor(np.linspace(-3, 3, num=self.h)),
                                    dim=0).to(self.device))
            diff_v1 = abs(
                Iv1 - torch.unsqueeze(torch.tensor(np.linspace(-3, 3, num=self.h)),
                                    dim=0).to(self.device))

            if self.method == 'thresholding':
                diff_u1 = torch.reshape(diff_u1, (-1, self.h)) <= self.eps / 2
                diff_v1 = torch.reshape(diff_v1, (-1, self.h)) <= self.eps / 2
            elif self.method == 'RBF':
                diff_u1 = torch.pow(torch.reshape(diff_u1, (-1, self.h)),
                                    2) / self.sigma ** 2
                diff_v1 = torch.pow(torch.reshape(diff_v1, (-1, self.h)),
                                    2) / self.sigma ** 2
                diff_u1 = torch.exp(-diff_u1)  # Gaussian
                diff_v1 = torch.exp(-diff_v1)
            elif self.method == 'inverse-quadratic':
                diff_u1 = torch.pow(torch.reshape(diff_u1, (-1, self.h)),
                                    2) / self.sigma ** 2
                diff_v1 = torch.pow(torch.reshape(diff_v1, (-1, self.h)),
                                    2) / self.sigma ** 2
                diff_u1 = 1 / (1 + diff_u1)  # Inverse quadratic
                diff_v1 = 1 / (1 + diff_v1)

            diff_u1 = diff_u1.type(torch.float32)
            diff_v1 = diff_v1.type(torch.float32)
            a = torch.t(Iy * diff_u1)
            hists[l, 1, :, :] = torch.mm(a, diff_v1)

            Iu2 = torch.unsqueeze(torch.log(I[:, 2] + EPS) - torch.log(I[:, 0] + EPS),
                                    dim=1)
            Iv2 = torch.unsqueeze(torch.log(I[:, 2] + EPS) - torch.log(I[:, 1] + EPS),
                                    dim=1)
            diff_u2 = abs(
                Iu2 - torch.unsqueeze(torch.tensor(np.linspace(-3, 3, num=self.h)),
                                    dim=0).to(self.device))
            diff_v2 = abs(
                Iv2 - torch.unsqueeze(torch.tensor(np.linspace(-3, 3, num=self.h)),
                                    dim=0).to(self.device))
            if self.method == 'thresholding':
                diff_u2 = torch.reshape(diff_u2, (-1, self.h)) <= self.eps / 2
                diff_v2 = torch.reshape(diff_v2, (-1, self.h)) <= self.eps / 2
            elif self.method == 'RBF':
                diff_u2 = torch.pow(torch.reshape(diff_u2, (-1, self.h)),
                                    2) / self.sigma ** 2
                diff_v2 = torch.pow(torch.reshape(diff_v2, (-1, self.h)),
                                    2) / self.sigma ** 2
                diff_u2 = torch.exp(-diff_u2)  # Gaussian
                diff_v2 = torch.exp(-diff_v2)
            elif self.method == 'inverse-quadratic':
                diff_u2 = torch.pow(torch.reshape(diff_u2, (-1, self.h)),
                                    2) / self.sigma ** 2
                diff_v2 = torch.pow(torch.reshape(diff_v2, (-1, self.h)),
                                    2) / self.sigma ** 2
                diff_u2 = 1 / (1 + diff_u2)  # Inverse quadratic
                diff_v2 = 1 / (1 + diff_v2)
            diff_u2 = diff_u2.type(torch.float32)
            diff_v2 = diff_v2.type(torch.float32)
            a = torch.t(Iy * diff_u2)
            hists[l, 2, :, :] = torch.mm(a, diff_v2)

        # normalization
        hists_normalized = hists / (
            ((hists.sum(dim=1)).sum(dim=1)).sum(dim=1).view(-1, 1, 1, 1) + EPS)

        return hists_normalized

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.vgg = vgg19[:44]
        self.vgg.load_state_dict(torch.load(args.encoder_weights, map_location='cpu'), strict=False)
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.align1 = PAMA(512)
        self.align2 = PAMA(512)
        self.align3 = PAMA(512)

        self.decoder = decoder
        self.hist = RGBuvHistBlock(insz=64, h=256, 
                                   intensity_scale=True, 
                                   method='inverse-quadratic',
                                   device=DEVICE)

        if args.pretrained == True:
            self.align1.load_state_dict(torch.load(args.align1_weights, map_location='cpu'), strict=True)
            self.align2.load_state_dict(torch.load(args.align2_weights, map_location='cpu'), strict=True)
            self.align3.load_state_dict(torch.load(args.align3_weights, map_location='cpu'), strict=True)
            self.decoder.load_state_dict(torch.load(args.decoder_weights, map_location='cpu'), strict=False)

        if args.requires_grad == False:
            for param in self.parameters():
                param.requires_grad = False


    def forward(self, Ic, Is):
        feat_c = self.forward_vgg(Ic)
        feat_s = self.forward_vgg(Is)
        Fc, Fs = feat_c[3], feat_s[3]

        Fcs1 = self.align1(Fc, Fs)
        Fcs2 = self.align2(Fcs1, Fs)
        Fcs3 = self.align3(Fcs2, Fs)

        Ics3 = self.decoder(Fcs3)

        if self.args.training == True:
            Ics1 = self.decoder(Fcs1)
            Ics2 = self.decoder(Fcs2)
            Irc = self.decoder(Fc)
            Irs = self.decoder(Fs)
            feat_cs1 = self.forward_vgg(Ics1)
            feat_cs2 = self.forward_vgg(Ics2)
            feat_cs3 = self.forward_vgg(Ics3)
            feat_rc = self.forward_vgg(Irc)
            feat_rs = self.forward_vgg(Irs)

            content_loss1, remd_loss1, moment_loss1, color_loss1 = 0.0, 0.0, 0.0, 0.0
            content_loss2, remd_loss2, moment_loss2, color_loss2 = 0.0, 0.0, 0.0, 0.0
            content_loss3, remd_loss3, moment_loss3, color_loss3 = 0.0, 0.0, 0.0, 0.0
            loss_rec = 0.0

            for l in range(2, 5):     
                content_loss1 += self.args.w_content1 * calc_ss_loss(feat_cs1[l], feat_c[l])
                remd_loss1 += self.args.w_remd1 * calc_remd_loss(feat_cs1[l], feat_s[l])
                moment_loss1 += self.args.w_moment1 * calc_moment_loss(feat_cs1[l], feat_s[l])

                content_loss2 += self.args.w_content2 * calc_ss_loss(feat_cs2[l], feat_c[l])
                remd_loss2 += self.args.w_remd2 * calc_remd_loss(feat_cs2[l], feat_s[l])
                moment_loss2 += self.args.w_moment2 * calc_moment_loss(feat_cs2[l], feat_s[l])

                content_loss3 += self.args.w_content3 * calc_ss_loss(feat_cs3[l], feat_c[l])
                remd_loss3 += self.args.w_remd3 * calc_remd_loss(feat_cs3[l], feat_s[l])
                moment_loss3 += self.args.w_moment3 * calc_moment_loss(feat_cs3[l], feat_s[l])

                loss_rec += 0.5 * calc_mse_loss(feat_rc[l], feat_c[l]) + 0.5 * calc_mse_loss(feat_rs[l], feat_s[l])
            loss_rec += 25 * calc_mse_loss(Irc, Ic)
            loss_rec += 25 * calc_mse_loss(Irs, Is)

            if self.args.color_on:
                color_loss1 += self.args.w_color1 * calc_histogram_loss(Ics1, Is, self.hist)
                color_loss2 += self.args.w_color2 * calc_histogram_loss(Ics2, Is, self.hist)
                color_loss3 += self.args.w_color3 * calc_histogram_loss(Ics3, Is, self.hist)
            
            loss1 = (content_loss1+remd_loss1+moment_loss1+color_loss1)/(self.args.w_content1+self.args.w_remd1+self.args.w_moment1+self.args.w_color1)
            loss2 = (content_loss2+remd_loss2+moment_loss2+color_loss2)/(self.args.w_content2+self.args.w_remd2+self.args.w_moment2+self.args.w_color2)
            loss3 = (content_loss3+remd_loss3+moment_loss3+color_loss3)/(self.args.w_content3+self.args.w_remd3+self.args.w_moment3+self.args.w_color3)
            loss = loss1 + loss2 + loss3 + loss_rec
            return loss
        else: 
            return Ics3

    def forward_vgg(self, x):
        relu1_1 = self.vgg[:4](x)
        relu2_1 = self.vgg[4:11](relu1_1)
        relu3_1 = self.vgg[11:18](relu2_1)
        relu4_1 = self.vgg[18:31](relu3_1)
        relu5_1 = self.vgg[31:44](relu4_1)
        return [relu1_1, relu2_1, relu3_1, relu4_1, relu5_1]
    
    def save_ckpts(self):
        torch.save(self.align1.state_dict(), "./checkpoints/PAMA1.pth")
        torch.save(self.align2.state_dict(), "./checkpoints/PAMA2.pth")
        torch.save(self.align3.state_dict(), "./checkpoints/PAMA3.pth")
        torch.save(self.decoder.state_dict(), "./checkpoints/decoder.pth")  

#---------------------------------------------------------------------------------------------------------------

vgg19 = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, 
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

#---------------------------------------------------------------------------------------------------------------

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),  
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),  #relu4_1
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),  #relu3_1
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),  #relu2_1
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  #relu1_1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

#---------------------------------------------------------------------------------------------------------------

class AttentionUnit(nn.Module):
    def __init__(self, channels):
        super(AttentionUnit, self).__init__()
        self.relu6 = nn.ReLU6()
        self.f = nn.Conv2d(channels, channels//2, (1, 1))
        self.g = nn.Conv2d(channels, channels//2, (1, 1))
        self.h = nn.Conv2d(channels, channels//2, (1, 1))

        self.out_conv = nn.Conv2d(channels//2, channels, (1, 1))
        self.softmax = nn.Softmax(dim = -1)
        
    def forward(self, Fc, Fs):
        B, C, H, W = Fc.shape
        f_Fc = self.relu6(self.f(mean_variance_norm(Fc)))
        g_Fs = self.relu6(self.g(mean_variance_norm(Fs)))
        h_Fs = self.relu6(self.h(Fs))
        f_Fc = f_Fc.view(f_Fc.shape[0], f_Fc.shape[1], -1).permute(0, 2, 1)
        g_Fs = g_Fs.view(g_Fs.shape[0], g_Fs.shape[1], -1)

        Attention = self.softmax(torch.bmm(f_Fc, g_Fs))

        h_Fs = h_Fs.view(h_Fs.shape[0], h_Fs.shape[1], -1)

        Fcs = torch.bmm(h_Fs, Attention.permute(0, 2, 1))
        Fcs = Fcs.view(B, C//2, H, W)
        Fcs = self.relu6(self.out_conv(Fcs))

        return Fcs

class FuseUnit(nn.Module):
    def __init__(self, channels):
        super(FuseUnit, self).__init__()
        self.proj1 = nn.Conv2d(2*channels, channels, (1, 1))
        self.proj2 = nn.Conv2d(channels, channels, (1, 1))
        self.proj3 = nn.Conv2d(channels, channels, (1, 1))

        self.fuse1x = nn.Conv2d(channels, 1, (1, 1), stride = 1)
        self.fuse3x = nn.Conv2d(channels, 1, (3, 3), stride = 1)
        self.fuse5x = nn.Conv2d(channels, 1, (5, 5), stride = 1)

        self.pad3x = nn.ReflectionPad2d((1, 1, 1, 1))
        self.pad5x = nn.ReflectionPad2d((2, 2, 2, 2))
        self.sigmoid = nn.Sigmoid()

    def forward(self, F1, F2):
        Fcat = self.proj1(torch.cat((F1, F2), dim=1))
        F1 = self.proj2(F1)
        F2 = self.proj3(F2)
        
        fusion1 = self.sigmoid(self.fuse1x(Fcat))      
        fusion3 = self.sigmoid(self.fuse3x(self.pad3x(Fcat)))
        fusion5 = self.sigmoid(self.fuse5x(self.pad5x(Fcat)))
        fusion = (fusion1 + fusion3 + fusion5) / 3

        return torch.clamp(fusion, min=0, max=1.0)*F1 + torch.clamp(1 - fusion, min=0, max=1.0)*F2 
        
class PAMA(nn.Module):
    def __init__(self, channels):
        super(PAMA, self).__init__()
        self.conv_in = nn.Conv2d(channels, channels, (3, 3), stride=1)
        self.attn = AttentionUnit(channels)
        self.fuse = FuseUnit(channels)
        self.conv_out = nn.Conv2d(channels, channels, (3, 3), stride=1)

        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.relu6 = nn.ReLU6()
    
    def forward(self, Fc, Fs):
        Fc = self.relu6(self.conv_in(self.pad(Fc)))
        Fs = self.relu6(self.conv_in(self.pad(Fs)))
        Fcs = self.attn(Fc, Fs)
        Fcs = self.relu6(self.conv_out(self.pad(Fcs)))
        Fcs = self.fuse(Fc, Fcs)
        
        return Fcs
    
#---------------------------------------------------------------------------------------------------------------


