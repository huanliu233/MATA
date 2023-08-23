import torch
from torch import nn
import torch.nn.functional as F
from datasetsA import weights_init



class Conv2dBlock(nn.Module): # padding 卷积+norm+激活 
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero',  use_bias = True):
        super(Conv2dBlock, self).__init__()
        self.use_bias = use_bias
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'softmax':
            self.activation == nn.Softmax()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution

        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu',use_bias = True):
        super(LinearBlock, self).__init__()
        
        # initialize fully connected layer
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = nn.LayerNorm(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'none' :
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'softmax':
            self.activation = nn.Softmax()       
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out

class ResBlock(nn.Module): #双卷积+ 自身
    def __init__(self, dim, norm='bn', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        out = torch.relu(out)
        return out

class ResBlocks(nn.Module): # 
    def __init__(self, num_blocks, dim, norm='bn', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)



class MTAttention_v2(nn.Module):
    def __init__(self, inputdim, base_dim,heads):
        super(MTAttention_v2, self).__init__()
        self.v = nn.Linear(inputdim,heads*base_dim)
        self.v_norm = nn.BatchNorm1d(heads*base_dim)
        self.v_act = nn.ReLU()
        self.k = nn.Linear(inputdim,heads*base_dim)
        self.k_norm = nn.BatchNorm1d(heads*base_dim)
        self.heads = heads
        self.scale = base_dim**(-0.5)
        self.v.apply(weights_init('kaiming'))
        self.v_norm.apply(weights_init('kaiming'))      
        self.k.apply(weights_init('xavier'))
        self.k_norm.apply(weights_init('kaiming')) 
        self.dim = heads*base_dim
    def forward(self, x):
        B, N, _= x.shape
        C = self.dim
        H = round(N**(0.5))
        v = self.v_act(self.v_norm(self.v(x).view(-1,C))).reshape(B,N,C).permute(0,2,1).view(B,C,H,H)

        k = self.k_norm(self.k(x).view(-1,C)).reshape(B,N,self.heads,C//self.heads).permute(0,2,1,3)
        q = k[:,:,N//2,:].unsqueeze(2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1).repeat(1,1,C//self.heads,1).view(B,C,H,H)
        x = attn * v 
        eps = 1e-16
        y = x[:,:,H//2,H//2].unsqueeze(1)
        for i in range(1,H//2+1):
            y_temp = F.adaptive_avg_pool2d(x[:,:,H//2-i:H//2+i+1,H//2-i:H//2+i+1],1) \
                        / (F.adaptive_avg_pool2d(attn[:,:,H//2-i:H//2+i+1,H//2-i:H//2+i+1],1) + eps)
            y = torch.cat( (y,y_temp[:,None,:].squeeze(-1).squeeze(-1)),dim=1)

            y_temp = F.adaptive_avg_pool2d(x[:,:,H//2-i:H//2+1,H//2-i:H//2+i+1],1) \
                        / (F.adaptive_avg_pool2d(attn[:,:,H//2-i:H//2+1,H//2-i:H//2+i+1],1) + eps)
            y = torch.cat( (y,y_temp[:,None,:].squeeze(-1).squeeze(-1)),dim=1)

            y_temp = F.adaptive_avg_pool2d(x[:,:,H//2:H//2+i+1,H//2-i:H//2+i+1],1) \
                        / (F.adaptive_avg_pool2d(attn[:,:,H//2:H//2+i+1,H//2-i:H//2+i+1],1) + eps)
            y = torch.cat( (y,y_temp[:,None,:].squeeze(-1).squeeze(-1)),dim=1)

            y_temp = F.adaptive_avg_pool2d(x[:,:,H//2-i:H//2+i+1,H//2-i:H//2+1],1) \
                        / (F.adaptive_avg_pool2d(attn[:,:,H//2-i:H//2+i+1,H//2-i:H//2+1],1) + eps)
            y = torch.cat( (y,y_temp[:,None,:].squeeze(-1).squeeze(-1)),dim=1)

            y_temp = F.adaptive_avg_pool2d(x[:,:,H//2-i:H//2+i+1,H//2:H//2+i+1],1) \
                        / (F.adaptive_avg_pool2d(attn[:,:,H//2-i:H//2+i+1,H//2:H//2+i+1],1) + eps)
            y = torch.cat( (y,y_temp[:,None,:].squeeze(-1).squeeze(-1)),dim=1)

        y = F.normalize(y,p=2,dim=2)
        
        return y

class MTAttention_v3(nn.Module):
    def __init__(self, inputdim, base_dim,heads,multi_scale):
        super(MTAttention_v3, self).__init__()
        self.multi_tarattn = []
        self.fc = []
        dim = 512
        for i in range(multi_scale-1):
            self.multi_tarattn += [CAttention(inputdim, base_dim,heads)]
            self.fc += [LinearBlock(inputdim+base_dim*heads, dim, norm='bn', activation='relu')]
        self.centralfc = LinearBlock(inputdim, base_dim*heads, norm='bn', activation='relu')
        self.fc2 = LinearBlock(inputdim+base_dim*heads, dim, norm='bn', activation='relu')
        self.multi_tarattn = nn.ModuleList(self.multi_tarattn)
        self.fc = nn.ModuleList(self.fc)
        self.dim = heads*base_dim
    def forward(self, x):
        B, C, H,W= x.shape
        eps = 1e-16
        x0 = x[:,:,H//2,W//2]
        y = self.fc2(torch.cat((x0,F.normalize(self.centralfc(x0),p=2,dim=-1)),dim=-1))
        y = y[:,None,:]
        for i in range(1,H//2+1):
            y_temp = F.normalize(self.multi_tarattn[5*(i-1)](x[:,:,H//2-i:H//2+i+1,H//2-i:H//2+i+1],x0),p=2,dim=-1)
            y_temp = self.fc[5*(i-1)](torch.cat((x0,y_temp),dim=-1))
            y = torch.cat( (y,y_temp[:,None,:]),dim=1)

            y_temp = F.normalize(self.multi_tarattn[5*(i-1)+1](x[:,:,H//2-i:H//2+1,H//2-i:H//2+i+1],x0),p=2,dim=-1)
            y_temp = self.fc[5*(i-1)+1](torch.cat((x0,y_temp),dim=-1))
            y = torch.cat( (y,y_temp[:,None,:]),dim=1)

            y_temp =F.normalize(self.multi_tarattn[5*(i-1)+2](x[:,:,H//2:H//2+i+1,H//2-i:H//2+i+1],x0),p=2,dim=-1)
            y_temp = self.fc[5*(i-1)+2](torch.cat((x0,y_temp),dim=-1))
            y = torch.cat( (y,y_temp[:,None,:]),dim=1)

            y_temp = F.normalize(self.multi_tarattn[5*(i-1)+3](x[:,:,H//2-i:H//2+i+1,H//2-i:H//2+1],x0),p=2,dim=-1)
            y_temp = self.fc[5*(i-1)+3](torch.cat((x0,y_temp),dim=-1))
            y = torch.cat( (y,y_temp[:,None,:]),dim=1)

            y_temp =F.normalize( self.multi_tarattn[5*(i-1)+4](x[:,:,H//2-i:H//2+i+1,H//2:H//2+i+1],x0),p=2,dim=-1)
            y_temp = self.fc[5*(i-1)+4](torch.cat((x0,y_temp),dim=-1))
            y = torch.cat( (y,y_temp[:,None,:]),dim=1)

        
        return y

class TAttention(nn.Module):
    def __init__(self, inputdim, base_dim,heads):
        super(TAttention, self).__init__()
        self.v = nn.Linear(inputdim,heads*base_dim)
        self.v_norm = nn.BatchNorm1d(heads*base_dim)
        self.v_act = nn.ReLU()
        self.k = nn.Linear(inputdim,heads*base_dim)
        self.k_norm = nn.BatchNorm1d(heads*base_dim)
        self.heads = heads
        self.scale = base_dim**(-0.5)
        
        self.v.apply(weights_init('kaiming'))
        self.v_norm.apply(weights_init('kaiming'))      
        self.k.apply(weights_init('xavier'))
        self.k_norm.apply(weights_init('kaiming')) 
        self.dim = heads*base_dim
    def forward(self, x,x0):
        B, C, H,W= x.shape
        x = x.reshape(B,C,H*W).permute(0,2,1)
        C = self.dim
        N = H*W

        v = self.v_act(self.v_norm(self.v(x).view(-1,C))).reshape(B,N,C).permute(0,2,1)
        k = self.k_norm(self.k(x).view(-1,C)).reshape(B,N,self.heads,C//self.heads).permute(0,2,1,3)
        q = self.k_norm(self.k(x0)).reshape(B,1,self.heads,C//self.heads).permute(0,2,1,3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1).repeat(1,1,C//self.heads,1).view(B,C,N)
        x = torch.sum(attn * v,dim=-1)/N 

        return x

class MTAttention_v23(nn.Module):
    def __init__(self, inputdim, base_dim,heads,L2=True):
        super(MTAttention_v23, self).__init__()
        self.v = nn.Linear(inputdim,heads*base_dim)
        self.v_norm = nn.BatchNorm1d(heads*base_dim)
        self.v_act = nn.ReLU()
        self.k = nn.Linear(inputdim,heads*base_dim)
        self.k_norm = nn.BatchNorm1d(heads*base_dim)
        self.heads = heads
        self.scale = base_dim**(-0.5)
        self.fc = LinearBlock(inputdim+base_dim*heads, 512, norm='bn', activation='relu')
        self.v.apply(weights_init('kaiming'))
        self.v_norm.apply(weights_init('kaiming'))      
        self.k.apply(weights_init('xavier'))
        self.k_norm.apply(weights_init('kaiming')) 
        self.fc.apply(weights_init('kaiming')) 
        self.dim = heads*base_dim
        self.L2 = L2
    def forward(self, x):
        B, C, H,W= x.shape
        N = H*W
        x = x.view(B,C,N).permute(0,2,1)
        x0 = x[:,N//2,:].unsqueeze(1)
        C = self.dim
        H = round(N**(0.5))
        v = self.v_act(self.v_norm(self.v(x).view(-1,C))).reshape(B,N,C).permute(0,2,1).view(B,C,H,H)
        k = self.k_norm(self.k(x).view(-1,C)).reshape(B,N,self.heads,C//self.heads).permute(0,2,1,3)
        q = k[:,:,N//2,:].unsqueeze(2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1).repeat(1,1,C//self.heads,1).view(B,C,H,H)
        x = attn * v 
        eps = 1e-16
        y = x[:,:,H//2,H//2].unsqueeze(1)
        for i in range(1,H//2+1):
            y_temp = F.adaptive_avg_pool2d(x[:,:,H//2-i:H//2+i+1,H//2-i:H//2+i+1],1) \
                        / (F.adaptive_avg_pool2d(attn[:,:,H//2-i:H//2+i+1,H//2-i:H//2+i+1],1) + eps)
            y = torch.cat( (y,y_temp[:,None,:].squeeze(-1).squeeze(-1)),dim=1)

            y_temp = F.adaptive_avg_pool2d(x[:,:,H//2-i:H//2+1,H//2-i:H//2+i+1],1) \
                        / (F.adaptive_avg_pool2d(attn[:,:,H//2-i:H//2+1,H//2-i:H//2+i+1],1) + eps)
            y = torch.cat( (y,y_temp[:,None,:].squeeze(-1).squeeze(-1)),dim=1)

            y_temp = F.adaptive_avg_pool2d(x[:,:,H//2:H//2+i+1,H//2-i:H//2+i+1],1) \
                        / (F.adaptive_avg_pool2d(attn[:,:,H//2:H//2+i+1,H//2-i:H//2+i+1],1) + eps)
            y = torch.cat( (y,y_temp[:,None,:].squeeze(-1).squeeze(-1)),dim=1)

            y_temp = F.adaptive_avg_pool2d(x[:,:,H//2-i:H//2+i+1,H//2-i:H//2+1],1) \
                        / (F.adaptive_avg_pool2d(attn[:,:,H//2-i:H//2+i+1,H//2-i:H//2+1],1) + eps)
            y = torch.cat( (y,y_temp[:,None,:].squeeze(-1).squeeze(-1)),dim=1)

            y_temp = F.adaptive_avg_pool2d(x[:,:,H//2-i:H//2+i+1,H//2:H//2+i+1],1) \
                        / (F.adaptive_avg_pool2d(attn[:,:,H//2-i:H//2+i+1,H//2:H//2+i+1],1) + eps)
            y = torch.cat( (y,y_temp[:,None,:].squeeze(-1).squeeze(-1)),dim=1)
        if self.L2:
            y = F.normalize(y,p=2,dim=2)
        b,n,c = y.shape
        y = torch.cat((x0.repeat(1,n,1),y),dim=-1)
        y = self.fc(y.view(b*n,-1)).view(b,n,-1)
        return y

class MTAttention_v0(nn.Module):
    def __init__(self, inputdim, base_dim,heads,L2=True):
        super(MTAttention_v0, self).__init__()
        self.v = nn.Linear(inputdim,heads*base_dim)
        self.v_norm = nn.BatchNorm1d(heads*base_dim)
        self.v_act = nn.ReLU()
        self.k = nn.Linear(inputdim,heads*base_dim)
        self.k_norm = nn.BatchNorm1d(heads*base_dim)
        self.heads = heads
        self.scale = base_dim**(-0.5)
        self.fc = LinearBlock(inputdim+base_dim*heads, 512, norm='bn', activation='relu')
        self.v.apply(weights_init('kaiming'))
        self.v_norm.apply(weights_init('kaiming'))      
        self.k.apply(weights_init('xavier'))
        self.k_norm.apply(weights_init('kaiming')) 
        self.fc.apply(weights_init('kaiming')) 
        self.dim = heads*base_dim
        self.L2 = L2
    def forward(self, x):
        B, C, H,W= x.shape
        N = H*W
        x = x.view(B,C,N).permute(0,2,1)
        x0 = x[:,N//2,:]
        C = self.dim
        H = round(N**(0.5))
        v = self.v_act(self.v_norm(self.v(x).view(-1,C))).reshape(B,N,C).permute(0,2,1).view(B,C,H,H)
        k = self.k_norm(self.k(x).view(-1,C)).reshape(B,N,self.heads,C//self.heads).permute(0,2,1,3)
        q = k[:,:,N//2,:].unsqueeze(2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1).repeat(1,1,C//self.heads,1).view(B,C,H,H)
        x = attn * v 
        eps = 1e-16
        y = F.adaptive_avg_pool2d(x,1).squeeze(-1).squeeze(-1)
        if self.L2:
            y = F.normalize(y,p=2,dim=1)
        y = torch.cat((x0,y),dim=-1)
        y = self.fc(y)
        return y


class MATA(nn.Module):
    def __init__(self, inputdim, dim, class_num,base_dim,heads,multi_cls_head=1,cls_shared = True,feature_shared=True,
                 MSFE=True,L2=True,weighting=True):
        super(MATA, self).__init__()
        # self.inputdim =  inputdim
        dim1 = dim
        self.class_num = class_num
        self.MSFE = MSFE
        if MSFE:
            if feature_shared:
                self.multi_centralattention = MTAttention_v23(inputdim,base_dim,heads,L2)
            else:
                self.multi_centralattention = MTAttention_v3(inputdim,base_dim,heads,multi_cls_head)
            if cls_shared:
                self.cls_heads= nn.Sequential(self._build_mlp(1,dim,dim,dim,last_bn=True), 
                            LinearBlock(dim, class_num, norm='bn', activation='none'))
                self.cls_heads.apply(weights_init('kaiming')) 
            else:
                self.cls_heads = []
                for i in range(multi_cls_head):
                    cls_heads = nn.Sequential(self._build_mlp(1,dim,dim,dim,last_bn=True), 
                                LinearBlock(dim, class_num, norm='bn', activation='none'))
                    cls_heads.apply(weights_init('kaiming')) 
                    self.cls_heads.append(cls_heads)  
                self.cls_heads = nn.ModuleList(self.cls_heads)              
            if weighting:
                self.w = nn.Parameter(torch.zeros(1,multi_cls_head,class_num),requires_grad=True)
            else:
                self.w = nn.Parameter(torch.zeros(1,multi_cls_head,class_num),requires_grad=False)        
        
        else:
            self.multi_centralattention =MTAttention_v0(inputdim,base_dim,heads,L2)
            self.cls_heads= nn.Sequential(self._build_mlp(1,dim,dim,dim,last_bn=True), 
                        LinearBlock(dim, class_num, norm='bn', activation='none'))
            self.cls_heads.apply(weights_init('kaiming')) 
        self.head = heads
        self.multi_cls_head = multi_cls_head
        self.cls_shared = cls_shared

    def forward(self, x):
        B,C,H,W =x.shape
        # X0 = x[:,:,H//2,W//2].unsqueeze(1)
        # x = x.view(B,C,H*W).permute(0,2,1)
        x = self.multi_centralattention(x)
        if self.MSFE:
            if self.cls_shared:
                y = self.cls_heads(x.view(-1,x.shape[2])).view(B,self.multi_cls_head,-1)
            else:
                y =torch.Tensor().cuda()
                for i in range(self.multi_cls_head):
                    ytemp = self.cls_heads[i](x[:,i,:])
                    y = torch.cat((y,ytemp[:,None,:]),dim=1)
        else:
            y = self.cls_heads(x)


        return y

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:

                mlp.append(nn.BatchNorm1d(dim2, affine=False))
        return nn.Sequential(*mlp) 
    def multi_cls_head_loss(self,y,label):
        if self.MSFE:
            label_mean = label[:,None]
            w = F.softmax(self.w,dim=1)
            probs_mean = F.softmax(torch.sum(w*y,dim=1),dim=-1)
            probs_mean = torch.gather(
                probs_mean, dim=-1, index=label_mean).squeeze(-1)
            loss = -torch.log(probs_mean)
        else: 
            label = label[:,None]
            probs = F.softmax(y,dim=-1) 
            probs = torch.gather( probs, dim=-1, index=label).squeeze(-1)
            loss = -torch.log(probs)
        return loss.mean()

    @torch.no_grad() 
    def test_cls(self,y):
        if self.MSFE:
            w = F.softmax(self.w,dim=1)
            probs = F.softmax(torch.sum(w*y,dim=1),dim=-1)
            y = probs
        else: 
            probs = F.softmax(y,dim=-1)
            y = probs             


        return y
