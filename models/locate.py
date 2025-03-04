import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dino import vision_transformer as vits
from models.dino.utils import load_pretrained_weights
from models.model_util import *  #导入该模块中的所有公开内容
from fast_pytorch_kmeans import KMeans


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features #如果out_features为None，则将out_features设置为in_features
        hidden_features = hidden_features or in_features #如果hidden_features为None，则将hidden_features设置为in_features
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)#dropout
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Net(nn.Module):
    #定义一个Net类，继承自nn.Module类
    def __init__(self, aff_classes=36):
        super(Net, self).__init__() #调用父类的构造函数

        self.aff_classes = aff_classes #定义aff_classes属性
        self.gap = nn.AdaptiveAvgPool2d(1) #定义一个全局平均池化层

        # --- hyper-parameters --- #
        self.aff_cam_thd = 0.6 #CAM激活阈值，值越高，选取的区域越关键
        self.part_iou_thd = 0.6 #部分IOU阈值，值越高，选取的区域越关键
        self.cel_margin = 0.5 #对比学习的边界值，控制特征embeddings之间的相似度

        # --- dino-vit features --- #
        self.vit_feat_dim = 384  # dino-vit feature dimension
        self.cluster_num = 3 #聚类数目，本文分为object, background, human三类
        self.stride = 16  # dino-vit stride
        self.patch = 16 # dino-vit patch size 对图像的切分处理需要两个参数，一个是patch size，一个是stride

        self.vit_model = vits.__dict__['vit_small'](patch_size=self.patch, num_classes=0) #加载预训练的vit模型，具体操作：._dict_能够访问模块的命名空间子弟啊，获取名为vits的模型类或者函数，然后完成实例化
        load_pretrained_weights(self.vit_model, '', None, 'vit_small', self.patch) #加载预训练权重

        # --- learning parameters --- #
        #对应原文里面定位关注的特征部分。
        #先投影，输入特征维度为dino vit的输出维度，输出维度=输入维度，隐藏层维度=输入维度的4倍，激活函数为GELU，dropout=0
        #self.aff_fc是一个卷积层，输入维度为dino vit的输出维度（也是前两个卷积的输出维度），输出维度为aff_classes，对应原文1x1class aware convolution ，为了生成每个标签类别的CAM
        self.aff_proj = Mlp(in_features=self.vit_feat_dim, hidden_features=int(self.vit_feat_dim * 4),
                            act_layer=nn.GELU, drop=0.)
        self.aff_ego_proj = nn.Sequential(
            nn.Conv2d(self.vit_feat_dim, self.vit_feat_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.vit_feat_dim),
            nn.ReLU(True),
            nn.Conv2d(self.vit_feat_dim, self.vit_feat_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.vit_feat_dim),
            nn.ReLU(True),
        )
        self.aff_exo_proj = nn.Sequential(
            nn.Conv2d(self.vit_feat_dim, self.vit_feat_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.vit_feat_dim),
            nn.ReLU(True),
            nn.Conv2d(self.vit_feat_dim, self.vit_feat_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.vit_feat_dim),
            nn.ReLU(True),
        )
        self.aff_fc = nn.Conv2d(self.vit_feat_dim, self.aff_classes, 1)

    def forward(self, exo, ego, aff_label, epoch):

        num_exo = exo.shape[1] # number of exocentric images in each sample
        exo = exo.flatten(0, 1)  # b*num_exo x 3 x 224 x 224

        # --- Extract deep descriptors from DINO-vit --- #
        # 以ego_desc为例：
        # ego_key维度: [batch_size, num_heads, N, C//num_heads]
        # 例如: [B, 6, 197, 64]，其中197 = 14*14 + 1(CLS token) 64=384/6 
        
        # 1. permute(0, 2, 3, 1): [B, 6, 197, 64] -> [B, 197, 64, 6]
        # 2. flatten(-2, -1): [B, 197, 64, 6] -> [B, 197, 384(64*6)]
        # 3. detach(): 分离计算图
        
        # 1. ego_desc[:, 1:]: 移除CLS token，[B, 197, 384] -> [B, 196, 384]
        # 2. self.aff_proj(): MLP投影
        # 3. 残差连接: 原始特征 + 投影特征
        
        #1. ego_desc[:, 1:, :]: 移除CLS token，[B, 197, 384] -> [B, 196, 384]
        # 2. _reshape_transform内部转换:
        #    - 计算height和width: (224-16)/16 + 1 = 14
        #    - reshape: [B, 196, 384] -> [B, 14, 14, 384]
        #    - transpose: [B, 14, 14, 384] -> [B, 384, 14, 14]
        
        with torch.no_grad():
            _, ego_key, ego_attn = self.vit_model.get_last_key(ego)  # attn: b x 6 x (1+hw) x (1+hw)
            _, exo_key, exo_attn = self.vit_model.get_last_key(exo)
            # permute: 调整维度顺序
            # flatten: 将最后两个维度合并
            # detach: 分离计算图，节省内存
            ego_desc = ego_key.permute(0, 2, 3, 1).flatten(-2, -1).detach() 
            exo_desc = exo_key.permute(0, 2, 3, 1).flatten(-2, -1).detach()

        ego_proj = ego_desc[:, 1:] + self.aff_proj(ego_desc[:, 1:]) #对应原文最后一层输出到Extract模块的跳跃连接
        exo_proj = exo_desc[:, 1:] + self.aff_proj(exo_desc[:, 1:]) #选取第一个token（CLS token）之外的所有特征，因为只需要空间特征，然后使用MLP（原文中的φ）进行特征变换，然后残差连接，保留原始信息。
        ego_desc = self._reshape_transform(ego_desc[:, 1:, :], self.patch, self.stride) 
        exo_desc = self._reshape_transform(exo_desc[:, 1:, :], self.patch, self.stride)
        ego_proj = self._reshape_transform(ego_proj, self.patch, self.stride)
        exo_proj = self._reshape_transform(exo_proj, self.patch, self.stride)

        b, c, h, w = ego_desc.shape #[B, 384, 14, 14]
        ego_cls_attn = ego_attn[:, :, 0, 1:].reshape(b, 6, h, w)
        ego_cls_attn = (ego_cls_attn > ego_cls_attn.flatten(-2, -1).mean(-1, keepdim=True).unsqueeze(-1)).float()
        head_idxs = [0, 1, 3]
        ego_sam = ego_cls_attn[:, head_idxs].mean(1)
        ego_sam = normalize_minmax(ego_sam)
        ego_sam_flat = ego_sam.flatten(-2, -1)

        # --- Affordance CAM generation --- #
        exo_proj = self.aff_exo_proj(exo_proj)
        aff_cam = self.aff_fc(exo_proj)  # b*num_exo x 36 x h x w
        aff_logits = self.gap(aff_cam).reshape(b, num_exo, self.aff_classes)
        aff_cam_re = aff_cam.reshape(b, num_exo, self.aff_classes, h, w)

        gt_aff_cam = torch.zeros(b, num_exo, h, w).cuda()
        for b_ in range(b):
            gt_aff_cam[b_, :] = aff_cam_re[b_, :, aff_label[b_]]

        # --- Clustering extracted descriptors based on CAM --- #
        ego_desc_flat = ego_desc.flatten(-2, -1)  # b x 384 x hw
        exo_desc_re_flat = exo_desc.reshape(b, num_exo, c, h, w).flatten(-2, -1)
        sim_maps = torch.zeros(b, self.cluster_num, h * w).cuda()
        exo_sim_maps = torch.zeros(b, num_exo, self.cluster_num, h * w).cuda()
        part_score = torch.zeros(b, self.cluster_num).cuda()
        part_proto = torch.zeros(b, c).cuda()
        for b_ in range(b):
            exo_aff_desc = []
            for n in range(num_exo):
                tmp_cam = gt_aff_cam[b_, n].reshape(-1)
                tmp_max, tmp_min = tmp_cam.max(), tmp_cam.min()
                tmp_cam = (tmp_cam - tmp_min) / (tmp_max - tmp_min + 1e-10)
                tmp_desc = exo_desc_re_flat[b_, n]
                tmp_top_desc = tmp_desc[:, torch.where(tmp_cam > self.aff_cam_thd)[0]].T  # n x c
                exo_aff_desc.append(tmp_top_desc)
            exo_aff_desc = torch.cat(exo_aff_desc, dim=0)  # (n1 + n2 + n3) x c

            if exo_aff_desc.shape[0] < self.cluster_num:
                continue

            kmeans = KMeans(n_clusters=self.cluster_num, mode='euclidean', max_iter=300)
            kmeans.fit_predict(exo_aff_desc.contiguous())
            clu_cens = F.normalize(kmeans.centroids, dim=1)

            # save the exocentric similarity maps for visualization in training
            for n_ in range(num_exo):
                exo_sim_maps[b_, n_] = torch.mm(clu_cens, F.normalize(exo_desc_re_flat[b_, n_], dim=0))

            # find object part prototypes and background prototypes
            sim_map = torch.mm(clu_cens, F.normalize(ego_desc_flat[b_], dim=0))  # self.cluster_num x hw
            tmp_sim_max, tmp_sim_min = torch.max(sim_map, dim=-1, keepdim=True)[0], \
                                       torch.min(sim_map, dim=-1, keepdim=True)[0]
            sim_map_norm = (sim_map - tmp_sim_min) / (tmp_sim_max - tmp_sim_min + 1e-12)

            sim_map_hard = (sim_map_norm > torch.mean(sim_map_norm, 1, keepdim=True)).float()
            sam_hard = (ego_sam_flat > torch.mean(ego_sam_flat, 1, keepdim=True)).float()

            inter = (sim_map_hard * sam_hard[b_]).sum(1)
            union = sim_map_hard.sum(1) + sam_hard[b_].sum() - inter
            p_score = (inter / sim_map_hard.sum(1) + sam_hard[b_].sum() / union) / 2

            sim_maps[b_] = sim_map
            part_score[b_] = p_score

            if p_score.max() < self.part_iou_thd:
                continue

            part_proto[b_] = clu_cens[torch.argmax(p_score)]

        sim_maps = sim_maps.reshape(b, self.cluster_num, h, w)
        exo_sim_maps = exo_sim_maps.reshape(b, num_exo, self.cluster_num, h, w)
        ego_proj = self.aff_ego_proj(ego_proj)
        ego_pred = self.aff_fc(ego_proj)
        aff_logits_ego = self.gap(ego_pred).view(b, self.aff_classes)

        # --- concentration loss --- #
        gt_ego_cam = torch.zeros(b, h, w).cuda()
        loss_con = torch.zeros(1).cuda()
        for b_ in range(b):
            gt_ego_cam[b_] = ego_pred[b_, aff_label[b_]]
            loss_con += concentration_loss(ego_pred[b_])

        gt_ego_cam = normalize_minmax(gt_ego_cam)
        loss_con /= b

        # --- prototype guidance loss --- #
        loss_proto = torch.zeros(1).cuda()
        valid_batch = 0
        if epoch[0] > epoch[1]:
            for b_ in range(b):
                if not part_proto[b_].equal(torch.zeros(c).cuda()):
                    mask = gt_ego_cam[b_]
                    tmp_feat = ego_desc[b_] * mask
                    embedding = tmp_feat.reshape(tmp_feat.shape[0], -1).sum(1) / mask.sum()
                    loss_proto += torch.max(
                        1 - F.cosine_similarity(embedding, part_proto[b_], dim=0) - self.cel_margin,
                        torch.zeros(1).cuda())
                    valid_batch += 1
            loss_proto = loss_proto / (valid_batch + 1e-15)

        masks = {'exo_aff': gt_aff_cam, 'ego_sam': ego_sam,
                 'pred': (sim_maps, exo_sim_maps, part_score, gt_ego_cam)}
        logits = {'aff': aff_logits, 'aff_ego': aff_logits_ego}

        return masks, logits, loss_proto, loss_con

    @torch.no_grad()
    def test_forward(self, ego, aff_label):
        _, ego_key, ego_attn = self.vit_model.get_last_key(ego)  # attn: b x 6 x (1+hw) x (1+hw)
        ego_desc = ego_key.permute(0, 2, 3, 1).flatten(-2, -1)
        ego_proj = ego_desc[:, 1:] + self.aff_proj(ego_desc[:, 1:])
        ego_desc = self._reshape_transform(ego_desc[:, 1:, :], self.patch, self.stride)
        ego_proj = self._reshape_transform(ego_proj, self.patch, self.stride)

        b, c, h, w = ego_desc.shape
        ego_proj = self.aff_ego_proj(ego_proj)
        ego_pred = self.aff_fc(ego_proj)

        gt_ego_cam = torch.zeros(b, h, w).cuda()
        for b_ in range(b):
            gt_ego_cam[b_] = ego_pred[b_, aff_label[b_]]

        return gt_ego_cam

    #先计算特征图的高度和宽度
    #然后将扁平张量变成2维张量
    #调整张量维度的顺序，送入卷积网络做进一步处理
    def _reshape_transform(self, tensor, patch_size, stride):
        height = (224 - patch_size) // stride + 1
        width = (224 - patch_size) // stride + 1
        result = tensor.reshape(tensor.size(0), height, width, tensor.size(-1))
        result = result.transpose(2, 3).transpose(1, 2).contiguous()
        return result
