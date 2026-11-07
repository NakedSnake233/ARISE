import torch
from torch import nn
from dgl.nn.pytorch import GATv2Conv
from dgl.nn import GlobalAttentionPooling
import torch.nn.functional as F
import json
import os
from datetime import datetime

class GraphModel(nn.Module):
    def __init__(self, in_dim, graph_hiddens=[64, 128], device='cpu', attn_head=4, activation=0.2, **kwargs):
        super(GraphModel, self).__init__()
        '''
        Params:
            in_dim: the feature dim of each node
        '''
        layers = []

        for i, hidden in enumerate(graph_hiddens):
            in_feats = graph_hiddens[i-1] if i > 0 else in_dim 
            dropout = kwargs["attn_drop"] if "attn_drop" in kwargs else 0
            layers.append(GATv2Conv(in_feats, out_feats=hidden, num_heads=attn_head, 
                                        attn_drop=dropout, negative_slope=activation, allow_zero_in_degree=True)) 
            self.maxpool = nn.MaxPool1d(attn_head)

        self.net = nn.Sequential(*layers).to(device)
        self.out_dim = graph_hiddens[-1]
        self.pooling = GlobalAttentionPooling(nn.Linear(self.out_dim, 1)) 

    
    def _edge_attention(self, graph, attn):
        if attn is None:
            return None
        src, dst = graph.edges()
        # GATv2 returns one score per head; RRRCA uses a single structural
        # coefficient per edge, so we average heads.
        values = attn
        while values.dim() > 1:
            values = values.mean(dim=-1)
        return {
            "src": src.detach().cpu(),
            "dst": dst.detach().cpu(),
            "weight": values.detach().cpu(),
        }

    def forward(self, graph, x, return_nodes=False, return_attention=False):
        '''
        Input:
            x -- tensor float [batch_size*node_num, feature_in_dim] N = {s1, s2, s3, e1, e2, e3}
        '''
        out = x
        last_attention = None
        for layer in self.net:
            if return_attention:
                try:
                    out, last_attention = layer(graph, out, get_attention=True)
                except TypeError:
                    out = layer(graph, out)
                    last_attention = None
            else:
                out = layer(graph, out)
            out = self.maxpool(out.permute(0, 2, 1)).permute(0, 2, 1).squeeze(1)

        graph_embedding = self.pooling(graph, out) #[bz*node, out_dim] --> [bz, out_dim]
        if return_nodes or return_attention:
            result = [graph_embedding, out]
            if return_attention:
                result.append(self._edge_attention(graph, last_attention))
            return tuple(result)
        return graph_embedding

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.0):
        super(TemporalResidualBlock, self).__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            dilation=dilation,
            padding=padding,
        )
        self.chomp = Chomp1d(padding)
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        out = self.conv(x)
        out = self.chomp(out)
        out = out.transpose(1, 2)
        out = self.norm(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = out.transpose(1, 2)

        residual = self.residual(x)
        if residual.size(-1) != out.size(-1):
            residual = residual[..., -out.size(-1):]
        return F.relu(out + residual)

class ConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_sizes, dilation=2, dropout=0.0, dev="cpu"):
        super(ConvNet, self).__init__()
        layers = []
        for i in range(len(kernel_sizes)):
            dilation_size = dilation ** i
            kernel_size = kernel_sizes[i]
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(
                TemporalResidualBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    dilation=dilation_size,
                    dropout=dropout,
                )
            )
            
        self.network = nn.Sequential(*layers)
        
        self.out_dim = num_channels[-1]
        self.network.to(dev)
        
    
    def forward(self, x): #[batch_size, T, in_dim]
        x = x.permute(0, 2, 1).float() #[batch_size, in_dim, T]
        out = self.network(x) #[batch_size, out_dim, T]
        out = out.permute(0, 2, 1) #[batch_size, T, out_dim]
        return out

import math
class SelfAttention(nn.Module):
    def __init__(self, input_size, seq_len):
        """
        Args:
            input_size: int, hidden_size * num_directions
            seq_len: window_size
        """
        super(SelfAttention, self).__init__()
        self.atten_w = nn.Parameter(torch.randn(seq_len, input_size, 1))
        self.atten_bias = nn.Parameter(torch.randn(seq_len, 1, 1))
        self.glorot(self.atten_w)
        self.atten_bias.data.fill_(0)

    def forward(self, x):
        # x: [batch_size, window_size, input_size]
        input_tensor = x.transpose(1, 0)  # w x b x h
        input_tensor = (torch.bmm(input_tensor, self.atten_w) + self.atten_bias)  # w x b x out
        input_tensor = input_tensor.transpose(1, 0)
        atten_weight = input_tensor.tanh()
        weighted_sum = torch.bmm(atten_weight.transpose(1, 2), x).squeeze(1)
        return weighted_sum

    def glorot(self, tensor):
        if tensor is not None:
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)

class TraceModel(nn.Module):
    def __init__(self, device='cpu', trace_hiddens=[20, 50], trace_kernel_sizes=[3, 3], self_attn=False, chunk_lenth=None, **kwargs):
        super(TraceModel, self).__init__()

        self.out_dim = trace_hiddens[-1]
        assert len(trace_hiddens) == len(trace_kernel_sizes)
        dropout = kwargs["tcn_dropout"] if "tcn_dropout" in kwargs else 0.0
        self.net = ConvNet(2, num_channels=trace_hiddens, kernel_sizes=trace_kernel_sizes, dropout=dropout, dev=device)

        self.self_attn = self_attn
        if self_attn:
            assert (chunk_lenth is not None)
            self.attn_layer = SelfAttention(self.out_dim, chunk_lenth)

    def forward(self, x: torch.tensor): #[bz, T, 1]
        hidden_states = self.net(x)
        if self.self_attn: 
            return self.attn_layer(hidden_states)
        return hidden_states[:,-1,:] #[bz, out_dim]

class MetricModel(nn.Module):
    def __init__(self, metric_num, device='cpu', metric_hiddens=[64, 128], metric_kernel_sizes=[3, 3], self_attn=False, chunk_lenth=None, **kwargs):
        super(MetricModel, self).__init__()
        self.metric_num = metric_num
        self.out_dim = metric_hiddens[-1]
        in_dim = metric_num

        assert len(metric_hiddens) == len(metric_kernel_sizes)
        dropout = kwargs["tcn_dropout"] if "tcn_dropout" in kwargs else 0.0
        self.net = ConvNet(num_inputs=in_dim, num_channels=metric_hiddens, kernel_sizes=metric_kernel_sizes, 
                            dropout=dropout,
                            dev=device)

        self.self_attn = self_attn
        if self_attn:
            assert (chunk_lenth is not None)
            self.attn_layer = SelfAttention(self.out_dim, chunk_lenth)

    
    def forward(self, x): #[bz, T, metric_num]
        assert x.shape[-1] == self.metric_num
        hidden_states = self.net(x)
        if self.self_attn: 
            return self.attn_layer(hidden_states)
        return hidden_states[:,-1,:] #[bz, out_dim]

class LogModel(nn.Module):
    def __init__(self, event_num, out_dim):
        super(LogModel, self).__init__()
        self.embedder = nn.Linear(event_num, out_dim) 
    def forward(self, paras: torch.tensor): #[bz, event_num]
        """
        Input:
            paras: mu with length of event_num
        """
        return self.embedder(paras)

class MultiSourceEncoder(nn.Module):
    def __init__(self, event_num, metric_num, node_num, device, log_dim=64, fuse_dim=64, alpha=0.5, temperature=0.07, modal_contrast_weight=0.1, modal_negative_weight=0.5, modal_negative_margin=0.0, **kwargs):
        super(MultiSourceEncoder, self).__init__()
        self.node_num = node_num
        self.alpha = alpha
        self.temperature = temperature
        self.modal_contrast_weight = modal_contrast_weight
        self.modal_negative_weight = modal_negative_weight
        self.modal_negative_margin = modal_negative_margin

        # 三个模态的编码器
        self.trace_model = TraceModel(device=device, **kwargs)
        trace_dim = self.trace_model.out_dim
        self.log_model = LogModel(event_num, log_dim) 
        self.metric_model = MetricModel(metric_num, device=device, **kwargs)
        metric_dim = self.metric_model.out_dim

        # 为每个模态添加投影头，将不同模态的特征映射到相同维度
        self.proj_dim = 128
        self.trace_proj = nn.Sequential(
            nn.Linear(trace_dim, self.proj_dim),
            nn.ReLU(),
            nn.Linear(self.proj_dim, self.proj_dim)
        )
        self.log_proj = nn.Sequential(
            nn.Linear(log_dim, self.proj_dim),
            nn.ReLU(),
            nn.Linear(self.proj_dim, self.proj_dim)
        )
        self.metric_proj = nn.Sequential(
            nn.Linear(metric_dim, self.proj_dim),
            nn.ReLU(),
            nn.Linear(self.proj_dim, self.proj_dim)
        )

        # 特征融合
        fuse_in = trace_dim + log_dim + metric_dim
        if not fuse_dim % 2 == 0: 
            fuse_dim += 1
        self.fuse = nn.Linear(fuse_in, fuse_dim)
        self.activate = nn.GLU()
        self.feat_in_dim = int(fuse_dim // 2)
        
        self.status_model = GraphModel(in_dim=self.feat_in_dim, device=device, **kwargs)
        self.feat_out_dim = self.status_model.out_dim

    def compute_modal_contrast_loss(self, trace_proj, log_proj, metric_proj):
        batch_size = trace_proj.shape[0]

        # RRRCA Eq.3 uses metrics as the semantic anchor: metric-log and
        # metric-trace pairs are pulled together, while companion log-trace
        # pairs are pushed apart to reduce redundant modality noise.
        sim_metric_log = torch.matmul(metric_proj, log_proj.T) / self.temperature
        sim_metric_trace = torch.matmul(metric_proj, trace_proj.T) / self.temperature
        eps = 1e-8

        pos = torch.exp(sim_metric_log.diag()) + torch.exp(sim_metric_trace.diag())
        denom = torch.exp(sim_metric_log).sum(dim=1) + torch.exp(sim_metric_trace).sum(dim=1)
        loss_pos = -torch.log((pos + eps) / (denom + eps)).mean()

        sim_log_trace = torch.matmul(log_proj, trace_proj.T) / self.temperature
        companion_log_prob = F.log_softmax(sim_log_trace, dim=1).diag()
        if self.modal_negative_margin > 0:
            companion_log_prob = companion_log_prob + self.modal_negative_margin
        loss_neg = companion_log_prob.mean()
        return loss_pos + self.modal_negative_weight * loss_neg
    
    def forward(self, graph, return_details=False):
        # 获取三个模态的原始特征
        trace_embedding = self.trace_model(graph.ndata["traces"]) 
        log_embedding = self.log_model(graph.ndata["logs"]) 
        metric_embedding = self.metric_model(graph.ndata["metrics"]) 

        # 投影到相同维度空间
        trace_proj = F.normalize(self.trace_proj(trace_embedding), dim=1)
        log_proj = F.normalize(self.log_proj(log_embedding), dim=1)
        metric_proj = F.normalize(self.metric_proj(metric_embedding), dim=1)
        
        # 计算模态对比损失
        modal_contrast_loss = self.compute_modal_contrast_loss(trace_proj, log_proj, metric_proj)

        # 特征融合
        feature = self.activate(self.fuse(torch.cat((trace_embedding, log_embedding, metric_embedding), dim=-1)))
        if return_details:
            embeddings, node_states, edge_attention = self.status_model(
                graph,
                feature,
                return_nodes=True,
                return_attention=True,
            )
            node_embeddings = node_states.reshape(graph.batch_size, self.node_num, -1)
            details = {
                "node_embeddings": node_embeddings,
                "edge_attention": edge_attention,
                "trace_embedding": trace_embedding,
                "log_embedding": log_embedding,
                "metric_embedding": metric_embedding,
            }
            return embeddings, self.modal_contrast_weight * modal_contrast_loss, details

        embeddings = self.status_model(graph, feature)
        return embeddings, self.modal_contrast_weight * modal_contrast_loss

class FullyConnected(nn.Module):
    def __init__(self, in_dim, out_dim, linear_sizes):
        super(FullyConnected, self).__init__()
        layers = []
        for i, hidden in enumerate(linear_sizes):
            input_size = in_dim if i == 0 else linear_sizes[i-1]
            layers += [nn.Linear(input_size, hidden), nn.ReLU()]
        layers += [nn.Linear(linear_sizes[-1], out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor): #[batch_size, in_dim]
        return self.net(x)

import numpy as np
class MainModel(nn.Module):
    def __init__(self, event_num, metric_num, node_num, device, alpha=0.5, temperature=0.07, modal_contrast_weight=0.1, modal_negative_weight=0.5, modal_negative_margin=0.0, debug=False, **kwargs):
        super(MainModel, self).__init__()

        self.device = device
        self.node_num = node_num
        self.alpha = alpha
        self.temperature = temperature
        self.modal_contrast_weight = modal_contrast_weight
        self.best_loss = float('inf')  # 记录最佳loss
        self.best_prompts = None  # 记录最佳prompts
        self.epoch_prompts = []  # 记录当前epoch的所有prompts

        self.encoder = MultiSourceEncoder(event_num, metric_num, node_num, device, 
                                        temperature=temperature,
                                        modal_contrast_weight=modal_contrast_weight,
                                        modal_negative_weight=modal_negative_weight,
                                        modal_negative_margin=modal_negative_margin,
                                        debug=debug, alpha=alpha, **kwargs)

        self.detector = FullyConnected(self.encoder.feat_out_dim, 2, kwargs['detect_hiddens']).to(device)
        self.decoder_criterion = nn.CrossEntropyLoss()
        self.locator = FullyConnected(self.encoder.feat_out_dim, node_num, kwargs['locate_hiddens']).to(device)
        self.locator_criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.get_prob = nn.Softmax(dim=-1)

    def generate_prompt(self, node_embeddings, node_probs, fault_indexs):
        """Generate prompts in English format for Llama-factory fine-tuning"""
        prompts = []
        batch_size = node_embeddings.shape[0]
        
        for i in range(batch_size):
            # Get current sample's embedding and probability
            emb = node_embeddings[i].detach().cpu().numpy()
            probs = node_probs[i]
            culprit = int(fault_indexs[i].item())
            
            # Build prompt
            human_message = "Given the feature vectors of the following service nodes, please analyze the potential fault nodes:\n\n"
            
            # Add node information to human part
            for j in range(self.node_num):
                node_info = f"Node {j}:\n"
                feature_vector = [round(float(v), 6) for v in emb[j].tolist()]
                node_info += f"Feature vector: {feature_vector}\n"
                node_info += f"Fast thinking score: {probs[j]:.4f}\n"
                human_message += node_info + "\n"
            
            # Build assistant response with only probabilities and final diagnosis
            assistant_message = "Based on my analysis, here are the fault probabilities for each node:\n\n"
            
            # Add only probabilities to assistant part
            for j in range(self.node_num):
                assistant_message += f"Node {j} fault probability: {probs[j]:.4f}\n"
            
            # Add analysis result
            assistant_message += "\nDiagnosis result: "
            if culprit == -1:
                assistant_message += "System is running normally, no fault nodes detected."
            else:
                assistant_message += f"Node {culprit} is the fault node with fault probability {probs[culprit]:.4f}."
            
            prompt = {
                "fault_node": culprit,
                "conversations": [
                    {
                        "from": "human",
                        "value": human_message
                    },
                    {
                        "from": "assistant",
                        "value": assistant_message
                    }
                ]
            }
            
            prompts.append(prompt)
        
        return prompts

    def forward(self, graph, fault_indexs, collect_prompts=True):
        batch_size = graph.batch_size
        embeddings, modal_contrast_loss, encoder_details = self.encoder(graph, return_details=True)
        
        y_prob = torch.zeros((batch_size, self.node_num)).to(self.device) 
        for i in range(batch_size):
            if fault_indexs[i] > -1: 
                y_prob[i, fault_indexs[i]] = 1
        y_anomaly = torch.zeros(batch_size).long().to(self.device)
        for i in range(batch_size):
            y_anomaly[i] = int(fault_indexs[i] > -1)

        # 计算定位和检测损失
        locate_logits = self.locator(embeddings)
        locate_loss = self.locator_criterion(locate_logits, fault_indexs.to(self.device))
        detect_logits = self.detector(embeddings)
        detect_loss = self.decoder_criterion(detect_logits, y_anomaly) 
        
        # 总损失 = 检测损失 + 定位损失 + 模态对比损失
        loss = self.alpha * detect_loss + (1-self.alpha) * locate_loss + modal_contrast_loss

        node_probs = self.get_prob(locate_logits.detach()).cpu().numpy()
        y_pred = self.inference(batch_size, node_probs, detect_logits)
        
        prompts = []
        if collect_prompts:
            # 生成prompt。论文中的 Prior Data 使用共享增强节点表征，而不是图级表征。
            prompts = self.generate_prompt(encoder_details["node_embeddings"], node_probs, fault_indexs)

            # 将当前batch的prompts添加到epoch_prompts中
            self.epoch_prompts.extend(prompts)
        
        result = {
            'loss': loss, 
            'y_pred': y_pred, 
            'y_prob': y_prob.detach().cpu().numpy(), 
            'pred_prob': node_probs,
            'fast_scores': node_probs,
            'detect_loss': detect_loss.item(),
            'locate_loss': locate_loss.item(),
            'modal_contrast_loss': modal_contrast_loss.item(),
            'edge_attention': encoder_details["edge_attention"],
            'prompts': prompts  # 添加返回当前batch的prompts
        }
        
        return result

    def inference(self, batch_size, node_probs, detect_logits=None):
        node_list = np.flip(node_probs.argsort(axis=1), axis=1)
        
        y_pred = []
        detect_pred = detect_logits.detach().cpu().numpy().argmax(axis=1)
        for i in range(batch_size):
            if detect_pred[i] < 1: y_pred.append([-1])
            else: y_pred.append(node_list[i])
        
        return y_pred

    def save_best_prompts(self):
        """保存最佳epoch的prompts到文件"""
        if self.best_prompts is not None:
            # 创建result文件夹（如果不存在）
            result_dir = "../result"
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            
            # 保存最佳prompts到文件
            filename = os.path.join(result_dir, "best_prompts.json")
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.best_prompts, f, ensure_ascii=False, indent=2)
            print(f"已保存最佳prompts到文件: {filename}")
            
            # 清空epoch_prompts，为下一个epoch做准备
            self.epoch_prompts = []

    def save_current_epoch_prompts(self, epoch):
        """保存当前epoch的所有prompts到文件"""
        if self.epoch_prompts:
            # 创建result文件夹（如果不存在）
            result_dir = "../result"
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            
            # 保存当前epoch的所有prompts到文件
            filename = os.path.join(result_dir, f"epoch_{epoch}_prompts.json")
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.epoch_prompts, f, ensure_ascii=False, indent=2)
            print(f"已保存epoch {epoch}的prompts到文件: {filename}, 共{len(self.epoch_prompts)}个样本")


    

        

        
