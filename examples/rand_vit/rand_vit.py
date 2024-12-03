import random
import networkx as nx
import torch
from torch import nn
import copy
from timm.models.layers import DropPath

def generate_vit_graph(n, split_blocks=True):
    """
    원래 ViT의 토폴로지를 반영한 그래프 생성
    - split_blocks=True: Attention과 MLP를 별도의 노드로 취급
    - split_blocks=False: Attention과 MLP를 묶은 블록을 하나의 노드로 취급
    """
    G = nx.DiGraph()
    if split_blocks:
        G.add_nodes_from(range(n * 2))  # 각 블록당 두 개의 노드 (Attention, MLP)
        for i in range(n):
            attn_node = i * 2
            mlp_node = i * 2 + 1

            # 블록 내의 연결 (Attention -> MLP)
            G.add_edge(attn_node, mlp_node)

            if i < n - 1:
                next_attn_node = (i + 1) * 2
                # 현재 블록의 MLP 노드에서 다음 블록의 Attention 노드로 연결
                G.add_edge(mlp_node, next_attn_node)
    else:
        G.add_nodes_from(range(n))  # 각 블록을 하나의 노드로 취급
        for i in range(n - 1):
            G.add_edge(i, i + 1)
    return G

def generate_ws_dag(n, k=4, p=0.5):
    """
    Watts-Strogatz small-world graph를 기반으로 DAG 생성
    """
    while True:
        G_undirected = nx.watts_strogatz_graph(n, k, p)
        G = nx.DiGraph()
        G.add_nodes_from(G_undirected.nodes())
        for edge in G_undirected.edges():
            u, v = edge
            if u < v:
                G.add_edge(u, v)
            else:
                G.add_edge(v, u)
        if nx.is_directed_acyclic_graph(G):
            return G

def generate_ba_dag(n, m=2):
    """
    Barabási-Albert scale-free graph를 기반으로 DAG 생성
    """
    while True:
        G_undirected = nx.barabasi_albert_graph(n, m)
        G = nx.DiGraph()
        G.add_nodes_from(G_undirected.nodes())
        for edge in G_undirected.edges():
            u, v = edge
            if u < v:
                G.add_edge(u, v)
            else:
                G.add_edge(v, u)
        if nx.is_directed_acyclic_graph(G):
            return G

class AttentionModule(nn.Module):
    def __init__(self, block, dp_rate=0.0, use_residual=True):
        super(AttentionModule, self).__init__()
        self.norm1 = copy.deepcopy(block.norm1)
        self.attn = copy.deepcopy(block.attn)
        self.drop_path = DropPath(dp_rate) if dp_rate > 0.0 else nn.Identity()
        self.use_residual = use_residual

    def forward(self, x):
        if self.use_residual:
            return x + self.drop_path(self.attn(self.norm1(x)))
        else:
            return self.drop_path(self.attn(self.norm1(x)))


class MixtureOfExpertsMLPModule(nn.Module):
    def __init__(self, block, m=16, use_residual=True):
        super(MixtureOfExpertsMLPModule, self).__init__()
        self.use_residual = use_residual
        self.m = m  # 노드 수

        # 원래 MLP 모듈의 fc1과 fc2 가중치 및 편향 가져오기
        fc1_weight = block.mlp.fc1.weight.data.clone()  # [hidden_dim, embed_dim]
        fc1_bias = block.mlp.fc1.bias.data.clone()      # [hidden_dim]
        fc2_weight = block.mlp.fc2.weight.data.clone()  # [embed_dim, hidden_dim]
        fc2_bias = block.mlp.fc2.bias.data.clone()      # [embed_dim]

        hidden_dim, input_dim = fc1_weight.shape  # hidden_dim: fc1의 출력 차원, input_dim: fc1의 입력 차원

        # 각 노드에 할당될 차원 계산
        chunk_size = hidden_dim // m
        remainder = hidden_dim % m

        # 노드별 MLP 서브모듈 생성
        self.mlp_modules = nn.ModuleDict()
        for i in range(m):
            start_idx = i * chunk_size + min(i, remainder)
            end_idx = start_idx + chunk_size + (1 if i < remainder else 0)
            chunk_size_i = end_idx - start_idx

            # fc1 슬라이스
            fc1_weight_i = fc1_weight[start_idx:end_idx, :]  # [chunk_size_i, input_dim]
            fc1_bias_i = fc1_bias[start_idx:end_idx]         # [chunk_size_i]

            # fc2 슬라이스
            fc2_weight_i = fc2_weight[:, start_idx:end_idx]  # [embed_dim, chunk_size_i]

            # MLP 서브모듈 생성
            mlp_submodule = nn.Sequential(
                nn.Linear(input_dim, chunk_size_i),
                nn.GELU(),
                torch.nn.LayerNorm(chunk_size_i),
                nn.Linear(chunk_size_i, input_dim)
            )

            # 가중치 초기화
            mlp_submodule[0].weight.data = fc1_weight_i
            mlp_submodule[0].bias.data = fc1_bias_i
            mlp_submodule[-1].weight.data = fc2_weight_i
            mlp_submodule[-1].bias.data.zero_()  # fc2의 편향은 노드별로 0으로 초기화

            self.mlp_modules[str(i)] = mlp_submodule

        # fc2의 편향을 버퍼로 등록
        self.register_buffer('fc2_bias', fc2_bias)

        # 랜덤 그래프 생성
        self.G = generate_ws_dag(m)

        # 각 노드에 대한 라우터 모듈 생성
        self.router_modules = nn.ModuleDict()
        for node in self.G.nodes():
            preds = list(self.G.predecessors(node))
            if preds:
                num_preds = len(preds)
                # 입력 차원에서 num_preds로 매핑되는 Linear 레이어
                self.router_modules[str(node)] = nn.Linear(input_dim, num_preds)

        # outdegree가 0인 노드들 (출력 노드) 저장
        self.terminal_nodes = [node for node in self.G.nodes() if self.G.out_degree(node) == 0]

        # 만약 출력 노드가 없다면, 위상 정렬에서 마지막 노드를 저장
        self.topo_order = list(nx.topological_sort(self.G))
        self.last_node = self.topo_order[-1]

        # 정규화 레이어 복사
        self.norm = copy.deepcopy(block.norm2)

    def forward(self, x):
        # 입력 정규화
        x_norm = self.norm(x)  # [B, N, embed_dim]

        node_outputs = {}

        for node in nx.topological_sort(self.G):
            preds = list(self.G.predecessors(node))
            if preds:
                # 이전 노드들의 출력 가져오기
                pred_outputs = torch.stack([node_outputs[pred] for pred in preds], dim=-1)  # [B, N, embed_dim, num_preds]

                # 라우터 모듈을 통해 가중치 계산
                router_module = self.router_modules[str(node)]
                router_logits = router_module(x_norm)  # [B, N, num_preds]
                router_weights = torch.sigmoid(router_logits)  # [B, N, num_preds]

                # 가중치 적용을 위한 차원 확장
                router_weights = router_weights.unsqueeze(2)  # [B, N, 1, num_preds]

                # 가중합 계산
                node_input = (pred_outputs * router_weights).sum(dim=-1)  # [B, N, embed_dim]
            else:
                # 이전 노드가 없으면 입력은 x_norm
                node_input = x_norm  # [B, N, embed_dim]

            # MLP 서브모듈 적용
            mlp_submodule = self.mlp_modules[str(node)]
            node_output = mlp_submodule(node_input)  # [B, N, embed_dim]

            node_outputs[node] = node_output

        # outdegree가 0인 노드들의 출력 합산
        if self.terminal_nodes:
            combined_output = sum(node_outputs[node] for node in self.terminal_nodes)  # [B, N, embed_dim]
        else:
            # 출력 노드가 없으면 위상 정렬에서 마지막 노드의 출력 사용
            combined_output = node_outputs[self.last_node]  # [B, N, embed_dim]

        # fc2의 편향 추가
        combined_output += self.fc2_bias.unsqueeze(0).unsqueeze(0)

        if self.use_residual:
            output = x + combined_output
        else:
            output = combined_output

        return output



class BlockModule(nn.Module):
    def __init__(self, block, dp_rate=0.0, use_residual=True):
        super(BlockModule, self).__init__()
        self.norm1 = copy.deepcopy(block.norm1)
        self.attn = copy.deepcopy(block.attn)
        self.norm2 = copy.deepcopy(block.norm2)
        self.mlp = copy.deepcopy(block.mlp)
        self.drop_path = DropPath(dp_rate) if dp_rate > 0.0 else nn.Identity()
        self.use_residual = use_residual

    def forward(self, x):
        if self.use_residual:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        else:
            x = self.drop_path(self.attn(self.norm1(x)))
            x = self.drop_path(self.mlp(self.norm2(x)))
            return x

class MLPModule(nn.Module):
    def __init__(self, block, dp_rate=0.0, use_residual=True):
        super(MLPModule, self).__init__()
        self.norm2 = copy.deepcopy(block.norm2)
        self.mlp = copy.deepcopy(block.mlp)
        self.drop_path = DropPath(dp_rate) if dp_rate > 0.0 else nn.Identity()
        self.use_residual = use_residual

    def forward(self, x):
        if self.use_residual:
            return x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            return self.drop_path(self.mlp(self.norm2(x)))

class ModifiedViTModel(nn.Module):
    def __init__(self, vit_model, m=16, k=4, use_residual=True, drop_path_rate=0.0):
        super(ModifiedViTModel, self).__init__()

        # ViT 모델의 필요한 부분 복사
        self.patch_embed = copy.deepcopy(vit_model.patch_embed)
        self.cls_token = copy.deepcopy(vit_model.cls_token)
        self.pos_embed = copy.deepcopy(vit_model.pos_embed)
        self.pos_drop = copy.deepcopy(vit_model.pos_drop)
        self.norm = copy.deepcopy(vit_model.norm)
        self.head = copy.deepcopy(vit_model.head)

        # 드롭 패스율 설정
        num_blocks = len(vit_model.blocks)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, num_blocks * 2)]

        # 블록 생성
        self.blocks = nn.ModuleList()
        for idx, block in enumerate(vit_model.blocks):
            # Attention 모듈
            attn_module = AttentionModule(block, dp_rate=dp_rates[idx * 2], use_residual=use_residual)

            if idx >= num_blocks - k:
                # 마지막 k개의 블록은 Mixture of Experts MLP 모듈 사용
                mlp_module = MixtureOfExpertsMLPModule(block, m=m, use_residual=use_residual)
            else:
                # 나머지 블록은 원래의 MLP 모듈 사용
                mlp_module = MLPModule(block, dp_rate=dp_rates[idx * 2 + 1], use_residual=use_residual)

            # 블록에 모듈 추가
            transformer_block = nn.ModuleDict({
                'attn_module': attn_module,
                'mlp_module': mlp_module
            })

            self.blocks.append(transformer_block)

    def forward(self, x):
        x = self.patch_embed(x)  # [B, N, embed_dim]
        B, N, _ = x.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, N+1, embed_dim]
        x = x + self.pos_embed  # 위치 임베딩 추가
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block['attn_module'](x)
            x = block['mlp_module'](x)

        x = self.norm(x)
        x = self.head(x[:, 0])

        return x


class RandomGraphTransformer(nn.Module):
    def __init__(self, G, node_module_dict, vit_model):
        super(RandomGraphTransformer, self).__init__()
        self.G = G
        self.node_module_dict = nn.ModuleDict(node_module_dict)
        # ViT 모델의 필요한 부분 복사
        self.patch_embed = copy.deepcopy(vit_model.patch_embed)
        self.cls_token = copy.deepcopy(vit_model.cls_token)
        self.pos_embed = copy.deepcopy(vit_model.pos_embed)
        self.pos_drop = copy.deepcopy(vit_model.pos_drop)
        self.norm = copy.deepcopy(vit_model.norm)
        self.head = copy.deepcopy(vit_model.head)

        # Edge별 학습 가능한 가중치 생성
        self.edge_weights = nn.ParameterDict()
        for node in self.G.nodes():
            preds = list(self.G.predecessors(node))
            for pred in preds:
                edge_key = f"{pred}_{node}"
                # 가중치 초기값을 1로 설정
                self.edge_weights[edge_key] = nn.Parameter(torch.ones(1))

    def forward(self, x):
        x = self.patch_embed(x)  # [B, N, embed_dim]
        B, N, _ = x.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, N+1, embed_dim]
        x = x + self.pos_embed  # 위치 임베딩 추가
        x = self.pos_drop(x)

        node_outputs = {}

        for node in nx.topological_sort(self.G):
            preds = list(self.G.predecessors(node))
            if preds:
                node_input = sum(
                    self.edge_weights[f"{pred}_{node}"] * node_outputs[pred]
                    for pred in preds
                )
            else:
                node_input = x

            module = self.node_module_dict[str(node)]
            # 모듈 적용
            node_output = module(node_input)
            node_outputs[node] = node_output

        # 마지막 노드의 출력을 사용
        final_node = list(self.G.nodes())[-1]
        final_output = node_outputs[final_node]
        x = self.norm(final_output)
        x = self.head(x[:, 0])

        return x


def get_random_graph_vit(vit_model, graph_type='WS', drop_path_rate=0.0, use_residual=True, split_blocks=False):
    # ViT의 블록 수
    num_blocks = len(vit_model.blocks)
    dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, num_blocks * (2 if split_blocks else 1))]

    modules = []
    if split_blocks:
        # 블록을 Attention과 MLP로 분리하여 사용
        for idx, block in enumerate(vit_model.blocks):
            block_copy = copy.deepcopy(block)

            # Attention Module
            attn_module = AttentionModule(block_copy, dp_rate=dp_rates[idx * 2], use_residual=use_residual)
            # MLP Module
            mlp_module = MLPModule(block_copy, dp_rate=dp_rates[idx * 2 + 1], use_residual=use_residual)

            modules.append(attn_module)
            modules.append(mlp_module)
        n = num_blocks * 2  # 총 노드 수
    else:
        # 블록을 하나의 모듈로 사용
        for idx, block in enumerate(vit_model.blocks):
            block_copy = copy.deepcopy(block)
            block_module = BlockModule(block_copy, dp_rate=dp_rates[idx], use_residual=use_residual)
            modules.append(block_module)
        n = num_blocks  # 총 노드 수

    # 그래프 생성
    if graph_type == 'WS':
        G = generate_ws_dag(n)
    elif graph_type == 'BA':
        G = generate_ba_dag(n)
    elif graph_type == 'vit':
        G = generate_vit_graph(num_blocks, split_blocks=split_blocks)
    else:
        raise ValueError("Unsupported graph_type. Use 'WS', 'BA', or 'vit'.")

    # 토폴로지 정렬
    topo_order = list(nx.topological_sort(G))

    # 노드에 모듈 할당
    node_module_dict = {}
    for idx, node in enumerate(topo_order):
        module = modules[idx]
        node_module_dict[str(node)] = module

    # 새로운 모델 인스턴스 생성
    random_model = RandomGraphTransformer(G, node_module_dict, vit_model)
    return random_model


def get_modified_vit_model(vit_model, m=16, k=3, use_residual=True, drop_path_rate=0.0):
    return ModifiedViTModel(vit_model, m=m, k=k, use_residual=use_residual, drop_path_rate=drop_path_rate)
