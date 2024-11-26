import torch
from torch import nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, PreTrainedModel
import argparse
import copy
import networkx as nx
import random

def generate_graph(n_nodes, graph_type='custom'):
    if graph_type == 'original':
        # GPT-2와 동일한 순차적인 그래프 생성
        G = nx.DiGraph()
        G.add_nodes_from(range(n_nodes))
        for i in range(n_nodes - 1):
            G.add_edge(i, i + 1)
    else:
        # 사용자 지정 그래프 생성 (예: scale-free 그래프)
        G = nx.DiGraph()
        G.add_nodes_from(range(n_nodes))
        for i in range(n_nodes - 1):
            G.add_edge(i, i + 1)

        G_ = nx.scale_free_graph(n_nodes)
        #G_ = nx.watts_strogatz_graph(n_nodes, 4, 0.25)
        G_ = nx.DiGraph([(u, v) for (u, v) in G_.edges() if u < v])

        G = nx.compose(G, G_)

    return G

class GraphGPT2Model(nn.Module):
    def __init__(self, config, n_nodes, use_residual=True, graph_type='custom', layer_sharing=True):
        super().__init__()
        self.config = config
        self.use_residual = use_residual
        self.graph_type = graph_type
        self.layer_sharing = layer_sharing
        self.n_nodes = n_nodes

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)

        # Transformer 블록 생성
        n_blocks = config.n_layer
        base_block = GPT2LMHeadModel(config).transformer.h[0]
        if self.layer_sharing:
            self.h = nn.ModuleList([copy.deepcopy(base_block) for _ in range(n_blocks)])
        else:
            self.h = nn.ModuleList([copy.deepcopy(base_block) for _ in range(self.n_nodes)])

        # 그래프 생성 및 노드-블록 매핑
        self.G = generate_graph(self.n_nodes, graph_type=self.graph_type)

        # 노드와 블록의 매핑 생성
        self.node_to_block = {}
        if self.layer_sharing:
            topo_order = list(nx.topological_sort(self.G))
            for idx, node in enumerate(topo_order):
                if idx < n_blocks:
                    self.node_to_block[node] = self.h[idx]
                else:
                    block_idx = random.randint(0, n_blocks - 1)
                    self.node_to_block[node] = self.h[block_idx]
        else:
            for idx, node in enumerate(self.G.nodes()):
                self.node_to_block[node] = self.h[idx]

        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        # 각 노드의 이전 노드들에 대한 가중치 파라미터 생성
        self.edge_weights = nn.ParameterDict()
        for node in self.G.nodes():
            predecessors = list(self.G.predecessors(node))
            if predecessors:
                weight = nn.Parameter(torch.randn(len(predecessors)))
                self.edge_weights[str(node)] = weight

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("input_ids와 inputs_embeds를 동시에 지정할 수 없습니다.")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("input_ids나 inputs_embeds 중 하나를 지정해야 합니다.")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        if position_ids is None:
            position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)

        node_outputs = {}
        topo_order = list(nx.topological_sort(self.G))

        for node in topo_order:
            predecessors = list(self.G.predecessors(node))
            if predecessors:
                if len(predecessors) == 1:
                    combined_input = node_outputs[predecessors[0]]
                else:
                    pred_outputs = [node_outputs[pred] for pred in predecessors]
                    weights = self.edge_weights[str(node)]
                    softmax_weights = torch.softmax(weights, dim=0)
                    pred_outputs_tensor = torch.stack(pred_outputs)  # [num_preds, batch_size, seq_len, hidden_size]
                    softmax_weights = softmax_weights.view(-1, 1, 1, 1)  # Broadcasting을 위해 reshape
                    combined_input = torch.sum(softmax_weights * pred_outputs_tensor, dim=0)
            else:
                combined_input = hidden_states

            block = self.node_to_block[node]
            block_output = block(
                combined_input,
                layer_past=None,
                attention_mask=attention_mask,
                head_mask=head_mask,
                use_cache=use_cache,
            )

            block_hidden_states = block_output[0]

            if self.use_residual:
                block_hidden_states = combined_input + block_hidden_states

            node_outputs[node] = block_hidden_states

        # 최종 출력 결정
        if self.graph_type == 'original':
            final_output = node_outputs[topo_order[-1]]
        else:
            output_nodes = [node for node in self.G.nodes() if self.G.out_degree(node) == 0]
            if output_nodes:
                final_output = torch.stack([node_outputs[node] for node in output_nodes]).sum(dim=0)
            else:
                final_output = node_outputs[topo_order[-1]]

        final_output = self.ln_f(final_output)

        return final_output

class ModifiedGPT2Model(PreTrainedModel):
    config_class = GPT2Config
    base_model_prefix = "transformer"

    def __init__(self, config, n_nodes, use_residual=True, model_type='original', graph_type='custom', layer_sharing=True):
        super().__init__(config)
        self.model_type = model_type

        if self.model_type == 'graph':
            self.transformer = GraphGPT2Model(config, n_nodes=n_nodes, use_residual=use_residual, graph_type=graph_type, layer_sharing=layer_sharing)
        else:
            self.transformer = GPT2LMHeadModel(config).transformer

        #self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head = GPT2LMHeadModel(config).lm_head

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        labels=None,
        use_cache=None,
    ):
        hidden_states = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            use_cache=use_cache,
        )

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return (loss, logits) if loss is not None else logits

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model_type = kwargs.pop('model_type', 'original')
        use_residual = kwargs.pop('use_residual', True)
        graph_type = kwargs.pop('graph_type', 'custom')
        layer_sharing = kwargs.pop('layer_sharing', True)
        n_nodes = kwargs["n_nodes"]

        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model.__init__(model.config, n_nodes=n_nodes, use_residual=use_residual, model_type=model_type, graph_type=graph_type, layer_sharing=layer_sharing)
        return model

def load_dataset(file_path, tokenizer):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=tokenizer.model_max_length
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='original', choices=['original', 'graph'],
                        help='Original GPT-2 또는 Graph 기반 모델 선택')
    parser.add_argument('--graph_type', type=str, default='custom', choices=['original', 'custom'],
                        help='그래프 유형 선택: original 또는 custom')
    parser.add_argument('--train_data', type=str, required=True, help='훈련 데이터셋 경로')
    parser.add_argument('--eval_data', type=str, required=True, help='평가 데이터셋 경로')
    parser.add_argument('--use_residual', type=bool, default=False, help='Residual connection 사용 여부')
    parser.add_argument('--n_nodes', type=int, default=16, help='그래프의 노드 수')
    parser.add_argument('--layer_sharing', type=bool, default=False, help='레이어 공유 여부')
    args = parser.parse_args()

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    if args.model_type == 'graph':
        model = ModifiedGPT2Model.from_pretrained(
            'gpt2',
            model_type='graph',
            use_residual=args.use_residual,
            graph_type=args.graph_type,
            layer_sharing=args.layer_sharing,
            n_nodes=args.n_nodes
        )
    else:
        model = GPT2LMHeadModel.from_pretrained('gpt2')

    # lm_head 로드
    model_ = GPT2LMHeadModel.from_pretrained('gpt2')

    # 모델의 매개변수를 딕셔너리로 가져옵니다.
    pretrained_params = dict(model_.named_parameters())
    modified_params = dict(model.named_parameters())

    # 가중치를 복사합니다.
    for name, param in modified_params.items():

        if "lm_head" in name:
            param.data.copy_(model_.lm_head.weight.data)
            continue

        if name in pretrained_params:
            if param.size() == pretrained_params[name].size():
                param.data.copy_(pretrained_params[name].data)
            else:
                print(f"크기 불일치: {name} - pretrained: {pretrained_params[name].size()}, modified: {param.size()}")
        else:
            print(f"매개변수 {name}이(가) pretrained 모델에 없습니다.")

    # 버퍼도 동일하게 처리합니다 (예: BatchNorm의 running_mean 등)
    pretrained_buffers = dict(model_.named_buffers())
    modified_buffers = dict(model.named_buffers())

    for name, buffer in modified_buffers.items():
        if name in pretrained_buffers:
            if buffer.size() == pretrained_buffers[name].size():
                buffer.data.copy_(pretrained_buffers[name].data)
            else:
                print(f"버퍼 크기 불일치: {name} - pretrained: {pretrained_buffers[name].size()}, modified: {buffer.size()}")
        else:
            print(f"버퍼 {name}이(가) pretrained 모델에 없습니다.")

    train_dataset = load_dataset(args.train_data, tokenizer)
    eval_dataset = load_dataset(args.eval_data, tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir='./results',
        overwrite_output_dir=True,
        num_train_epochs=100,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy='steps',
        save_steps=10_000,
        eval_steps=50,
        logging_steps=50,
        learning_rate=1e-3,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

if __name__ == '__main__':
    main()

