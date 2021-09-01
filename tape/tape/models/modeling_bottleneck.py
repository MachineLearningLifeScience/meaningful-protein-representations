import torch
import torch.nn as nn
import torch.nn.functional as F
from tape import ProteinModel, ProteinConfig
from tape.models.modeling_utils import SequenceToSequenceClassificationHead
from tape.registry import registry
from .modeling_utils import LayerNorm, MLMHead
from .modeling_bert import ProteinBertModel, ProteinBertConfig
from .modeling_lstm import ProteinLSTMModel, ProteinLSTMConfig
from .modeling_resnet import ProteinResNetModel, ProteinResNetConfig


class BottleneckConfig(ProteinConfig):
    def __init__(self,
                 hidden_size: int = 1024,
                 max_size: int = 300,
                 backend_name: str = 'resnet',
                 **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.max_size = max_size
        self.backend_name = backend_name


class BottleneckAbstractModel(ProteinModel):
    """ All your models will inherit from this one - it's used to define the
        config_class of the model set and also to define the base_model_prefix.
        This is used to allow easy loading/saving into different models.
    """
    config_class = BottleneckConfig
    base_model_prefix = 'bottleneck'

    def __init__(self, config):
        super().__init__(config)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                module.bias.data.zero_()
        # elif isinstance(module, ProteinResNetBlock):
            # nn.init.constant_(module.bn2.weight, 0)

@registry.register_task_model('embed', 'bottleneck')
class ProteinBottleneckModel(BottleneckAbstractModel):

    def __init__(self, config):
        super().__init__(config)
        if config.backend_name == 'resnet':
            config = ProteinResNetConfig()
            self.backbone1 = ProteinResNetModel(config)
        elif config.backend_name == 'transformer':
            config = ProteinBertConfig()
            self.backbone1 = ProteinBertModel(config)
        elif config.backend_name == 'lstm':
            config = ProteinLSTMConfig(hidden_size=256)
            self.backbone1 = ProteinLSTMModel(config)
            config.hidden_size = config.hidden_size * 2
        else:
            raise ValueError('Somethings wrong')
        self.linear1 = nn.Linear(self.config.max_size*config.hidden_size, self.config.hidden_size)
        self.linear2 = nn.Linear(self.config.hidden_size, self.config.max_size*config.hidden_size)

    def forward(self, input_ids, input_mask=None):
        pre_pad_shape = input_ids.shape[1]
        if pre_pad_shape >= self.config.max_size:
            input_ids = input_ids[:,:self.config.max_size]
            if not input_mask is None:
                input_mask = input_mask[:,:self.config.max_size]
        else:
            input_ids = F.pad(input_ids, (0, self.config.max_size - pre_pad_shape))
            if not input_mask is None:
                input_mask = F.pad(input_mask, (0, self.config.max_size - pre_pad_shape))
        assert input_ids.shape[1] == self.config.max_size

        output = self.backbone1(input_ids, input_mask)
        sequence_output = output[0]
        pre_shape = sequence_output.shape
        embeddings = self.linear1(sequence_output.reshape(sequence_output.shape[0], -1))
        sequence_output = self.linear2(embeddings).reshape(*pre_shape)
        sequence_output = sequence_output[:,:pre_pad_shape]
        outputs = (sequence_output, embeddings) + output[2:]
        return outputs

@registry.register_task_model('beta_lactamase', 'bottleneck')
@registry.register_task_model('masked_language_modeling', 'bottleneck')
@registry.register_task_model('language_modeling', 'bottleneck')
class ProteinBottleneckForPretraining(BottleneckAbstractModel):
    
    def __init__(self, config):
        super().__init__(config)
        self.backbone1 = ProteinBottleneckModel(config)
        
        if config.backend_name == 'resnet':
            config = ProteinResNetConfig()
            self.backbone2 = MLMHead(config.hidden_size, config.vocab_size, config.hidden_act, 
                                     config.layer_norm_eps, ignore_index=-1)
        elif config.backend_name == 'transformer':
            config = ProteinBertConfig()
            self.backbone2 = MLMHead(config.hidden_size, config.vocab_size, config.hidden_act, 
                                     config.layer_norm_eps, ignore_index=-1)
        elif config.backend_name == 'lstm':
            config = ProteinLSTMConfig(hidden_size=256)
            self.backbone2 = nn.Linear(config.hidden_size, config.vocab_size)
            config.hidden_size = config.hidden_size * 2
        else:
            raise ValueError('Somethings wrong')

    def forward(self,
                input_ids,
                input_mask=None,
                targets=None):
        if input_ids.shape[1]>self.config.max_size:
            targets = targets[:,:self.config.max_size]
            
        outputs = self.backbone1(input_ids, input_mask)
        sequence_output = outputs[0]
        if self.config.backend_name == 'resnet' or self.config.backend_name == 'transformer':
            outputs = self.backbone2(sequence_output, targets) + outputs[2:]
        elif self.config.backend_name == 'lstm':
            sequence_output, pooled_output = outputs[:2]

            forward_prediction, reverse_prediction = sequence_output.chunk(2, -1)
            forward_prediction = F.pad(forward_prediction[:, :-1], [0, 0, 1, 0])
            reverse_prediction = F.pad(reverse_prediction[:, 1:], [0, 0, 0, 1])
            prediction_scores = \
                self.backbone2(forward_prediction) + self.backbone2(reverse_prediction)
            prediction_scores = prediction_scores.contiguous()

            # add hidden states and if they are here
            outputs = (prediction_scores,) + outputs[2:]

            if targets is not None:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
                lm_loss = loss_fct(
                    prediction_scores.view(-1, 30), targets.view(-1))
                outputs = (lm_loss,) + outputs           

        # (loss), prediction_scores, (hidden_states), (attentions)
        return outputs
