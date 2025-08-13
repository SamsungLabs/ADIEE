import torch
import torch.nn as nn
from transformers import PreTrainedModel, LlavaNextForConditionalGeneration
from transformers.models.llava_next import LlavaNextConfig
from utils_generation import generate

def init_weights(m):
    """
    xavier model weights initialization
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class ADIEEScorer(PreTrainedModel):
    config_class = LlavaNextConfig
    def __init__(self, config, **kwargs):
        super().__init__(config)

        # initialize the vlm backbone with existing copy in kwargs if possible
        if "vlm" in kwargs:
            self.vlm = kwargs["vlm"]
        else:
            self.vlm = LlavaNextForConditionalGeneration._from_config(config)

        # define score token decoder 
        if "score_token_idx" in kwargs:
            self.score_token_idx = kwargs["score_token_idx"]
        else:
            self.score_token_idx = None
        in_channel = self.vlm.config.text_config.hidden_size
        self.out_channels = kwargs["out_channels"]
        self.score_decoder = nn.Sequential(
            nn.Linear(in_channel, 256),
            nn.ReLU(),
            nn.Linear(256, 16),
            nn.ReLU(),
            nn.Linear(16, self.out_channels),
        )
        init_weights(self.score_decoder)

    def forward(self, **kwargs):
        # if model ground-truth output (label) is provided, we are at the training stage
        # else we are the testing stage
        if kwargs["labels"] is not None:
            return self.forward_train(**kwargs)
        else:
            return self.inference(**kwargs)
    
    def get_score(self, ids, last_hidden_states):
        """
        Decode score from score_token embedding
        """
        score_token_mask = ids == self.score_token_idx

        # ensure there is one score_token
        for i in range(score_token_mask.shape[0]):
            count = score_token_mask[i].int().sum().item()
            if count == 0:
                score_token_mask[i, -1] = True
            elif count > 1:
                true_indices = torch.nonzero(score_token_mask[i])
                score_token_mask[i, true_indices[:-1]] = False
            assert score_token_mask[i].int().sum().item() == 1, score_token_mask[i].int().sum().item()
            
        pred_embed = last_hidden_states[score_token_mask]
        score = self.score_decoder(pred_embed)
        return score

    def forward_train(self, **kwargs):
        """
        forward function for training
        """
        outputs = self.vlm(
            output_hidden_states=True,
            **kwargs,
        )
        last_hidden_states = outputs.hidden_states[-1]
        score = self.get_score(kwargs["input_ids"], last_hidden_states)
        
        # return score and output cross-entropy loss
        return {
            "score": score,
            "ce_loss": outputs.loss
        }

    def inference(self, **kwargs):
        """
        forward function for validation
        """
        outputs = self.vlm.generate(
            max_new_tokens=8,
            num_beams=1,
            output_hidden_states=True,
            return_dict_in_generate=True,
            **kwargs,
        )

        # get the embedding of the output
        hs_list = []
        for hs in outputs.hidden_states:
            hs_list.append(hs[-1][:, -1:, :]) 
        output_ids = outputs.sequences
        output_ids = output_ids[:, -len(hs_list):]
        last_hidden_states = torch.cat(hs_list, dim=1)

        # decode score from the embedding corresponds to the score token
        score = self.get_score(output_ids, last_hidden_states)

        # return score and model text output
        if self.out_channels == 1:
            score_name = "score"
        else:
            score_name = "class_prob"
        return {
            score_name: score,
            "output_ids": output_ids,
        }


    def inference_refl(self, **kwargs):
        """
        forward function for validation
        """
        outputs = generate(
            self.vlm,
            max_new_tokens=8,
            num_beams=1,
            output_hidden_states=True,
            return_dict_in_generate=True,
            **kwargs,
        )

        # get the embedding of the output
        hs_list = []
        for hs in outputs.hidden_states:
            hs_list.append(hs[-1][:, -1:, :]) 
        output_ids = outputs.sequences
        output_ids = output_ids[:, -len(hs_list):]
        last_hidden_states = torch.cat(hs_list, dim=1)

        # decode score from the embedding corresponds to the score token
        score = self.get_score(output_ids, last_hidden_states)

        # return score and model text output
        if self.out_channels == 1:
            score_name = "score"
        else:
            score_name = "class_prob"
        return {
            score_name: score,
            "output_ids": output_ids,
        }
