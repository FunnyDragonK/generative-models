import torch
from torch import nn

from sgm.data.personalized import per_img_token_list
from transformers import CLIPTokenizer
from functools import partial
import open_clip
import itertools


DEFAULT_PLACEHOLDER_TOKEN = ["*"]

PROGRESSIVE_SCALE = 2000

def get_clip_token_for_string(tokenizer, string):
    batch_encoding = tokenizer(string, truncation=True, max_length=77, return_length=True,
                               return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
    tokens = batch_encoding["input_ids"]
    assert torch.count_nonzero(tokens - 49407) == 2, f"String '{string}' maps to more than a single token. Please use another string"
    return tokens
    # return tokens[0, 1]

def get_bert_token_for_string(tokenizer, string):
    token = tokenizer(string)
    assert torch.count_nonzero(token) == 3, f"String '{string}' maps to more than a single token. Please use another string"

    token = token[0, 1]

    return token

def get_embedding_for_clip_token(embedder, token):
    return embedder(token)


class EmbeddingManager(nn.Module):
    def __init__(
            self,
            embedder,
            placeholder_strings=None,
            initializer_words=None,
            per_image_tokens=False,
            num_vectors_per_token=1,
            progressive_words=False,
            **kwargs
    ):
        super().__init__()

        self.string_to_token_dict = {}
        self.string_to_token_dict_oc = {}
        
        self.string_to_param_dict = nn.ParameterDict()

        self.initial_embeddings = nn.ParameterDict() # These should not be optimized

        self.string_to_param_dict_oc = nn.ParameterDict()

        self.initial_embeddings_oc = nn.ParameterDict() # These should not be optimized

        self.progressive_words = progressive_words
        self.progressive_counter = 0

        self.max_vectors_per_token = num_vectors_per_token

        if hasattr(embedder.embedders[0], 'tokenizer'): # using Stable Diffusion's CLIP encoder
            self.is_clip = True
            get_token_for_string = partial(get_clip_token_for_string, embedder.embedders[0].tokenizer)
            get_embedding_for_tkn = partial(get_embedding_for_clip_token, embedder.embedders[0])
            token_dim = 768
        else: # using LDM's BERT encoder
            self.is_clip = False
            get_token_for_string = partial(get_bert_token_for_string, embedder.tknz_fn)
            get_embedding_for_tkn = embedder.transformer.token_emb
            token_dim = 1280

        if per_image_tokens:
            placeholder_strings.extend(per_img_token_list)

        for idx, placeholder_string in enumerate(placeholder_strings):
            
            tokens = get_token_for_string(placeholder_string)
            token = tokens[0, 1]

            if initializer_words and idx < len(initializer_words):
                init_word_token = get_token_for_string(initializer_words[idx])

                with torch.no_grad():
                    init_word_embedding = get_embedding_for_tkn(init_word_token.cpu()) # init_word_embedding.shape = [token_dim]
                    init_word_embedding = init_word_embedding[0, 1].cpu()

                token_params = torch.nn.Parameter(init_word_embedding.unsqueeze(0).repeat(num_vectors_per_token, 1), requires_grad=True)
                self.initial_embeddings[placeholder_string] = torch.nn.Parameter(init_word_embedding.unsqueeze(0).repeat(num_vectors_per_token, 1), requires_grad=False)
            else:
                token_params = torch.nn.Parameter(torch.rand(size=(num_vectors_per_token, token_dim), requires_grad=True))
            
            self.string_to_token_dict[placeholder_string] = token
            self.string_to_param_dict[placeholder_string] = token_params

        for idx, placeholder_string in enumerate(placeholder_strings):
            tokens = open_clip.tokenize(placeholder_string)
            token = tokens[0, 1]
            if initializer_words and idx < len(initializer_words):
                init_word_token = open_clip.tokenize(initializer_words[idx])

                with torch.no_grad():
                    init_word_embedding = embedder.embedders[1].model.token_embedding(init_word_token.cpu())
                    init_word_embedding = init_word_embedding[0, 1].cpu()
                token_params = torch.nn.Parameter(init_word_embedding.unsqueeze(0).repeat(num_vectors_per_token, 1), requires_grad=True)
                self.initial_embeddings_oc[placeholder_string] = torch.nn.Parameter(init_word_embedding.unsqueeze(0).repeat(num_vectors_per_token, 1), requires_grad=False)
            else:
                token_params = torch.nn.Parameter(torch.rand(size=(num_vectors_per_token, token_dim), requires_grad=True))
            self.string_to_token_dict_oc[placeholder_string] = token
            self.string_to_param_dict_oc[placeholder_string] = token_params

    def forward(
            self,
            tokenized_text,
            embedded_text,
            oc = False
    ):
        b, n, device = *tokenized_text.shape, tokenized_text.device

        for placeholder_string, placeholder_token in self.string_to_token_dict.items():

            if oc:
                placeholder_embedding = self.string_to_param_dict_oc[placeholder_string].to(device)
                placeholder_token = self.string_to_token_dict_oc[placeholder_string]
            else:
                placeholder_embedding = self.string_to_param_dict[placeholder_string].to(device)
            # print('placeholder_embedding')
            # print(placeholder_embedding)

            if self.max_vectors_per_token == 1: # If there's only one vector per token, we can do a simple replacement
                placeholder_idx = torch.where(tokenized_text == placeholder_token.to(device))
                embedded_text[placeholder_idx] = placeholder_embedding
                #print(f'replaced {placeholder_idx} tokens with {placeholder_string}')
            else: # otherwise, need to insert and keep track of changing indices
                if self.progressive_words:
                    self.progressive_counter += 1
                    max_step_tokens = 1 + self.progressive_counter // PROGRESSIVE_SCALE
                else:
                    max_step_tokens = self.max_vectors_per_token

                num_vectors_for_token = min(placeholder_embedding.shape[0], max_step_tokens)

                placeholder_rows, placeholder_cols = torch.where(tokenized_text == placeholder_token.to(device))

                if placeholder_rows.nelement() == 0:
                    continue

                sorted_cols, sort_idx = torch.sort(placeholder_cols, descending=True)
                sorted_rows = placeholder_rows[sort_idx]

                for idx in range(len(sorted_rows)):
                    row = sorted_rows[idx]
                    col = sorted_cols[idx]

                    new_token_row = torch.cat([tokenized_text[row][:col], placeholder_token.repeat(num_vectors_for_token).to(device), tokenized_text[row][col + 1:]], axis=0)[:n]
                    new_embed_row = torch.cat([embedded_text[row][:col], placeholder_embedding[:num_vectors_for_token], embedded_text[row][col + 1:]], axis=0)[:n]

                    embedded_text[row]  = new_embed_row
                    tokenized_text[row] = new_token_row

        return embedded_text

    def save(self, ckpt_path):
        torch.save({"string_to_token": self.string_to_token_dict,
                    "string_to_param": self.string_to_param_dict,
                    "string_to_token_oc": self.string_to_token_dict_oc,
                    "string_to_param_oc": self.string_to_param_dict_oc}, ckpt_path)

    def load(self, ckpt_path, is_norm=False):
        ckpt = torch.load(ckpt_path, map_location='cpu')

        self.string_to_token_dict = ckpt["string_to_token"]
        self.string_to_param_dict = ckpt["string_to_param"]

        self.string_to_token_dict_oc = ckpt["string_to_token_oc"]
        self.string_to_param_dict_oc = ckpt["string_to_param_oc"]
        if is_norm:
            temp = self.string_to_param_dict['*']
            temp1 = self.initial_embeddings['*']
            temp = temp /(temp.norm() / temp1.norm())
            self.string_to_param_dict['*'] = temp

            temp_oc = self.string_to_param_dict_oc['*']
            temp1_oc = self.initial_embeddings_oc['*']
            temp_oc = temp_oc /(temp_oc.norm() / temp1_oc.norm())
            self.string_to_param_dict_oc['*'] = temp_oc


    def get_embedding_norms_squared(self):
        all_params = torch.cat(list(self.string_to_param_dict.values()), axis=0) # num_placeholders x embedding_dim
        param_norm_squared = (all_params * all_params).sum(axis=-1)              # num_placeholders

        return param_norm_squared

    def embedding_parameters(self):
        # union of all parameters
        union_parameters = itertools.chain(self.string_to_param_dict.parameters(), self.string_to_param_dict_oc.parameters())
        return union_parameters
        # return self.string_to_param_dict.parameters()

    def embedding_to_coarse_loss(self):
        
        loss = 0.
        num_embeddings = len(self.initial_embeddings)

        for key in self.initial_embeddings:
            optimized = self.string_to_param_dict[key]
            coarse = self.initial_embeddings[key].clone().to(optimized.device)

            loss = loss + (optimized - coarse) @ (optimized - coarse).T / num_embeddings

        for key in self.initial_embeddings:
            optimized = self.string_to_param_dict_oc[key]
            coarse = self.initial_embeddings_oc[key].clone().to(optimized.device)

            loss = loss + (optimized - coarse) @ (optimized - coarse).T / num_embeddings

        return loss[0][0]
