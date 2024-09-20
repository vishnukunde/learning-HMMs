import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional
from transformers import GPT2Model, GPT2Config
from tqdm import tqdm
import os


class TransformerModel(nn.Module):
    def __init__(self, 
                seq_len=50,
                n_hidden_states=2, 
                n_obs_states=2,
                n_embd=16, 
                n_layer=3, 
                n_head=4):
        super(TransformerModel, self).__init__()
        configuration = GPT2Config(
            n_positions=2 * seq_len,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )

        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.n_hidden_states = n_hidden_states
        self.n_obs_states = n_obs_states
        self.seq_len = seq_len
        self.n_states = self.n_hidden_states + self.n_obs_states

        self.n_positions = 2*seq_len
        self._read_in = nn.Linear(self.n_obs_states+self.n_hidden_states, n_embd) # learn seperate embeddings for each of the hidden states and obs states
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, n_hidden_states)  # logits for each of n_hidden_states


    def forward(self, obs_state_ids, hidden_state_ids=None):
        if hidden_state_ids is not None:
            ids = torch.concat((obs_state_ids, self.n_obs_states+hidden_state_ids), dim=-1)
            # print('ids = ', ids.shape)
            generate = False
        else:
            ids = obs_state_ids
            generate = True

        one_hot_tokens = torch.eye(self.n_states)[ids]

        # print('one_hot_tokens = ', one_hot_tokens.shape)
        
        if not generate:  # train the model
            embeds = self._read_in(one_hot_tokens)
            output = self._backbone(inputs_embeds=embeds).last_hidden_state
            logits = self._read_out(output)
            # print('all logits = ', logits.shape)
            x_logits = (logits[:, self.seq_len-1:-1, :]).reshape(-1,self.n_hidden_states)  # the logits for x0 only
            x_probs = torch.nn.functional.softmax(x_logits, dim=-1)
            return x_logits, x_probs
        if generate:  # generate the posterior probs
            embeds = self._read_in(one_hot_tokens)
            output = self._backbone(inputs_embeds=embeds).last_hidden_state
            logits = self._read_out(output)



# Function to generate the hidden states and observations
def generate_data(N, A, p_BSC, pi, batch_size=128):
    K=2
    x = np.zeros((batch_size, N), dtype=int)
    y = np.zeros((batch_size, N), dtype=int)
    for b in range(batch_size):
        x[b,0] = np.random.choice(K, p=pi)
        y[b,0] = x[b,0] if np.random.rand() >= p_BSC else (1-x[b,0])
        for t in range(1, N):
            x[b,t] = np.random.choice(K, p=A[x[b,t-1], :])
            y[b,t] = x[b,t] if np.random.rand() >= p_BSC else (1-x[b,t])
    x_tensor, y_tensor = torch.from_numpy(x),torch.from_numpy(y)
    return x_tensor, y_tensor
    

def train(model, model_id=0, p_BSC=0.3, p0=0.9, p1=0.1, max_epochs=50000, lr=0.001, seq_len=50):

    A = np.array([[p0,1-p0],[1-p1,p1]])
    pi = np.array([0.5,0.5])

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    CEloss = torch.nn.CrossEntropyLoss()

    pbar = tqdm(range(1, max_epochs+1))

    for n in pbar:

        x_seq, y_seq = generate_data(N=seq_len, A=A, p_BSC=p_BSC, pi=pi)
        targets = x_seq.flatten()

        x_logits, x_probs = model(obs_state_ids=y_seq, hidden_state_ids=x_seq)

        loss = CEloss(x_logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description(f'loss = {loss}')

    torch.save(model.state_dict(), os.path.join('../trained_models', f"model_{model_id}.pt"))



# Main program

seq_len = 50
n_hidden_states = 2
n_obs_states = 2
n_embd = 16
n_layer = 3 
n_head = 4

model_id = 0


# Traning hyperparameters

p_BSC=0.3
p0=0.9
p1=0.1
max_epochs=50000
lr=0.001

# Train the model

my_model = TransformerModel(seq_len = 50,
                            n_hidden_states = 2,
                            n_obs_states = 2,
                            n_embd = 16,
                            n_layer = 3,
                            n_head = 4)

train(model=my_model, p_BSC=p_BSC, p0=p0, p1=p1, max_epochs=max_epochs, lr=lr, seq_len=seq_len)














