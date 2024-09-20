import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional
from transformers import GPT2Model, GPT2Config
from tqdm import tqdm
import os
import matplotlib.pyplot as plt


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
        self._read_in = nn.Linear(self.n_states, n_embd) # learn seperate embeddings for each of the hidden states and obs states
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, n_hidden_states)  # logits for each of n_hidden_states


    def forward(self, obs_state_ids, hidden_state_ids=None, n_symbs=None):
        batch_size, context_len = obs_state_ids.shape
        if hidden_state_ids is not None:
            ids = torch.concat((obs_state_ids, self.n_obs_states+hidden_state_ids), dim=-1)
            generate = False
        else:
            ids = obs_state_ids
            generate = True
            dummy_obs_one_hot = torch.zeros(batch_size, 1, self.n_obs_states)

        one_hot_tokens = torch.eye(self.n_states)[ids]

        if n_symbs is None:
            n_symbs = self.seq_len

        if not generate:  # train the model
            embeds = self._read_in(one_hot_tokens)
            output = self._backbone(inputs_embeds=embeds).last_hidden_state
            logits = self._read_out(output)
            # print('all logits = ', logits.shape)
            x_logits = (logits[:, context_len-1:-1, :]).reshape(-1,self.n_hidden_states)  # the logits for x0 only
            x_probs = torch.nn.functional.softmax(x_logits, dim=-1)
            return x_logits, x_probs
        if generate:  # generate the posterior probs

            x_post_dist = torch.zeros(batch_size, n_symbs, self.n_hidden_states)
            x_hat = torch.zeros(batch_size, n_symbs)

            for t in range(n_symbs):
                embeds = self._read_in(one_hot_tokens)
                output = self._backbone(inputs_embeds=embeds).last_hidden_state
                logits = self._read_out(output)
                x_logits = logits[:, -1, :]
                x_probs = torch.nn.functional.softmax(x_logits, dim=-1) # posterior gamma[t] of the x[t]  (batch_size, 1, n_hidden_states)
                x_post_dist[:, t, :] = x_probs
                x_hat[:, t] = torch.argmax(x_probs, dim=-1)
                one_hot_token = torch.concat((dummy_obs_one_hot, (x_probs).unsqueeze(dim=1)), dim=-1)  # (batch_size, 1, n_obs_states+n_hidden_states)
                one_hot_tokens = torch.concat((one_hot_tokens, one_hot_token), dim=1)  # (batch_size, n_tkns+1, n_obs_states+n_hidden_states)
            return x_post_dist, x_hat


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


class HMM:
    def __init__(self, p0, p1, p_BSC):
        self.p0 = p0
        self.p1 = p1
        self.p_BSC = p_BSC
        self.A = np.array([[p0,1-p0],[1-p1,p1]])
    

def train(model, HMM_list, model_id=0, max_epochs=50000, lr=0.001, seq_len=50):

    pi = np.array([0.5,0.5])

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    CEloss = torch.nn.CrossEntropyLoss()

    pbar = tqdm(range(1, max_epochs+1))

    n_classes = len(HMM_list)

    model.train()

    for n in pbar:

        curr_HMM = HMM_list[np.random.randint(n_classes)]

        context_len = np.random.randint(10, seq_len+1)

        x_seq, y_seq = generate_data(N=context_len, A=curr_HMM.A, p_BSC=curr_HMM.p_BSC, pi=pi)
        targets = x_seq.flatten()

        x_logits, x_probs = model(obs_state_ids=y_seq, hidden_state_ids=x_seq)

        loss = CEloss(x_logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description(f'loss = {loss}')

    torch.save(model.state_dict(), os.path.join('../trained_models', f"model_{model_id}_joint_training.pt"))


def generate_post_prob(model, curr_HMM, seq_len=50, batch_size=100, permute=False):

    pi = np.array([0.5,0.5])

    x_seq, y_seq = generate_data(N=seq_len, A=curr_HMM.A, p_BSC=curr_HMM.p_BSC, pi=pi, batch_size=batch_size)

    model.eval()

    accuracy = np.zeros(seq_len+1)

    for n_symbs in range(1, seq_len+1):

        # Generate the x_seq sequence after y_seq sequentially

        if permute and n_symbs >= 5:
            y_short_seq = y_seq[:,:n_symbs]
            perm_mask = np.random.permutation(n_symbs-1)
            print('perm_mask = ', perm_mask)
            y_perm = y_short_seq[:, perm_mask]
            print('y_perm = ', y_perm.shape)
            y_in = torch.concat((y_perm, y_short_seq[:,-1].unsqueeze(-1)), dim=-1)
            print('y_in = ', y_in.shape)
        else:
            y_in = y_seq[:,:n_symbs]

        x_prob_dist, x_hat = model(obs_state_ids = y_in, 
                                n_symbs = n_symbs)
        
        # x_prob_dist, x_hat = model(obs_state_ids = y_seq)

        accuracy[n_symbs] = (torch.mean(1.0*(x_hat == x_seq[:,:n_symbs])) * 100).detach().cpu().numpy()

        print('x_hat = ', x_hat[0])
        print('x_seq = ', x_seq[0,:n_symbs])
        print(f'acc = {accuracy[n_symbs]}%')

    return accuracy

    


# Main program

# model_id = 1
# seq_len = 50
# n_hidden_states = 2
# n_obs_states = 2
# n_embd = 16
# n_layer = 3
# n_head = 4
# p_BSC=0.3
# p0=0.7
# p1=0.3

# model_id = 2
# seq_len = 50
# n_hidden_states = 2
# n_obs_states = 2
# n_embd = 32
# n_layer = 6 
# n_head = 4
# p_BSC=0.3
# p0=0.7
# p1=0.3

# model_id = 3 
# model_id = 4
model_id = 5
seq_len = 50
n_hidden_states = 2
n_obs_states = 2
n_embd = 16
n_layer = 3
n_head = 4
p_BSC=0.3
p0=0.95
p1=0.95
batch_size = 500
HMM0 = HMM(p0=0.95, p1=0.95, p_BSC=0.3)
HMM_list = [HMM0]

# max_epochs = 10000

# model_id = 6
# seq_len = 50
# n_hidden_states = 2
# n_obs_states = 2
# n_embd = 16
# n_layer = 3
# n_head = 4
# HMM0 = HMM(p0=0.95, p1=0.95, p_BSC=0.3)
# HMM1 = HMM(p0=0.5, p1=0.5, p_BSC=0.3)
# HMM_list = [HMM0, HMM1]
# max_epochs = 10000
# batch_size = 128

# Traning hyperparameters

# max_epochs=50000
lr=0.0005

# Train the model

my_model = TransformerModel(seq_len = seq_len,
                            n_hidden_states = n_hidden_states,
                            n_obs_states = n_obs_states,
                            n_embd = n_embd,
                            n_layer = n_layer,
                            n_head = n_head)

# train(model=my_model, model_id=model_id, HMM_list=HMM_list, max_epochs=max_epochs, lr=lr, seq_len=seq_len)

# my_model.load_state_dict(torch.load(f'../trained_models/model_{model_id}_joint_training.pt'))
my_model.load_state_dict(torch.load(f'../trained_models/model_{model_id}.pt'))

# Plot the accuracies

for HMM_id, curr_HMM in enumerate(HMM_list):
    plt.figure(figsize=(10,10))
    accuracy_perm = generate_post_prob(my_model, curr_HMM=curr_HMM, seq_len=seq_len, batch_size=batch_size, permute=True)
    accuracy_not_perm = generate_post_prob(my_model, curr_HMM=curr_HMM, seq_len=seq_len, batch_size=batch_size, permute=False)
    plt.plot(accuracy_perm, color='black', lw=2, ls='dashed', label='permuted')
    plt.plot(accuracy_not_perm, color='darkgreen', lw=2, label='not permuted')
    plt.title(f'HMM: p(BSC) = {curr_HMM.p_BSC}, p0 = {curr_HMM.p0}, p1 ={curr_HMM.p1}')
    plt.xlabel('Sequence length (N)')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid()
    plt.savefig(f'../plots/permuted-vs-not-permuted-HMM-{HMM_id}-p_BSC-{curr_HMM.p_BSC}-p0-{curr_HMM.p0}-p1-{curr_HMM.p1}.png', dpi=400, bbox_inches='tight')











