device = 'cuda:1'
devices = [1]

epochs = 10

input_feat_dim=2
enc_hidden_dim=16
dec_hidden_dim=32
num_res_blocks=2
codebook_dim= 128 
codebook_slots= 64
VQVAE_PATH = f'src/SavedModels/VQVAE_epochs_{epochs}.pt'

classes = ["4ask","8pam","16psk","32qam_cross","2fsk","ofdm-256"]
iq_samples = 1024
samples_per_class= 1000
batch_size = 32

eff_net_PATH = f'src/SavedModels/efficientNet_epochs_{epochs}.pt'
in_channels = 2


block_size = 512
vocab_size = codebook_slots
n_head = 16
n_layer = 16
n_embd = 32
dropout = 0.1
TRANSFORMER_MODEL_PATH = f'src/SavedModels/Transformer_epochs_{epochs}.pt'
MONAI_TRANSFORMER_MODEL_PATH = f'src/SavedModels/MonaiTransformer_epochs_{epochs}.pt'
eval_folder = f'EvaluationResults/Monai_results/'