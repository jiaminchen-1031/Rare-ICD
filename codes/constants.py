import argparse

FULL = 'full'
TOP50 = '50'

# Debug version
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--log', default="INFO", help="Logging level.")
    parser.add_argument('--random_seed', type=int, default=271, help="Random seed.")
    parser.add_argument('--test', type=bool, default=False, help="if test")
    parser.add_argument('--adj', type=str, default='../mimicdata/adj/adj_training.pt', help="path to adj")

    parser.add_argument(
        '--name',
        type=str,
        default='test',
        help='folder name for saving'
    )

    parser.add_argument(
        '--folder_path',
        type=str,
        default='../mimicdata/generated_data/',
        help='folder name for generated data'
    )

    parser.add_argument(
        '--nfold',
        type=int,
        default=0,
        help='whether using augmentation >1 by UMLS and number of folds'
    )

    parser.add_argument(
        '--make_sentence',
        type=bool,
        default=False,
        help='whether adopting syntactical alignment lstm to code mapping'
    )

    parser.add_argument(
        '--patience_1',
        type=int,
        default=10,
        help='early stopping patience for stage 1'
    )
    
    parser.add_argument(
        '--bpe',
        type=bool,
        default=False,
        help='whether use pretrained bpe for training'
    )

    parser.add_argument(
        '--patience_2',
        type=int,
        default=10,
        help='early stopping patience for stage 2'
    )

    parser.add_argument(
        '--data_setting',
        type=str,
        default=FULL,
        help='Data Setting (full or top50)'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='ours',
        help='model to choose'
    )

    parser.add_argument(
        '--num_epoch',
        type=int,
        default=[100],
        nargs='+',
        help='Number of epochs to train.'
    )

    parser.add_argument(
        '--learning_rate',
        type=float,
        default=[0.001],
        nargs='+',
        help='Initial learning rate.'
    )

    parser.add_argument(
        '--learning_rate_fine',
        type=float,
        default=[0.0001],
        nargs='+',
        help='Initial learning rate.'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size. Must divide evenly into the dataset sizes.'
    )

    parser.add_argument(
        '--max_len',
        type=int,
        default=2500,
        help='Max Length of discharge summary'
    )

    parser.add_argument(
        '--embed_size',
        type=int,
        default=128,
        help='Embedding dimension for text token'
    )

    parser.add_argument(
        '--freeze_embed',
        action='store_true',
        default=True,
        help='Freeze CBOW embedding or fine tune'
    )

    parser.add_argument(
        '--label_attn_expansion',
        type=int,
        default=2,
        help='Expansion factor for attention model'
    )

    parser.add_argument(
        '--num_trans_layers',
        type=int,
        default=2,
        help='Number of transformer layers'
    )

    parser.add_argument(
        '--num_attn_heads',
        type=int,
        default=8,
        help='Number of transformer attention heads'
    )

    parser.add_argument(
        '--trans_forward_expansion',
        type=int,
        default=4,
        help='Factor to expand transformers hidden representation'
    )

    parser.add_argument(
        '--dropout_rate',
        type=float,
        default=0.3,
        help='Dropout rate for transformers'
    )

    parser.add_argument(
        '--hidden_size',
        type=int,
        default=256,
        help='hidden size for rnn layer'
    )
    
    parser.add_argument(
        '--rnn_layer',
        type=int,
        default=1,
        help='number of rnn layer'
    )


    parser.add_argument('--pre_file', type=str, default='checkpoint_epoch_11.pt', help="pretrained model")

    parser.add_argument('--local_rank', default=0, type=int,
                        help='node rank for distributed training')

    args = parser.parse_args()  # '--target_kernel_size 4 8'.split()
    return args


args = get_args()

PAD_SYMBOL = "<PAD>"
UNK_SYMBOL = "<UNK>"

DATA_DIR = '../mimicdata/'
CAML_DIR = '../mimicdata/caml/'

GENERATED_DIR = '../mimicdata/generated_data/'
NOTEEVENTS_FILE_PATH = '../mimicdata/NOTEEVENTS.csv'
DIAGNOSES_FILE_PATH = '../mimicdata/DIAGNOSES_ICD.csv'
PORCEDURES_FILE_PATH = '../mimicdata/PROCEDURES_ICD.csv'
DIAG_CODE_DESC_FILE_PATH = '../mimicdata/D_ICD_DIAGNOSES.csv'
PROC_CODE_DESC_FILE_PATH = '../mimicdata/D_ICD_PROCEDURES.csv'
ICD_DESC_FILE_PATH = '../mimicdata/ICD9_descriptions'

FILE_DIR = args.folder_path
EMBED_FILE_PATH = args.folder_path + 'vocab.embed'
UMLS_EMBED_FILE_PATH = args.folder_path + 'umls.embed'
CODE_FREQ_PATH = args.folder_path + 'code_freq.csv'
CODE_DESC_VECTOR_PATH = args.folder_path + 'code_desc_vectors.csv'
# PROC_DIAG_EMBEDDING = '../mimicdata/generated_umls/proc_diag_embedding.txt'
# CODE_ADJ_PATH = '../mimicdata/generated_full/adj.pt'
# EMBED_DESC_FILE_PATH = '../mimicdata/generated_bpe_sep/vocab_desc.embed'
# EMBED_LAAT_FILE_PATH = '../mimicdata/generated_laat/word2vec_sg0_100.model'