# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default="dataset",
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--dataset",
                        default="dnli",
                        type=str)
    parser.add_argument("--output_dir",
                        default="saved_models",
                        type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--max_seq_length",
                        default=100,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case",
                        default=True,
                        type=bool,
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=4,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=4,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=1e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=2,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=2,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--optimize_on_cpu',
                        default=False,
                        action='store_true',
                        help="Whether to perform optimization and keep the optimizer averages on CPU")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument("--bert_config_file",
                        type=str,
                        default="bert_config.json",
                        help="The config json file corresponding to the pre-trained BERT model. \n"
                             "This specifies the model architecture.")
    parser.add_argument("--load_model",type=str,default="pytorch_model.bin",help="Path to the PyTorch model.")
    parser.add_argument("--tau",type=float,default=1.0,help="kd distillation temperature")
    parser.add_argument("--alpha",type=float,default=0.99,help="weight for combining two loss")
    parser.add_argument("--train_name",default="train")
    parser.add_argument("--dev_name",default="dev")
    parser.add_argument("--test_name",default="test")
    parser.add_argument("--depth",type=int)
    parser.add_argument('--vocab_file',type=str,default="vocab.txt")
    parser.add_argument("--small",action='store_true',help = "for temporary debugging")
    parser.add_argument("--log_step",type=int,default=5)
    parser.add_argument("--eval_step",type=int,default=20)
    args = parser.parse_args()
    return args
