from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from module import *
import os
import torch.nn.functional as F


os.environ["CUDA_VISIBLE_DEVICES"] = "3"
formatter = '%(asctime)s %(levelname)s %(message)s'

def save_logit(model, features,args,device,name,logger):
    eval_sampler = SequentialSampler(features)
    eval_dataloader = DataLoader(features, sampler=eval_sampler, collate_fn = collate_fn,batch_size=args.eval_batch_size)
    nb_total_true = 0
    nb_eval_examples = 0
    model.eval() # NOTE this is important.
    with open(os.path.join(args.data_dir,name+"_tea_logits.tsv"),"w") as f:
        for batch in eval_dataloader:
            guid = batch[0]
            batch = batch[1:]
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask, labels=None)

            probs = F.softmax(logits, -1).cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            nb_true = accuracy(probs, label_ids)
            nb_total_true += nb_true
            nb_eval_examples += input_ids.size(0)
            logits = logits.to("cpu").numpy()
            for id, logit in zip(guid, logits):
                f.write("{}\t{:.6f}\t{:.6f}\t{:.6f}\n".format(id, logit[0], logit[1], logit[2]))

        eval_accuracy = nb_total_true / nb_eval_examples
        logger.info("{} acc: {:.4f}".format(name,eval_accuracy))



def get_train_logit(model,processor,args,device,label_list,tokenizer,logger):
    train_examples = processor.get_examples(args.data_dir, args.train_name)
    train_features = convert_examples_to_features(args.small, train_examples, label_list, args.max_seq_length,
                                                  tokenizer, logger=logger,
                                                  cache_file=os.path.join(args.data_dir, "train.pkl"), num=640)
    save_logit(model, train_features, args, device, name="train", logger=logger)




def main():
    args = get_parser()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    logger = Logger(filename=args.output_dir + '/train.log',fmt=formatter).logger
    bert_config = BertConfig.from_json_file(args.bert_config_file)
    bert_config.num_hidden_layers = args.depth
    processors = {'dnli': DnliProcessor, 'mnli': MnliProcessor}
    processor = processors[args.dataset]()
    label_list = processor.get_labels()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    model = BertForSequenceClassification(config=bert_config, num_labels=len(label_list))
    model.to(device)
    tokenizer = BertTokenizer(args.vocab_file, args.do_lower_case)

    ckpt = torch.load(args.load_model)
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    try:
        model.load_state_dict(state_dict,strict=True)
    except Exception as e:
        logger.info(e)


    get_train_logit(model,processor,args,device,label_list,tokenizer,logger)






if __name__=="__main__":
    main()







