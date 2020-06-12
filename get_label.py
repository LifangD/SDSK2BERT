'''
This is mainly for predicting the test_matched set to submint online.
'''


from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from module import *
import os
import torch.nn.functional as F


os.environ["CUDA_VISIBLE_DEVICES"] = "3"
formatter = '%(asctime)s %(levelname)s %(message)s'



def save_label(model, features,args,device,name,label_list):
    eval_sampler = SequentialSampler(features)
    eval_dataloader = DataLoader(features, sampler=eval_sampler, collate_fn = collate_fn,batch_size=args.eval_batch_size)

    model.eval() # NOTE this is important.
    with open(os.path.join(args.output_dir,name+".tsv"),"w") as f:
        writer = csv.writer(f)
        writer.writerow(["pairID", "gold_label"])
        for batch in eval_dataloader:
            guid = batch[0]
            batch = batch[1:]
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask, labels=None)
            probs = F.softmax(logits, -1).cpu().numpy()
            for id, prob in zip(guid, probs):
                gold_id = np.argmax(prob, axis=0)
                gold_label = label_list[gold_id]
                writer.writerow([id, gold_label])


def get_test_label(model,processor,args,device,label_list,tokenizer,logger):
    test_examples = processor.get_examples(args.data_dir, args.test_name)
    test_features = convert_examples_to_features(args.small, test_examples, label_list, args.max_seq_length, tokenizer,
                                                 logger=logger, cache_file=os.path.join(args.data_dir, "test_0.9.pkl"),
                                                 num=32)
    save_label(model,test_features,args,device,"L-{}_C".format(args.depth),label_list)


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



    processor = MnliJsonProcessor()
    args.test_name="multinli_0.9_test_matched_unlabeled"
    get_test_label(model,processor,args,device,label_list,tokenizer,logger)





if __name__=="__main__":
    main()






