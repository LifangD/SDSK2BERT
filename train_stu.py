from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import time
import torch.nn.functional as F
from module import *
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
formatter = '%(asctime)s %(levelname)s %(message)s'

def loss_fn_kd(outputs, labels, teacher_outputs, T,alpha):
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T)+F.cross_entropy(outputs, labels) * (1. - alpha)
    return KD_loss

def eval(model, eval_features,args,device,n_gpu):
    eval_sampler = SequentialSampler(eval_features)
    eval_dataloader = DataLoader(eval_features, sampler=eval_sampler, collate_fn = collate_fn,batch_size=args.eval_batch_size)
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for batch in eval_dataloader:
        batch = batch[1:]
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        with torch.no_grad():
            loss,logits= model(input_ids, segment_ids, input_mask, label_ids)
        if n_gpu > 1:
            eval_loss += loss.mean().item()
        else:
            eval_loss += loss.item()
        probs = F.softmax(logits,-1).cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        tmp_eval_accuracy = accuracy(probs, label_ids)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss/nb_eval_steps
    eval_accuracy = eval_accuracy/nb_eval_examples
    return eval_accuracy, eval_loss





def main():
    args = get_parser()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    logger = Logger(filename=args.output_dir + '/train.log',fmt=formatter).logger
    bert_config = BertConfig.from_json_file(args.bert_config_file)
    processors = {'dnli': DnliProcessor, 'mnli': MnliProcessor}
    processor = processors[args.dataset]()
    label_list = processor.get_labels()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    bert_config.num_hidden_layers = args.depth # try the depth
    model = BertForSequenceClassification(config=bert_config, num_labels=len(label_list))
    model.to(device)
    tokenizer = BertTokenizer(args.vocab_file, args.do_lower_case)
    # two options to initialize the model
    ckpt = torch.load(args.load_model)
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt



    old_keys = []
    new_keys = []
    for key in state_dict.keys():
        new_key = None
        if 'gamma' in key:
            new_key = key.replace('gamma', 'weight')
        if 'beta' in key:
            new_key = key.replace('beta', 'bias')
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)
    try:
        model.load_state_dict(state_dict,strict=True)
    except Exception as e:
        logger.info(e)

    train_examples = processor.get_examples(args.data_dir, args.train_name)
    train_logits = processor.get_teacher_logit(args.data_dir, args.train_name)
    dev_examples = processor.get_examples(args.data_dir, args.dev_name)

    train_features = convert_examples_logits_to_features(args.small, train_examples, train_logits,label_list, args.max_seq_length,tokenizer, logger=logger,cache_file=os.path.join(args.data_dir,"train_lgt.pkl"),num=640)
    dev_features = convert_examples_to_features(args.small, dev_examples, label_list, args.max_seq_length,tokenizer, logger=logger,cache_file=os.path.join(args.data_dir,"dev.pkl"),num=32)




    num_train_steps = math.ceil(len(train_features)/args.train_batch_size)*args.num_train_epochs
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm.bias', 'LayerNorm.weight']
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if (not any(nd in n for nd in no_decay))],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}]

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_steps)

    n_gpu = torch.cuda.device_count()
    if n_gpu > 1: model = torch.nn.DataParallel(model)
    logger.info("begin training,stored in {}...".format(args.output_dir))


    train_sampler = RandomSampler(train_features)
    train_dataloader = DataLoader(train_features, sampler=train_sampler, collate_fn=t_collate_fn,batch_size=args.train_batch_size)
    global_step, best_acc, eval_score_history = 0, 0, [0, 0]
    train_start_time = time.time()
    for epochs in range(int(args.num_train_epochs)):
        for step, batch in enumerate(train_dataloader):
            batch = batch[1:]
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids,t_logits = batch
            logits = model(input_ids, segment_ids, input_mask, None)
            loss = loss_fn_kd(logits,label_ids,t_logits,args.tau,args.alpha)

            if n_gpu>1:
                loss = loss.mean()
            loss.backward()

            if step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                model.zero_grad()

            if (global_step) % args.log_step == 0:
                label_ids = label_ids.to('cpu').numpy()
                t_probs = F.softmax(logits, -1).detach().cpu().numpy()
                acc = accuracy(t_probs, label_ids) / len(label_ids)
                logger.info("step {}/{}, acc: {:.4f}".format(global_step, num_train_steps,acc))
            global_step += 1


        logger.info("========begin evaluating========")
        model.eval()
        eval_acc, eval_loss = eval(model, dev_features, args, device,n_gpu)
        model.train()
        logger.info("step {}/{}, eval_acc:{:.4f}".format(global_step, num_train_steps, eval_acc))
        if eval_acc > best_acc:
            best_acc = eval_acc
            model_name = os.path.join(args.output_dir, "best_model.pt")
            logger.info("new best saved!")
            if n_gpu > 1:
                torch.save({'global_step': global_step, 'state_dict': model.module.state_dict()},open(model_name, 'wb'))
            else:
                torch.save({'global_step': global_step, 'state_dict': model.state_dict()}, open(model_name, 'wb'))


    tt = sec_to_h(time.time() - train_start_time)
    logger.info("train time: {} h, {} m, {} s".format(tt[0], tt[1], tt[2]))

if __name__=="__main__":
    main()






