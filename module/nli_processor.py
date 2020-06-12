# encoding = utf-8
import csv
import os
import jsonlines
import sys
import torch


class InputExample(object):
    def __init__(self, guid, feature_a, feature_b, label=None):
        self.guid = guid
        self.feature_a = feature_a
        self.feature_b = feature_b
        self.label = label

class DistillInputFeatures(object):
    def __init__(self, guid, input_ids, input_mask, segment_ids, label_id,t_logits):
        self.guid = guid
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.t_logits = t_logits


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, guid, input_ids, input_mask, segment_ids, label_id):
        self.guid = guid
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines




class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_examples(self, data_dir, name):
        """See base class."""

        return self._create_examples(self._read_tsv(os.path.join(data_dir, name + ".tsv")))

    def get_labels(self):
        """See base class."""
        return ["entailment", "neutral", "contradiction"]

    def _create_examples(self, lines):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = line[2]
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            if label not in self.get_labels():
                label = "neutral"  # default
            examples.append(
                InputExample(guid=guid, feature_a=text_a, feature_b=text_b, label=label))
        return examples

    def get_teacher_logit(self,data_dir,name):
        return self._read_tsv(os.path.join(data_dir,name+"_tea_logits.tsv"))



class DnliProcessor(DataProcessor):  #
    """Processor for the Persona data set."""

    def get_examples(self, data_path, name):
        """See base class."""
        lines = []
        data = jsonlines.Reader(open(os.path.join(data_path, name + ".jsonl"), 'r')).read()
        for item in data:
            lines.append(item)
        return self._create_examples(lines)

    def get_labels(self):

        return ["negative", "neutral", "positive"]

    def _create_examples(self, lines):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = i
            feature_a = line["sentence1"]
            feature_b = line['sentence2']
            label = line['label']
            examples.append(InputExample(guid=guid, feature_a=feature_a, feature_b=feature_b, label=label))
        return examples

    def get_teacher_logit(self, data_dir, name):
        return self._read_tsv(os.path.join(data_dir, name + "_tea_logits.tsv"))

class MnliJsonProcessor(DataProcessor):  #
    """Processor for the Persona data set."""

    def get_examples(self, data_path,name):
        """See base class."""
        lines = []
        with jsonlines.open(os.path.join(data_path, name + ".jsonl")) as f:
            for item in f:
                lines.append(item)
        return self._create_examples(lines)

    def get_labels(self):

        return ["entailment", "neutral", "contradiction"]

    def _create_examples(self, lines):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = line['pairID']
            feature_a = line["sentence1"]
            feature_b = line["sentence2"]
            if "label" not in line:
                label = "neutral" # this is only for test
            else:
                label = line["label"]
            examples.append(InputExample(guid=guid, feature_a=feature_a, feature_b=feature_b, label=label))
        return examples


def convert_examples_to_features(small, examples, label_list, max_seq_length, tokenizer, logger, cache_file, num=10000):
    """Loads a data file into a list of `InputBatch`s."""
    if os.path.exists(cache_file) and not small:
        logger.info("loading from {}".format(cache_file))
        return torch.load(cache_file)
    else:
        label_map = {}

        for (i, label) in enumerate(label_list):
            label_map[label] = i
        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 1000 == 0:
                sys.stdout.write("\rprocessing  {} samples...".format(ex_index))
                sys.stdout.flush()
            guid = example.guid
            tokens_a = tokenizer.tokenize(example.feature_a)
            tokens_b = tokenizer.tokenize(example.feature_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            if example.label == "-":
                pass  # some examples can't be used.
            else:
                label_id = label_map[example.label]  # mapping the label to label_id
                features.append(
                    InputFeatures(guid=guid,
                                  input_ids=input_ids,
                                  input_mask=input_mask,
                                  segment_ids=segment_ids,
                                  label_id=label_id,
                                  ))
                if logger is not None:
                    if ex_index < 1:
                        logger.info("*** Example ***")
                        logger.info("guid: %s" % (example.guid))
                        logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
                        logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                        logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                        logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                        logger.info("label: %s " % (example.label))

            # small=True
            if small and ex_index > num:  # for quickly testing
                break
        if not small:
            torch.save(features, cache_file)
        return features
def convert_examples_logits_to_features(small, examples, logits, label_list, max_seq_length, tokenizer, logger, cache_file, num=10000):
    """Loads a data file into a list of `InputBatch`s."""
    logger.info("input examples and logits from teachers.")
    if os.path.exists(cache_file) and not small:
        logger.info("loading from {}".format(cache_file))
        return torch.load(cache_file)
    else:
        label_map = {}

        for (i, label) in enumerate(label_list):
            label_map[label] = i
        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 1000 == 0:
                sys.stdout.write("\rprocessing  {} samples...".format(ex_index))
                sys.stdout.flush()
            guid = example.guid
            cur_logits = logits[ex_index]

            #assert example.guid==int(cur_logits[0])
            tokens_a = tokenizer.tokenize(example.feature_a)
            tokens_b = tokenizer.tokenize(example.feature_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            if example.label == "-":
                pass  # some examples can't be used.
            else:
                label_id = label_map[example.label]  # mapping the label to label_id
                features.append(
                    DistillInputFeatures(guid=guid,
                                  input_ids=input_ids,
                                  input_mask=input_mask,
                                  segment_ids=segment_ids,
                                  label_id=label_id,
                                  t_logits = [float(x) for x in cur_logits[1:]]
                                  ))

                if ex_index < 1:
                    logger.info("*** Example ***")
                    logger.info("guid: %s" % (example.guid))
                    logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
                    logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                    logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                    logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                    logger.info("label: %s " % example.label)
                    logger.info("t_logits:%s"% " ".join(cur_logits[1:]))


            # small=True
            if small and ex_index > num:  # for quick debug
                break
        if not small:
            torch.save(features, cache_file)
        return features



def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def collate_fn(train_features):
    all_guids = [f.guid for f in train_features]
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    return all_guids, all_input_ids, all_input_mask, all_segment_ids, all_label_ids

def t_collate_fn(train_features):
    all_guids = [f.guid for f in train_features]
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    all_t_logits = torch.tensor([f.t_logits for f in train_features],dtype=torch.float)
    return all_guids, all_input_ids, all_input_mask, all_segment_ids, all_label_ids,all_t_logits