# encoding: utf-8


from transformers import BertConfig, XLMRobertaConfig, RobertaConfig


class BertQueryNerConfig(BertConfig):
    def __init__(self, **kwargs):
        print(f"kwargs: {kwargs}")
        super(BertQueryNerConfig, self).__init__(**kwargs)
        self.mrc_dropout = kwargs.get("mrc_dropout", 0.1)
        self.bert_config_dir = kwargs.get("pretrained_model_name_or_path")


class XLMRobertaQueryNerConfig(XLMRobertaConfig):
    def __init__(self, **kwargs):
        super(XLMRobertaQueryNerConfig, self).__init__(**kwargs)
        self.mrc_dropout = kwargs.get("mrc_dropout", 0.1)
        self.bert_config_dir = kwargs.get("pretrained_model_name_or_path")


class RobertaConfigQueryNerConfig(RobertaConfig):
    def __init__(self, **kwargs):
        super(RobertaConfigQueryNerConfig, self).__init__(**kwargs)
        self.mrc_dropout = kwargs.get("mrc_dropout", 0.1)
        self.bert_config_dir = kwargs.get("pretrained_model_name_or_path")


def get_config(
    bert_config_dir,
    hidden_dropout_prob,
    attention_probs_dropout_prob,
    mrc_dropout,
):

    if "bert" in bert_config_dir:
        print(f"Bert config found: {bert_config_dir}.")
        return BertQueryNerConfig.from_pretrained(
            pretrained_model_name_or_path=bert_config_dir,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            mrc_dropout=mrc_dropout,
        )

    elif "xlmr" in bert_config_dir:
        print(f"XLMRoberta config found: {bert_config_dir}.")
        return XLMRobertaQueryNerConfig.from_pretrained(
            pretrained_model_name_or_path=bert_config_dir,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            mrc_dropout=mrc_dropout,
        )

    elif "roberta" in bert_config_dir:
        print(f"Roberta config found: {bert_config_dir}.")
        return RobertaConfigQueryNerConfig.from_pretrained(
            pretrained_model_name_or_path=bert_config_dir,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            mrc_dropout=mrc_dropout,
        )

    else:
        raise NotImplementedError(
            f"Model {bert_config_dir} not supported. Supported model: [bert,roberta,xlmroberta]"
        )
