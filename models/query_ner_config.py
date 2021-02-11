# encoding: utf-8


from transformers import BertConfig, XLMRobertaConfig

class BertQueryNerConfig(BertConfig):
    def __init__(self, **kwargs):
        super(BertQueryNerConfig, self).__init__(**kwargs)
        self.mrc_dropout = kwargs.get("mrc_dropout", 0.1)


class XLMRobertaQueryNerConfig(XLMRobertaConfig):
    def __init__(self, **kwargs):
        super(XLMRobertaQueryNerConfig, self).__init__(**kwargs)
        self.mrc_dropout = kwargs.get("mrc_dropout", 0.1)
