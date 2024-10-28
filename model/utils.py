
from torch import nn
from transformers import AutoModel, AutoTokenizer

pretrained_model_dic = {
        "robert":"google-bert/roberta-base",
        "xlnet":"google-bert/xlnet-base-cased",
        "bert":"google-bert/bert-base-chinese",
        }


class TextClassificationModel(nn.Module):
    def __init__(self, pretrained_model_name):
        super(TextClassificationModel, self).__init__()
        print(pretrained_model_name)
        # 加载bert模型
        self.bert = AutoModel.from_pretrained(pretrained_model_dic[pretrained_model_name])

        # 最后的预测层
        self.predictor = nn.Sequential(
            nn.Linear(768, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, src):
        """
        :param src: 分词后的推文数据
        """

        # 将src直接序列解包传入bert，因为bert和tokenizer是一套的，所以可以这么做。
        # 得到encoder的输出，用最前面[CLS]的输出作为最终线性层的输入
        outputs = self.bert(**src).last_hidden_state[:, 0, :]

        # 使用线性层来做最终的预测
        return self.predictor(outputs)


def get_tokenizer(pretrained_model_name):
    return AutoTokenizer.from_pretrained(pretrained_model_dic[pretrained_model_name])

def get_model(pretrained_model_name):
    return AutoModel.from_pretrained(pretrained_model_dic[pretrained_model_name])