import os
import random
import torch
import json
import pandas as pd
from utils import itemIndex, load_plm


def generate_item_embedding(word_drop_ratio, emb_type, device, output_path, item2index, item_text_list, plm_tokenizer,
                            plm_model):
    items, texts = zip(*item_text_list)
    order_texts = [[0]] * len(items)
    for item, text in zip(items, texts):
        order_texts[item2index[item]] = text
    for text in order_texts:
        assert text != [0]
    embeddings = []
    start, batch_size = 0, 4
    while start < len(order_texts):
        sentences = order_texts[start: start + batch_size]
        if word_drop_ratio > 0:
            new_sentences = []
            for sent in sentences:
                new_sent = []
                sent = sent.split(' ')
                for wd in sent:
                    rd = random.random()
                    if rd > word_drop_ratio:
                        new_sent.append(wd)
                new_sent = ' '.join(new_sent)
                new_sentences.append(new_sent)
            sentences = new_sentences
        encoded_sentences = plm_tokenizer(sentences, padding=True, max_length=512,
                                          truncation=True, return_tensors='pt').to(device)
        outputs = plm_model(**encoded_sentences)
        if emb_type == 'CLS':
            cls_output = outputs.last_hidden_state[:, 0, ].detach().cpu()
            embeddings.append(cls_output)
        elif emb_type == 'Mean':
            masked_output = outputs.last_hidden_state * encoded_sentences['attention_mask'].unsqueeze(-1)
            mean_output = masked_output[:, 1:, :].sum(dim=1) / \
                          encoded_sentences['attention_mask'][:, 1:].sum(dim=-1, keepdim=True)
            mean_output = mean_output.detach().cpu()
            embeddings.append(mean_output)
        start += batch_size
    embeddings = torch.cat(embeddings, dim=0).numpy()
    print('Embeddings shape: ', embeddings.shape, '\n')

    # suffix=1, output DATASET.feat1CLS, with word drop ratio 0;
    # suffix=2, output DATASET.feat2CLS, with word drop ratio > 0;
    if word_drop_ratio > 0:
        suffix = '2'
    else:
        suffix = '1'

    file = os.path.join(output_path, 'item' + '.feat' + suffix + emb_type)
    embeddings.tofile(file)
    print("item embedding completed")


if __name__ == '__main__':
    csv_file_path = './dataset/raw/item.csv'
    output_path = './dataset/downstream/'

    # 构建商品和商品编号的对应关系
    itemIndex(csv_file_path, output_path)

    # 使用bert对商品中午描述进行embedding
    word_drop_ratio = -1
    emb_type = 'CLS'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(output_path, 'item2index.json'), 'r', encoding='utf-8') as f:
        item2index = json.load(f)

    item_text_list = pd.read_csv(csv_file_path, dtype=str)[['prod_id', 'prod_description']].drop_duplicates(
        subset=['prod_id'], keep='first').values.tolist()

    plm_path = './dataset/m3e-base/'
    plm_tokenizer, plm_model = load_plm(plm_path)
    plm_model = plm_model.to(device)

    generate_item_embedding(word_drop_ratio, emb_type, device, output_path, item2index, item_text_list, plm_tokenizer,
                            plm_model)
