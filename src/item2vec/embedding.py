import argparse
import os
import random

from item2vec.io import load_item_index, load_plm, write_item_indexes


def build_item_index(item_csv, output_dir):
    import pandas as pd

    write_item_indexes(pd.read_csv(item_csv)['prod_id'], output_dir)
    return load_item_index(output_dir)


def generate_item_embedding(word_drop_ratio, emb_type, device, output_path, item2index, item_text_list,
                            plm_tokenizer, plm_model):
    import torch

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
                new_sentences.append(' '.join(new_sent))
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
            embeddings.append(mean_output.detach().cpu())
        start += batch_size
    embeddings = torch.cat(embeddings, dim=0).numpy()
    print('Embeddings shape: ', embeddings.shape, '\n')

    suffix = '2' if word_drop_ratio > 0 else '1'
    file = os.path.join(output_path, 'item' + '.feat' + suffix + emb_type)
    embeddings.tofile(file)
    print("item embedding completed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('item_csv')
    parser.add_argument('output_dir')
    parser.add_argument('model_dir')
    args = parser.parse_args()

    import pandas as pd
    import torch

    item2index = build_item_index(args.item_csv, args.output_dir)
    word_drop_ratio = -1
    emb_type = 'CLS'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    item_text_list = pd.read_csv(args.item_csv, dtype=str)[['prod_id', 'prod_description']].drop_duplicates(
        subset=['prod_id'], keep='first').values.tolist()
    plm_tokenizer, plm_model = load_plm(args.model_dir)
    plm_model = plm_model.to(device)
    generate_item_embedding(word_drop_ratio, emb_type, device, args.output_dir, item2index, item_text_list,
                            plm_tokenizer, plm_model)


if __name__ == '__main__':
    main()
