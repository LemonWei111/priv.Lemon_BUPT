# 加载训练好的模型对测试文件进行标注
import torch
from load_data import read_test_data, get_idx_v2, load_dict

best_model = './bc_model_0.9951.pth'

#### TEST ####
test_file_path_data = 'test.txt'
tag_dict_path = './tag_dict_v2.txt'

output_file_path = '2021213513.txt'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load(best_model)
model.to(device)
print(f"model ok to {device}")

### TODO ###
test_data, test_mask = read_test_data(test_file_path_data)
# print(test_data[0], type(test_data))
# print(test_mask[0], type(test_mask))

# 将字或词转换为索引
# data_word2idx, _ = load_dict('./data_dict_v2.txt') # no_pretrain
test_data_idx = get_idx_v2(test_data) # , data_word2idx) # pretrain
# print(test_data_idx[0], type(test_data_idx))
del test_data # , data_word2idx
# input()

tag_word2idx, _  = load_dict(tag_dict_path)
tag_idx2word = {idx:word for word, idx in tag_word2idx.items()}
del tag_word2idx
# print(tag_idx2word)
# print("testing")

# 测试模型
model.eval()  # 设置模型为评估模式
with torch.no_grad():
    predicted_labels = []
    m = 128
    for input_ids, attention_mask in zip(test_data_idx, test_mask):
        # print(len(input_ids))
        # print(input_ids)
        if len(input_ids) >= m:
            start, end = 0, m
            predicted_label = []
            while start < len(input_ids):
                # print(start, end)
                if end > len(input_ids):
                    n_input_ids = input_ids[start:]
                    n_attention_mask = attention_mask[start:]
                else:
                    n_input_ids = input_ids[start:end]
                    n_attention_mask = attention_mask[start:end]
                start += m
                end += m
                outputs = model(torch.tensor([n_input_ids]).to(device), torch.tensor([n_attention_mask]).to(device))
                predicted = outputs.logits if isinstance(outputs, dict) else outputs
                # print(predicted, type(predicted))

                predicted_label.extend([tag_idx2word[pd]for p in predicted for pd in p])
                # print(predicted_label)

            # assert len(input_ids) == len(predicted_label), "TAG ERROR"

            predicted_labels.append(predicted_label)
            # input()
            continue

        outputs = model(torch.tensor([input_ids]).to(device), torch.tensor([attention_mask]).to(device))
        predicted = outputs.logits if isinstance(outputs, dict) else outputs
        # print(predicted, type(predicted))

        predicted_label = [tag_idx2word[pd]for p in predicted for pd in p]
        # print(predicted_label)

        predicted_labels.append(predicted_label)
        # input()

    # 输出预测结果
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for predicted_label in predicted_labels:
            f.write(' '.join(str(label) for label in predicted_label) + '\n')

    print(f"Predictions saved to {output_file_path}")
