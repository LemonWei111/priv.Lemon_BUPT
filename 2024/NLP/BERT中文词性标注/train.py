# 模型训练
from load_data import *
from BERT import *
from utils import *
# from tqdm.notebook import tqdm
# from torch.utils.tensorboard import SummaryWriter

###################    load_data    #################

train_file_path_data = 'train.txt'
train_file_path_tags = 'train_TAG.txt'

dev_file_path_data = 'dev.txt'
dev_file_path_tags = 'dev_TAG.txt'

# test_file_path_data = 'test.txt'

max_length = 128

train_data, train_mask = read_data_v2(train_file_path_data, max_length)
train_tags = read_tag_data_v2(train_file_path_tags, max_length)

# data_word2idx, _ = load_dict('./data_dict_v2.txt') # if no-pretrain
tag_word2idx, num_tags = load_dict('./tag_dict_v2.txt')

# 将字或词转换为索引
train_data_idx = get_idx_v2(train_data) # , data_word2idx)
train_tag_idx = get_idx(train_tags, tag_word2idx)
del train_data, train_tags

dev_data, dev_mask = read_data_v2(dev_file_path_data, max_length)
dev_tags = read_tag_data_v2(dev_file_path_tags, max_length)
dev_data_idx = get_idx_v2(dev_data) # , data_word2idx)
dev_tag_idx = get_idx(dev_tags, tag_word2idx)
del dev_data, dev_tags

del tag_word2idx # ,data_word2idx

# 创建DataLoader
batch_size = 8  # 批大小

train_data_loader = DataLoader(dataset=CustomDataset_v2(train_data_idx, train_tag_idx, train_mask), batch_size=batch_size, num_workers = 0, drop_last = True, shuffle=True)
dev_data_loader = DataLoader(dataset=CustomDataset_v2(dev_data_idx, dev_tag_idx, dev_mask), batch_size=batch_size, num_workers = 0, drop_last = True, shuffle=False)

'''
# save dict
train_data, train_tags = read_data_and_tags(train_file_path_data, train_file_path_tags)
data_word2idx, _, _ = build_vocab(train_data)
tag_word2idx, _, num_tags = build_vocab(train_tags)

train_data_idx = get_idx(train_data, data_word2idx)
train_tag_idx = get_idx(train_tags, tag_word2idx)
del train_data, train_tags
print("ok")

dev_data, dev_tags = read_data_and_tags(dev_file_path_data, dev_file_path_tags)
dev_data_idx = get_idx(dev_data, data_word2idx)
dev_tag_idx = get_idx(dev_tags, tag_word2idx)
del dev_data, dev_tags
print("ok")

write_dict2file("data_dict.txt", data_word2idx)
del data_word2idx
write_dict2file("tag_dict.txt", tag_word2idx)
del tag_word2idx
'''
'''
# load dict
train_data, train_tags = read_data_and_tags(train_file_path_data, train_file_path_tags)
data_word2idx = load_dict("data_dict.txt")
tag_word2idx = load_dict("tag_dict.txt")

train_data_idx = get_idx(train_data, data_word2idx)
train_tag_idx = get_idx(train_tags, tag_word2idx)
del train_data, train_tags
print("ok")

dev_data, dev_tags = read_data_and_tags(dev_file_path_data, dev_file_path_tags)
dev_data_idx = get_idx(dev_data, data_word2idx)
dev_tag_idx = get_idx(dev_tags, tag_word2idx)
print("ok")
num_tags = len(tag_word2idx)
del dev_data, dev_tags, data_word2idx, tag_word2idx

batch_size = 1  # 批大小
train_data_loader = DataLoader(dataset=CustomDataset(train_data_idx, train_tag_idx), batch_size=batch_size, num_workers = 3, drop_last = True, shuffle=True)
dev_data_loader = DataLoader(dataset=CustomDataset(dev_data_idx, dev_tag_idx), batch_size=batch_size, num_workers = 3, drop_last = True, shuffle=False)
'''
# print(len(dev_data_idx),len(dev_tags))
# print(len(dev_data_idx[0]),len(dev_tags[0]))
# print(len(dev_data_idx[1]),len(dev_tags[1]))

###################    load_BERT    #################

# 初始化模型
# print(num_tags, tag_word2idx)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BERT_NER(num_tags, drop = 0.2)

model.to(device)
print(f"model ok to {device}")

lr = 1e-6
optimizer = torch.optim.AdamW(model.parameters(), lr=lr) # , weight_decay=1e-6)

# l_train = len(train_data_idx)
# data_pad = data_word2idx['<O>']
# tag_pad = tag_word2idx['<O>']

###################    train&test    #################

# 训练模型
num_epochs = 10 # 训练的轮数

log_loss = []
log_acc = []
best_acc = 0
bad_force = 0
# pb = tqdm(range(len(train_data_loader)))
# witer = SummaryWriter(log_dir=f'./runs/{lr}')

for epoch in range(num_epochs):
    model.train() # 设置模型为训练模式
    for input_ids, input_labels, attention_mask in train_data_loader:
        input_ids, input_labels, attention_mask = input_ids.to(device), input_labels.to(device), attention_mask.to(device)
        optimizer.zero_grad()
        # attention_mask = torch.ones_like(input_ids)
        # outputs = model(torch.tensor(input_ids), attention_mask, torch.tensor(input_labels))
        outputs = model(input_ids, attention_mask, input_labels)

        out_loss = outputs.loss if isinstance(outputs, dict) else outputs
        loss = torch.mean(out_loss)
        
        # print(loss)
        loss.backward()
        # print('ok')
        optimizer.step() # capable for memory problem
        # print('ok')

        # pb.update(1)

    if epoch%1==0:
        print('evaling')
        model.eval() # 设置模型为评估模式
        with torch.no_grad():
            correct_predictions = 0
            total_predictions = 0
            for input_ids, input_labels, attention_mask in dev_data_loader:
                input_ids, input_labels, attention_mask = input_ids.to(device), input_labels.to(device), attention_mask.to(device)
                # attention_mask = torch.ones_like(input_ids)
                # outputs = model(torch.tensor(input_ids), attention_mask)
                outputs = model(input_ids, attention_mask)

                predicted_labels = outputs.logits if isinstance(outputs, dict) else outputs
                # print(predicted_labels)

                correct_predictions += correct_num(predicted_labels, input_labels)
                total_predictions += input_labels.size(1) * input_labels.size(0)
        accuracy = correct_predictions / total_predictions # ) * 100
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}, Dev Accuracy: {accuracy:.4f}")

        log_loss.append(loss.cpu())
        log_acc.append(accuracy)
        # writer.add_scalar('Loss/train', loss, epoch)
        # writer.add_scalar('Acc/dev', accuracy, epoch)
        
        # torch.save(model, f"bc_model_last_epoch{epoch}_{accuracy:.4f}.pth")
        
        if accuracy > best_acc:
            torch.save(model, f"bc_model_{accuracy:.4f}.pth")
            best_acc = accuracy
            bad_force = 0
        else:
            bad_force += 1

        # early stop
        if bad_force >= 2:
            break

show(log_loss, 'Loss')
show(log_acc, 'Acc')

# writer.close()
# print("begin test")
del train_data_loader, train_data_idx, train_tag_idx, train_mask, dev_data_loader, dev_data_idx, dev_tag_idx, dev_mask

'''
#### TEST ####
test_file_path_data = 'test.txt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load('./bc_model_0.9897.pth') # BERT_NER(num_tags, drop = 0.2)

model.to(device)
print(f"model ok to {device}")

### TODO ###
test_data, test_mask = read_test_data(test_file_path_data)

data_word2idx, _ = load_dict('./data_dict_v2.txt')

# 将字或词转换为索引
test_data_idx = get_idx_v2(test_data) # , data_word2idx)
del test_data, data_word2idx

tag_word2idx, _  = load_dict('./tag_dict_v2.txt')
tag_idx2word = {idx:word for word, idx in tag_word2idx.items()}
del tag_word2idx
print(tag_idx2word)
print("testing")
# 测试模型
model.eval()  # 设置模型为评估模式
with torch.no_grad():
    predicted_labels = []
    m = 128
    for input_ids, attention_mask in zip(test_data_idx, test_mask):
        # print(len(input_ids))
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
            predicted_labels.append(predicted_label)
            continue

        outputs = model(torch.tensor([input_ids]).to(device), torch.tensor([attention_mask]).to(device))
        predicted = outputs.logits if isinstance(outputs, dict) else outputs
        # print(predicted, type(predicted))
        predicted_label = [tag_idx2word[pd]for p in predicted for pd in p]
        # print(predicted_label)
        predicted_labels.append(predicted_label)
    # 输出预测结果
    output_file_path = '2021213513.txt'
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for predicted_label in predicted_labels:
            f.write(' '.join(str(label) for label in predicted_label) + '\n')

    print(f"Predictions saved to {output_file_path}")
'''
