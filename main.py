from cnn_network import *
import json
from dataset_loader import *
from evaluate import evaluate_model_mAP

with open("settings.json", "r") as f:
    data = json.loads(f.read())

train_data, train_labels, ori_test_data, ori_test_labels, query_data, query_labels = load_data(data["dataset_dir"])
train_data, train_labels = generate_set(train_data, train_labels, data["train_set_length"])
test_data, test_labels = generate_set(ori_test_data, ori_test_labels, data["test_set_length"])
id_num = generate_id_num(test_labels)

model = generate_model()
history = train_model(model, train_data, train_labels, test_data, test_labels)
print_train_result(history)
evaluate_model(model, test_data, test_labels)
evaluate_model_mAP(model, query_data, query_labels, ori_test_data, ori_test_labels, id_num)
