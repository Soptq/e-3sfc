import random

import torch
from torchvision import datasets, transforms
import numpy as np

from models import get_model, count_parameters
from helper import ExperimentLogger, display_train_stats
from fl_devices import Server, Client
from data_utils import get_dataset

from config import get_config

import wandb

args = get_config()
device = args.device if torch.cuda.is_available() else "cpu"
print("Using device: %s" % device)

if "," in device:
    device = [d.strip() for d in device.split(",")]
    print("Using multiple devices: %s" % str(device))

wandb.init(
    project="fl-data-distillation",
    entity="soptq",
    config={k: v for k, v in vars(args).items()}
)

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True

client_data, test_data, n_classes = get_dataset(args)

clients = [Client(
    lambda: get_model(args, n_classes),
    lambda x: torch.optim.SGD(x, lr=args.lr, momentum=0.0),
    dat,
    i,
    args,
    batch_size=args.batch_size,
    device=device if type(device) is str else device[i]
)
    for i, dat in enumerate(client_data)]
server = Server(lambda: get_model(args, n_classes), test_data, args, device=device if type(device) is str else device[0])

print("Number of model parameters: ", count_parameters(server.model))

cfl_stats = ExperimentLogger()

acc_clients = []
acc_servers = []
cos_dict = {}
best_inputs_server, best_labels_server, scale_factor_server, cos_server = None, None, None, None

for c_round in range(1, args.n_epoch + 1):
    participating_clients = server.select_clients(clients, frac=args.client_fraction)
    training_loss, cos = [], []
    report = {}

    if args.method == "fedavg":
        for client in clients:
            client.synchronize_with_server(server)

        for client in participating_clients:
            train_stats = client.compute_weight_update(epochs=args.n_client_epoch)
            training_loss.append(train_stats)
            cos.append(1.0)
            client.reset()
        server.aggregate(participating_clients)

    elif args.method == "sign":
        for client in clients:
            client.synchronize_with_server(server)

        for client in participating_clients:
            train_stats = client.compute_weight_update(epochs=args.n_client_epoch)
            training_loss.append(train_stats)
            client.reset()
        server.aggregate_sign_compression(participating_clients)

    elif args.method == "stc":
        for client in clients:
            client.synchronize_with_server(server)

        for client in participating_clients:
            train_stats = client.compute_weight_update(epochs=args.n_client_epoch)
            training_loss.append(train_stats)
            client.reset()
        server.aggregate_stc_compression(participating_clients, args.topk)

    elif args.method == "topk":
        for client in clients:
            client.synchronize_with_server(server)

        topk_dws = []
        for client in participating_clients:
            train_stats = client.compute_weight_update(epochs=args.n_client_epoch)
            training_loss.append(train_stats)
            client.reset()

            topk_dw, topk_cos = client.compute_topk(args.topk)
            topk_dws.append(topk_dw)
            cos.append(abs(topk_cos.item()))
        server.aggregate_fusion(topk_dws)

    elif args.method == "fedsynth":
        for client in clients:
            client.synchronize_with_server(server)

        synthetics, scale_factors = [], []
        for client in participating_clients:
            train_stats = client.compute_weight_update(epochs=args.n_client_epoch)
            training_loss.append(train_stats)
            client.reset()

            inputs, labels, scale, _cos = client.compute_fedsynth(args.ours_n_sample, n_classes, args.lr, args.lr, epochs=args.n_client_epoch)
            scale_factors.append(scale)
            synthetics.append((inputs, labels))

            cos.append(abs(_cos.item()))
        server.aggregate_synthetic_gradients(synthetics, scale_factors, [client.dW for client in participating_clients])

    elif args.method == "ours":
        for client in clients:
            client.synchronize_with_server(server)

        synthetics, scale_factors = [], []
        for client in participating_clients:
            train_stats = client.compute_weight_update(epochs=args.n_client_epoch)
            training_loss.append(train_stats)
            client.reset()

            ours_n_sample = args.ours_n_sample
            s = args.s
            print("Using samples: ", ours_n_sample, " iteration: ", s)
            best_inputs, best_labels, scale_factor, _cos = client.compute_synthetic_sample(ours_n_sample, n_classes, s=s, lambd=args.lambd)
            scale_factors.append(scale_factor)
            synthetics.append((best_inputs, best_labels))

            cos.append(abs(_cos.item()))
        server.aggregate_synthetic_gradients(synthetics, scale_factors, [client.dW for client in participating_clients])

    elif args.method == "ours-dwc":
        for client in clients:
            if best_inputs_server is not None:
                client.synchronize_with_synthetic_samples_from_server(best_inputs_server, best_labels_server, scale_factor_server)
            else:
                client.synchronize_with_server(server)

        synthetics, scale_factors = [], []
        ours_n_sample = args.ours_n_sample

        for client in participating_clients:
            train_stats = client.compute_weight_update(epochs=args.n_client_epoch)
            training_loss.append(train_stats)
            client.reset()
            
            best_inputs, best_labels, scale_factor, _cos = client.compute_synthetic_sample(ours_n_sample, n_classes, s=args.s, lambd=args.lambd)
            scale_factors.append(scale_factor)
            synthetics.append((best_inputs, best_labels))

            cos.append(abs(_cos.item()))
        
        best_inputs_server, best_labels_server, scale_factor_server, cos_server = server.aggregate_synthetic_gradients_and_compute_samples(
            synthetics, scale_factors, [client.dW for client in participating_clients], ours_n_sample, n_classes, s=args.s)

    cos_mean = np.mean(cos)
    cos_std = np.std(cos)

    training_loss_mean = np.mean(training_loss)
    training_loss_std = np.std(training_loss)

    acc_clients = [client.evaluate() for client in clients]
    acc_clients_mean = np.mean(acc_clients)
    acc_clients_std = np.std(acc_clients)

    acc_servers = [server.evaluate()]
    acc_servers_mean = np.mean(acc_servers)
    acc_servers_std = np.std(acc_servers)

    report["cos_lowest"] = cos_mean - cos_std
    report["cos_highest"] = cos_mean + cos_std
    report["training_loss_lowest"] = training_loss_mean - training_loss_std
    report["training_loss_highest"] = training_loss_mean + training_loss_std
    report["acc_clients_lowest"] = acc_clients_mean - acc_clients_std
    report["acc_clients_highest"] = acc_clients_mean + acc_clients_std
    report["acc_servers_lowest"] = acc_servers_mean - acc_servers_std
    report["acc_servers_highest"] = acc_servers_mean + acc_servers_std

    wandb.log(report)

    print(f"Round {c_round}, Clients Acc: {acc_clients}, Server Acc: {acc_servers}")
    # cfl_stats.log({"acc_clients": acc_clients, "acc_servers": acc_servers, "rounds": c_round})
    # if c_round % 10 == 0:
    #     display_train_stats(cfl_stats, 100)
 