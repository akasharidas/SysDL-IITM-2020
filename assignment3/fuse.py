import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import wandb
import time
import os

from SysDLNet import Net


def run(config=None):
    if config:
        wandb.init(config=config, project="SysDL Assignment 3")
    else:
        wandb.init(project="SysDL Assignment 3")
    config = wandb.config
    device = "cuda"

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = torchvision.datasets.CIFAR100(
        root="/data", train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=config["batch_size"], shuffle=True, num_workers=4
    )

    testset = torchvision.datasets.CIFAR100(
        root="/data", train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=config["batch_size"], shuffle=False, num_workers=4
    )

    net = Net()
    net.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()

    if config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(
            net.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
        )
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            momentum=config["momentum"],
        )

    if config["schedule"]:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, len(trainloader) * config["epochs"]
        )

    print("Starting training...\n")
    num_examples = 0

    wandb.watch(net)

    for epoch in range(config["epochs"]):
        start_time = time.time()
        pbar = tqdm(desc="Training epoch: {}".format(epoch), total=len(trainloader))
        correct = 0
        total = 0

        net.train()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            if config["schedule"]:
                scheduler.step()

            num_examples += labels.shape[0]
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            wandb.log({"train_loss": loss.item(), "examples_seen": num_examples})
            pbar.update()

        train_accuracy = 100 * correct / total
        pbar.close()

        pbar = tqdm(desc="Testing", total=len(testloader))
        running_loss = 0.0
        correct = 0
        total = 0

        net.eval()
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                running_loss += loss_fn(outputs, labels).item() * labels.shape[0]
                pbar.update()

        test_loss = running_loss / len(testset)
        test_accuracy = 100 * correct / total
        print("Accuracy on testset: {}\n".format(test_accuracy))
        pbar.close()

        wandb.log(
            {
                "epoch": epoch,
                "train_acc": train_accuracy,
                "test_loss": test_loss,
                "test_acc": test_accuracy,
                "epoch_time": time.time() - start_time,
            },
        )

        if (epoch + 1) % 5 == 0:
            torch.save(
                net.state_dict(),
                os.path.join(wandb.run.dir, "SysDLNet_{}.pt".format(epoch)),
            )


if __name__ == "__main__":
    wandb.login(key="df416cf0e6b9361efc64aa08d4715af979c8d070")

    run(
        {
            "epochs": 150,
            "optimizer": "adam",
            "lr": 0.0025,
            "weight_decay": 0.00001,
            "batch_size": 512,
            "momentum": 0.95,
            "schedule": True,
        }
    )

    # sweep_config = {"method": "random"}
    # metric = {"name": "loss", "goal": "minimize"}
    # sweep_config["metric"] = metric
    # parameters_dict = {
    #     "epochs": {"value": 20},
    #     "optimizer": {"values": ["adam", "sgd"]},
    #     "lr": {"values": [0.0001, 0.0025, 0.0075, 0.001, 0.003]},
    #     "weight_decay": {"values": [0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]},
    #     "batch_size": {"values": [64, 256, 512]},
    #     "momentum": {"values": [0, 0.1, 0.3, 0.6, 0.8, 0.9, 0.95]},
    #     "schedule": {"values": [True, False]},
    # }
    # sweep_config["parameters"] = parameters_dict

    # sweep_id = wandb.sweep(sweep_config, project="SysDL Assignment 3")
    # wandb.agent(sweep_id, run, count=25)
