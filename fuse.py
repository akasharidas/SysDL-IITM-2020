import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import wandb
import time
import os

from SysDLNet import Net

if __name__ == "__main__":
    wandb.login(key="df416cf0e6b9361efc64aa08d4715af979c8d070")
    config = dict(
        epochs=5,
        batch_size=64,
        optimizer="adam",
        lr=0.001,
        momentum=None,
        weight_decay=0,
        schedule=False,
    )

    wandb.init(config=config, project="SysDL Assignment 3")

    device = "cuda" if torch.cuda.is_available() else "cpu"

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
            optimizer, len(trainloader) * config["epochs"], verbose=True
        )

    print("Starting training...\n")

    wandb.watch(net)

    for epoch in range(config["epochs"]):
        wandb.log({"epoch": epoch}, commit=False)

        start_time = time.time()
        pbar = tqdm(desc=f"Training epoch: {epoch}", total=len(trainloader))
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

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            wandb.log({"train_loss": loss.item()})
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
        print(f"Accuracy on testset: {test_accuracy}\n")
        pbar.close()

        wandb.log(
            {
                "train_acc": train_accuracy,
                "test_loss": test_loss,
                "test_acc": test_accuracy,
                "epoch_time": time.time() - start_time,
            },
        )

        if (epoch + 1) % 5 == 0:
            torch.save(
                net.state_dict(), os.path.join(wandb.run.dir, f"SysDLNet_{epoch}.pt")
            )
