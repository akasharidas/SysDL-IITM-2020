import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from SysDLNet import Net

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = 5

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
        trainset, batch_size=64, shuffle=True, num_workers=4
    )

    testset = torchvision.datasets.CIFAR100(
        root="/data", train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=64, shuffle=False, num_workers=4
    )

    net = Net()
    net.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())

    print("Starting training...\n")

    for epoch in range(EPOCHS):
        pbar = tqdm(desc=f"Training epoch: {epoch}", total=len(trainloader))
        running_loss = 0.0
        correct = 0
        total = 0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            running_loss += loss.item() * labels.shape[0]
            pbar.update()

        train_loss = running_loss / len(trainset)
        train_accuracy = 100 * correct / total
        pbar.close()

        pbar = tqdm(desc="Testing", total=len(testloader))
        running_loss = 0.0
        correct = 0
        total = 0
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
        print(f"Accuracy on testset: {test_accuracy}")
        pbar.close()
