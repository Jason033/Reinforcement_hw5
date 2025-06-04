import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from mlp_mixer_pytorch import MLPMixer

if __name__ == "__main__":
    # 1. 設定設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. 設定超參數
    EPOCHS = 50
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3

    # 3. 準備 CIFAR-10 數據集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # CIFAR-10 圖像標準化
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # 4. 初始化 MLP-Mixer 模型
    # 針對 CIFAR-10 (32x32 圖像, 10 類別) 調整參數
    model = MLPMixer(
        image_size = 32,
        channels = 3,
        patch_size = 4,       # 32x32 圖像, 4x4 patch -> (32/4)^2 = 64 patches
        dim = 128,            # 較小的維度以加速訓練
        depth = 4,            # 較淺的層數
        num_classes = 10      # CIFAR-10 有 10 個類別
    ).to(device)

    # 5. 定義損失函數和優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 6. 訓練模型
    print("\nStarting training...")
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 100 == 0: # 每 100 個 mini-batches 印一次 log
                print(f'Epoch [{epoch + 1}/{EPOCHS}], Step [{i + 1}/{len(trainloader)}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0
    print("Finished Training.")

    # 7. 評估模型
    print("\nStarting evaluation...")
    correct = 0
    total = 0
    # 由於評估時不需要計算梯度，可以關閉梯度計算以節省記憶體和加速
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the 10000 test images: {accuracy:.2f} %')

    print("\nMLP-Mixer validation on CIFAR-10 finished.")
