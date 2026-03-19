from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_transforms(image_size: int = 224):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    return train_transform, eval_transform


def get_pet_binary_datasets(
    data_dir: str = "data/raw",
    image_size: int = 224,
    train_ratio: float = 0.8,
    random_seed: int = 42,
):
    data_root = Path(data_dir)
    data_root.mkdir(parents=True, exist_ok=True)

    train_transform, eval_transform = get_transforms(image_size=image_size)

    full_dataset_for_train = datasets.OxfordIIITPet(
        root=str(data_root),
        split="trainval",
        target_types="binary-category",
        transform=train_transform,
        download=True,
    )

    full_dataset_for_eval = datasets.OxfordIIITPet(
        root=str(data_root),
        split="trainval",
        target_types="binary-category",
        transform=eval_transform,
        download=False,
    )

    total_size = len(full_dataset_for_train)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size

    generator = torch.Generator().manual_seed(random_seed)

    train_indices, val_indices = random_split(
        range(total_size),
        [train_size, val_size],
        generator=generator
    )

    train_dataset = torch.utils.data.Subset(full_dataset_for_train, train_indices.indices)
    val_dataset = torch.utils.data.Subset(full_dataset_for_eval, val_indices.indices)

    test_dataset = datasets.OxfordIIITPet(
        root=str(data_root),
        split="test",
        target_types="binary-category",
        transform=eval_transform,
        download=False,
    )

    return train_dataset, val_dataset, test_dataset


def create_dataloaders(
    data_dir: str = "data/raw",
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 0,
):
    train_dataset, val_dataset, test_dataset = get_pet_binary_datasets(
        data_dir=data_dir,
        image_size=image_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    class_names = ["cat", "dog"]

    return train_loader, val_loader, test_loader, class_names


if __name__ == "__main__":
    train_loader, val_loader, test_loader, class_names = create_dataloaders()

    print(f"class names: {class_names}")
    print(f"train batches: {len(train_loader)}")
    print(f"val batches: {len(val_loader)}")
    print(f"test batches: {len(test_loader)}")

    images, labels = next(iter(train_loader))
    print(f"image batch shape: {images.shape}")
    print(f"label batch shape: {labels.shape}")
    print(f"sample labels: {labels[:10]}")