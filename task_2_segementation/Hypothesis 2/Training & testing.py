# Training the segmentation model with augmented dataset and early stopping
num_epochs_aug = 10
early_stopping_patience = 3
best_val_loss = float('inf')
epochs_no_improve = 0

for epoch in range(num_epochs_aug):
    segmentation_model_h2.train()
    running_loss_aug = 0.0

    # Training loop
    for batch_idx, batch_aug in enumerate(train_loader_aug):
        images_aug, masks_aug = batch_aug['image'], batch_aug['mask']

        optimizer_seg_h2.zero_grad()

        outputs_aug = segmentation_model_h2(images_aug)
        masks_aug_resized = nn.functional.interpolate(masks_aug, size=outputs_aug.shape[2:], mode='bilinear', align_corners=True)

        loss_aug = criterion_seg(outputs_aug, masks_aug_resized)
        loss_aug.backward()
        optimizer_seg_h2.step()

        running_loss_aug += loss_aug.item() * images_aug.size(0)

    epoch_loss_aug = running_loss_aug / len(train_dataset_aug)
    print(f'Augmented Segmentation Epoch [{epoch + 1}/{num_epochs_aug}], Training Loss: {epoch_loss_aug:.4f}')

    # Validation loop
    segmentation_model_h2.eval()
    val_loss_aug = 0.0

    with torch.no_grad():
        for batch_idx, batch_val in enumerate(val_loader_aug):
            images_val, masks_val = batch_val['image'], batch_val['mask']
            outputs_val = segmentation_model_h2(images_val)
            masks_val_resized = nn.functional.interpolate(masks_val, size=outputs_val.shape[2:], mode='bilinear', align_corners=True)
            val_loss_aug += criterion_seg(outputs_val, masks_val_resized).item() * images_val.size(0)

    val_loss_aug /= len(val_dataset_aug)
    print(f'Augmented Segmentation Validation Loss: {val_loss_aug:.4f}')

    # Check for early stopping
    if val_loss_aug < best_val_loss:
        best_val_loss = val_loss_aug
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= early_stopping_patience:
            print(f'Early stopping after epoch {epoch + 1}')
            break
