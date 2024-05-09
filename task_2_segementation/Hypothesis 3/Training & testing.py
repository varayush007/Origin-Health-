# Initializing variables for early stopping
num_epochs_aug = 10
best_val_loss = float('inf')
patience = 3  # Number of epochs to wait if validation loss stops improving
counter = 0  # Counter to keep track of epochs without improvement

# Training the segmentation model with augmented dataset and early stopping
for epoch in range(num_epochs_aug):
    segmentation_model_h3.train()
    running_loss_aug = 0.0

    for batch_idx, batch_aug in enumerate(train_loader_aug):
        images_aug, masks_aug = batch_aug['image'], batch_aug['mask']

        optimizer_seg_h3.zero_grad()

        outputs_aug = segmentation_model_h3(images_aug)
        masks_aug_resized = nn.functional.interpolate(masks_aug, size=outputs_aug.shape[2:], mode='bilinear', align_corners=True)

        loss_aug = criterion_seg(outputs_aug, masks_aug_resized)
        loss_aug.backward()
        optimizer_seg_h3.step()

        running_loss_aug += loss_aug.item() * images_aug.size(0)

    epoch_loss_aug = running_loss_aug / len(train_dataset_aug)
    print(f'Augmented Segmentation Epoch [{epoch + 1}/{num_epochs_aug}], Training Loss: {epoch_loss_aug:.4f}')

    # Validation loss calculation
    segmentation_model_h3.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_idx, batch_val in enumerate(val_loader_aug):
            images_val, masks_val = batch_val['image'], batch_val['mask']
            outputs_val = segmentation_model_h3(images_val)
            outputs_val_resized = nn.functional.interpolate(outputs_val, size=(256, 256), mode='bilinear', align_corners=False)
            loss_val = criterion_seg(outputs_val_resized, masks_val)
            val_loss += loss_val.item() * images_val.size(0)
    val_loss /= len(val_dataset_aug)
    print(f'Validation Loss: {val_loss:.4f}')

    # Checking for improvement in validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
    else:
        counter += 1

    # Checking if early stopping conditions are met
    if counter >= patience:
        print("Early stopping triggered! Validation loss has not improved for {} epochs.".format(patience))
        break