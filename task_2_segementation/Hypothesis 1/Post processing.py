import cv2
import numpy as np

# Define function for post-processing
def post_process_segmentation(output_masks):
    biometry_points = []

    for mask in output_masks:
        # Convert mask to numpy array
        mask_np = mask.squeeze().cpu().numpy()

        # Find contours in the mask
        contours, _ = cv2.findContours((mask_np * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Fit ellipse to the largest contour
        if contours:
            contour = max(contours, key=cv2.contourArea)
            ellipse = cv2.fitEllipse(contour)

            # Extract biometry points (center and top)
            center = ellipse[0]
            top_point = (center[0], center[1] - ellipse[1][1] / 2)
            biometry_points.append((center, top_point))
        else:
            # If no contour found, add None values
            biometry_points.append((None, None))

    return biometry_points

# Example usage:
output_masks = [torch.rand(1, 128, 128) > 0.5 for _ in range(8)]  # Example output masks
biometry_points = post_process_segmentation(output_masks)
print(biometry_points)
