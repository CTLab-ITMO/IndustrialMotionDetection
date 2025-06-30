import numpy as np
import matplotlib.pyplot as plt


def show_mask(mask, ax, random_color=False, borders=True):
    """Displays a mask on the given matplotlib axes with optional borders and random coloring.

    :param mask: A 2D numpy array representing the mask to display
    :param ax: Matplotlib axes object where the mask will be displayed
    :param random_color: If True, uses a random color for the mask
    :param borders: If True, draws contours around the mask boundaries
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    """Displays positive and negative points on the given matplotlib axes.

    :param coords: Nx2 array of point coordinates
    :param labels: N array of point labels
    :param ax: Matplotlib axes object where points will be displayed
    :param marker_size: Size of the point markers.
    """
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    """Displays a bounding box on the given matplotlib axes.

    :param box: Array of [x0, y0, x1, y1] coordinates representing the bounding box
    :param ax: Matplotlib axes object where the box will be displayed
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    """Displays segmentation masks with optional points and bounding boxes.

    :param image: Original image to display
    :param masks: Array of segmentation masks to display
    :param scores: Array of scores for each mask
    :param point_coords: Optional Nx2 array of point coordinates
    :param box_coords: Optional array of [x0,y0,x1,y1] bounding box coordinates
    :param input_labels: Optional array of point labels (required if point_coords is provided)
    :param borders: If True, draws contours around mask boundaries
    """
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        if point_coords is not None:
            show_points(point_coords, input_labels, plt.gca())
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
        plt.imshow(image)


def visualize_results(image: np.ndarray,
                      masks: np.ndarray,
                      alpha: float = 0.5) -> None:
    """Visualizes the results of segmentation.

    :param image: Source image
    :param masks: Segmentation masks
    :param alpha: Overlay transparency
    """

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(masks[0], cmap='gray')
    plt.title("Segmentation Mask")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(image)
    plt.imshow(masks[0], alpha=alpha, cmap='jet')
    plt.title("Overlay")
    plt.axis('off')

    plt.tight_layout()
    plt.show()
