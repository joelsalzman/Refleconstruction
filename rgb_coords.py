import cv2
import matplotlib.pyplot as plt


def get_point_from_image_p(file_path):
    # Read the image using OpenCV
    img = cv2.imread(file_path)

    # Check if the image is loaded correctly
    if img is None:
        print("Failed to load image")
        return

    # Convert the image from BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the image
    plt.imshow(img_rgb)
    plt.title("Click on the Image to Select a Point")
    plt.axis("on")  # Show axes for reference

    # Use ginput to select points
    print("Please click on the image to select a point...")
    points = plt.ginput(1)  # Number of points to select
    plt.show(block=False)  # Show the plot non-blocking

    # Check if points were selected
    if points:
        print("Selected Point:", points[0])
    else:
        print("No point was selected.")

    # Close the plot automatically
    plt.close()

    return points[0] if points else None

def get_point_from_image(file_path):
    # Read the image using OpenCV
    img = cv2.imread(file_path)

    # Check if the image is loaded correctly
    if img is None:
        print("Failed to load image")
        return

    # Convert the image from BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the image
    plt.imshow(img_rgb)
    plt.title("Click on the Image to Select a Point")
    plt.axis("on")  # Show axes for reference

    # Use ginput to select points
    print("Please click on the image to select a point...")
    points = plt.ginput(1)  # Number of points to select
    plt.show(block=False)  # Show the plot non-blocking

    # Check if points were selected
    if points:
        print("Selected Point:", points[0])
    else:
        print("No point was selected.")

    # Close the plot automatically
    plt.close()

    return points[0] if points else None

# Example of how to use the function
point = get_point_from_image("./data/parrot_test_5_Color.png")
print("The coordinates of the selected point are:", point)
