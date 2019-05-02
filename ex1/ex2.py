
from basicFunc import init_centroids

from load import show_image

def main():
    kArr = [2, 4, 8, 16]
    for k in kArr:
        centroids = init_centroids(1, k)
        print("k=%d:" %k)
        show_image(centroids, k)

main()
