sample_size = (1024, 1024)
h, w = sample_size
bound = 256
H = 13312
W = 6400
for y in range(-bound, H-bound, h):
    for x in range(-bound, W-bound, w):
        print(y, x)