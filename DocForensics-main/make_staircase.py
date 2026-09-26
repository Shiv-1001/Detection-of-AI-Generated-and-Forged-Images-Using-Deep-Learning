from PIL import Image

src = r"C:\Users\SHIV\Downloads\img.jpeg"
img = Image.open(src).convert("RGB")
native_max = max(img.size)
print(f"native max dimension: {native_max}px")

sizes_real = [28, 50, 75, 100, 150, 200, 300, 450, 600, 900, 1200]
sizes_upsampled = [1600, 2000, 2500, 3000]

for size in sizes_real:
    if size > native_max:
        continue
    w, h = img.size
    new_w, new_h = (size, int(size * h / w)) if w >= h else (int(size * w / h), size)
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    resized.save(f"test_{size:04d}_real.jpg", quality=92)
    print(f"saved test_{size:04d}_real.jpg  ({new_w}x{new_h})")

img.save(f"test_{native_max:04d}_native.jpg", quality=95)
print(f"saved test_{native_max:04d}_native.jpg  ({img.size[0]}x{img.size[1]})")

for size in sizes_upsampled:
    w, h = img.size
    new_w, new_h = (size, int(size * h / w)) if w >= h else (int(size * w / h), size)
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    resized.save(f"test_{size:04d}_upsampled.jpg", quality=92)
    print(f"saved test_{size:04d}_upsampled.jpg  ({new_w}x{new_h})")
