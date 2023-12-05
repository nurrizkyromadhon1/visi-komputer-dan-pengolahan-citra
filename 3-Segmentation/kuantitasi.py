from PIL import Image
def quantize_image(input_image_path, output_image_path,num_colors):
    gambar = Image.open(input_image_path)
    quantized = gambar.convert("P", palette=Image.ADAPTIVE,colors=num_colors)
    quantized = quantized.convert("RGB")
    quantized.save(output_image_path, format="JPEG")

if __name__ == "__main__":
    input_image_path = "bola2.jpg"
    output_image_path = "output.jpg"
    num_colors = 256
    quantize_image(input_image_path, output_image_path,num_colors)