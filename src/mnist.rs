use std::fs::File;
use std::io::{self, Read};

pub struct MnistData {
    pub images: Vec<Vec<f32>>,
    pub labels: Vec<u8>,
}

pub fn load_mnist(images_file: &str, labels_file: &str) -> io::Result<MnistData> {
    let mut image_file = File::open(images_file)?;
    let mut label_file = File::open(labels_file)?;

    // Read image file header
    let mut buffer = [0u8; 16];
    image_file.read_exact(&mut buffer)?;
    let num_images = u32::from_be_bytes([buffer[4], buffer[5], buffer[6], buffer[7]]) as usize;
    let num_rows = u32::from_be_bytes([buffer[8], buffer[9], buffer[10], buffer[11]]) as usize;
    let num_cols = u32::from_be_bytes([buffer[12], buffer[13], buffer[14], buffer[15]]) as usize;

    // Read label file header
    let mut buffer = [0u8; 8];
    label_file.read_exact(&mut buffer)?;
    let num_labels = u32::from_be_bytes([buffer[4], buffer[5], buffer[6], buffer[7]]) as usize;

    assert_eq!(num_images, num_labels, "Number of images and labels must match");

    // Read image data
    let mut image_data = vec![0u8; num_images * num_rows * num_cols];
    image_file.read_exact(&mut image_data)?;

    // Read label data
    let mut label_data = vec![0u8; num_labels];
    label_file.read_exact(&mut label_data)?;

    // Convert image data to f32 and normalize pixel values
    let images: Vec<Vec<f32>> = image_data
        .chunks(num_rows * num_cols)
        .map(|chunk| chunk.iter().map(|&x| x as f32 / 255.0).collect())
        .collect();
    

    Ok(MnistData { images, labels: label_data })
}
