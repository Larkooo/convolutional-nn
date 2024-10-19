use neural_network::NeuralNetwork;
use ndarray::{Array1, Array2};

mod neuron;
mod neural_network;
mod mnist;

const IMAGE_SIZE: usize = 28 * 28;
const HIDDEN_SIZE: usize = 128;
const OUTPUT_SIZE: usize = 10;

fn main() {
    // Load MNIST data
    let mnist_data = mnist::load_mnist("dataset/train-images.idx3-ubyte", "dataset/train-labels.idx1-ubyte").unwrap();
    
    let mut nn = NeuralNetwork::new(IMAGE_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    
    // Prepare training data
    let mut training_data = Vec::new();
    for (image, label) in mnist_data.images.iter().zip(mnist_data.labels.iter()) {
        let input = Array1::from_iter(image.iter().map(|&x| x as f32 / 255.0));
        let mut target = Array1::zeros(OUTPUT_SIZE);
        target[*label as usize] = 1.0;
        training_data.push((input, target));
    }

    // Train the network
    nn.train(&training_data, 100, 0.01); // 100 epochs, learning rate 0.01
    nn.train(training_data.as_slice(), 100, 0.01); // 10 epochs, learning rate 0.01

    // Test the network on a few samples
    for i in 0..5 {
        let (input, target) = &training_data[i];
        let (_, output) = nn.forward(input);
        let predicted_digit = output.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        let actual_digit = target.iter().position(|&x| x == 1.0).unwrap();
        println!("Sample {}: Predicted {}, Actual {}", i, predicted_digit, actual_digit);
    }
}
