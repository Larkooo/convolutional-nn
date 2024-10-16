use neural_network::NeuralNetwork;

mod neuron;
mod neural_network;
mod mnist;

const IMAGE_SIZE: usize = 28 * 28;

fn main() {
    // Load MNIST data
    let mnist_data = mnist::load_mnist("dataset/train-images.idx3-ubyte", "dataset/train-labels.idx1-ubyte").unwrap();
    
    let mut nn = NeuralNetwork::<IMAGE_SIZE, 128, 10>::new();
    
    // Prepare training data
    let mut training_data = Vec::new();
    for (image, label) in mnist_data.images.iter().zip(mnist_data.labels.iter()) {
        let input: [f32; IMAGE_SIZE] = image.iter().map(|&x| x as f32 / 255.0).collect::<Vec<_>>().try_into().unwrap();
        let mut target = [0.0; 10];
        target[*label as usize] = 1.0;
        training_data.push((input, target));
    }

    // Train the network
    nn.train(training_data.as_slice(), 10, 0.01); // 10 epochs, learning rate 0.01

    // Test the network on a few samples
    for i in 0..5 {
        let (input, target) = &training_data[i];
        let (_, output) = nn.forward(*input);
        let predicted_digit = output.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        let actual_digit = target.iter().position(|&x| x == 1.0).unwrap();
        println!("Sample {}: Predicted {}, Actual {}", i, predicted_digit, actual_digit);
    }
}
