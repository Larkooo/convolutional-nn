use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

#[derive(Debug, Clone)]
pub struct Neuron {
    pub weights: Array1<f32>,
    pub bias: f32,
}

impl Neuron {
    pub fn new(weights: Array1<f32>, bias: f32) -> Self {
        Self { weights, bias }
    }

    pub fn new_random(input_size: usize) -> Self {
        let weights = Array1::random(input_size, Uniform::new(-1.0, 1.0));
        let bias = rand::random::<f32>() * 2.0 - 1.0;
        Self { weights, bias }
    }

    pub fn activate(&self, inputs: &Array1<f32>) -> f32 {
        let sum = self.weights.dot(inputs) + self.bias;
        // Leaky ReLU activation function
        if sum > 0.0 {
            sum
        } else {
            0.01 * sum
        }
    }
}

#[derive(Debug, Clone)]
pub struct Layer {
    pub weights: Array2<f32>,
    pub biases: Array1<f32>,
}

impl Layer {
    pub fn new_random(input_size: usize, output_size: usize) -> Self {
        let weights = Array2::random((output_size, input_size), Uniform::new(-1.0, 1.0));
        let biases = Array1::random(output_size, Uniform::new(-1.0, 1.0));
        Self { weights, biases }
    }

    pub fn forward(&self, inputs: &Array1<f32>) -> Array1<f32> {
        let output = self.weights.dot(inputs) + &self.biases;
        output.mapv(|x| if x > 0.0 { x } else { 0.01 * x }) // Leaky ReLU activation
    }
}
