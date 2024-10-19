use ndarray::{Array1, Array2, Axis};
use crate::neuron::Layer;

#[derive(Debug, Clone)]
pub struct NeuralNetwork {
    hidden_layer: Layer,
    output_layer: Layer,
}

impl NeuralNetwork {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        Self {
            hidden_layer: Layer::new_random(input_size, hidden_size),
            output_layer: Layer::new_random(hidden_size, output_size),
        }
    }

    pub fn forward(&self, inputs: &Array1<f32>) -> (Array1<f32>, Array1<f32>) {
        let hidden_outputs = self.hidden_layer.forward(inputs);
        let outputs = self.output_layer.forward(&hidden_outputs);
        (hidden_outputs, outputs)
    }

    pub fn backpropagate(&mut self, inputs: &Array1<f32>, targets: &Array1<f32>, learning_rate: f32) {
        let (hidden_outputs, outputs) = self.forward(inputs);
        
        // Calculate output layer errors
        let output_errors = &outputs - targets;
        let output_gradients = &output_errors * &outputs.mapv(|x| if x > 0.0 { 1.0 } else { 0.01 });

        // Calculate hidden layer errors
        let hidden_errors = self.output_layer.weights.t().dot(&output_gradients);
        let hidden_gradients = &hidden_errors * &hidden_outputs.mapv(|x| if x > 0.0 { 1.0 } else { 0.01 });

        // Update output layer weights and biases
        let output_weight_updates = output_gradients.clone().into_shape((output_gradients.len(), 1)).unwrap()
            .dot(&hidden_outputs.clone().into_shape((1, hidden_outputs.len())).unwrap());
        self.output_layer.weights -= &(learning_rate * output_weight_updates);
        self.output_layer.biases -= &(learning_rate * &output_gradients);

        // Update hidden layer weights and biases
        let hidden_weight_updates = hidden_gradients.clone().into_shape((hidden_gradients.len(), 1)).unwrap()
            .dot(&inputs.clone().into_shape((1, inputs.len())).unwrap());
        self.hidden_layer.weights -= &(learning_rate * hidden_weight_updates);
        self.hidden_layer.biases -= &(learning_rate * &hidden_gradients);
    }

    pub fn train(&mut self, dataset: &[(Array1<f32>, Array1<f32>)], epochs: usize, learning_rate: f32) {
        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            for (inputs, targets) in dataset {
                let (_, outputs) = self.forward(inputs);
                let loss: f32 = outputs.iter()
                    .zip(targets.iter())
                    .map(|(&output, &target)| (target - output).powi(2))
                    .sum();
                total_loss += loss;
                self.backpropagate(inputs, targets, learning_rate);
            }
            println!("Epoch {}: Average Loss = {}", epoch, total_loss / dataset.len() as f32);
        }
    }
}
