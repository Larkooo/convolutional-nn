use crate::neuron::Neuron;

#[derive(Debug, Clone, Copy)]
pub struct Layer<const N: usize, const W: usize> {
    neurons: [Neuron<W>; N],
}

impl<const N: usize, const W: usize> Layer<N, W> {
    pub fn new(neurons: [Neuron<W>; N]) -> Self {
        Self { neurons }
    }

    pub fn forward(&self, inputs: &[f32]) -> [f32; N] {
        self.neurons.map(|neuron| neuron.activate(inputs))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct NeuralNetwork<const I: usize, const H: usize, const O: usize> {
    hidden_layer: Layer<H, I>,
    output_layer: Layer<O, H>,
}

impl<const I: usize, const H: usize, const O: usize> NeuralNetwork<I, H, O> {
    pub fn new() -> Self {
        let mut hidden_layer = Vec::with_capacity(H);
        for _ in 0..H {
            hidden_layer.push(Neuron::new_random());
        }

        let mut output_layer = Vec::with_capacity(O);
        for _ in 0..O {
            output_layer.push(Neuron::new_random());
        }
        Self {
            hidden_layer: Layer::new(hidden_layer.try_into().unwrap()),
            output_layer: Layer::new(output_layer.try_into().unwrap()),
        }
    }

    pub fn forward(&self, inputs: [f32; I]) -> ([f32; H], [f32; O]) {
        let hidden_layer_inputs = self.hidden_layer.forward(&inputs);
        let outputs = self.output_layer.forward(&hidden_layer_inputs);
        (hidden_layer_inputs, outputs)
    }

    pub fn backpropagate(&mut self, inputs: [f32; I], targets: [f32; O], learning_rate: f32) {
        let (hidden_outputs, outputs) = self.forward(inputs);
        
        let output_errors = outputs.iter()
            .zip(targets.iter())
            .map(|(&output, &target)| {
                let error_delta = target - output;
                error_delta * if output > 0.0 { 1.0 } else { 0.0 } // ReLU derivative
            })
            .collect::<Vec<_>>();

        let hidden_errors = hidden_outputs.iter()
            .enumerate()
            .map(|(i, &hidden_output)| {
                let error_sum: f32 = output_errors.iter()
                    .zip(self.output_layer.neurons.iter())
                    .map(|(&error, neuron)| error * neuron.weights[i])
                    .sum();
                error_sum * if hidden_output > 0.0 { 1.0 } else { 0.0 } // ReLU derivative
            })
            .collect::<Vec<_>>();

        // Correct output layer weights
        for (neuron, &error) in self.output_layer.neurons.iter_mut().zip(output_errors.iter()) {
            for (weight, &hidden_output) in neuron.weights.iter_mut().zip(hidden_outputs.iter()) {
                *weight += learning_rate * error * hidden_output;
            }
            neuron.bias += learning_rate * error;
        }
        
        // Correct hidden layer weights
        for (neuron, &error) in self.hidden_layer.neurons.iter_mut().zip(hidden_errors.iter()) {
            for (weight, &input) in neuron.weights.iter_mut().zip(inputs.iter()) {
                *weight += learning_rate * error * input;
            }
            neuron.bias += learning_rate * error;
        }
    }

    pub fn train(&mut self, dataset: &[([f32; I], [f32; O])], epochs: usize, learning_rate: f32) {
        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            for (inputs, targets) in dataset {
                let (_, outputs) = self.forward(*inputs);
                let loss: f32 = outputs.iter()
                    .zip(targets.iter())
                    .map(|(&output, &target)| (target - output).powi(2))
                    .sum();
                total_loss += loss;
                self.backpropagate(*inputs, *targets, learning_rate);
            }
            println!("Epoch {}: Average Loss = {}", epoch, total_loss / dataset.len() as f32);
        }
    }

}
