use rand::random;

#[derive(Debug, Clone, Copy)]
pub struct Neuron<const W: usize> {
    pub weights: [f32; W],
    pub bias: f32,
}

impl<const W: usize> Neuron<W> {
    pub fn new(weights: [f32; W], bias: f32) -> Self {
        Self { weights, bias }
    }

    pub fn new_random() -> Self {
        let mut weights = [0.0; W];
        for weight in weights.iter_mut() {
            *weight = random::<f32>();
        }
        let bias = random::<f32>() - 0.5;
        Self { weights, bias }
    }

    pub fn activate(&self, inputs: &[f32; W]) -> f32 {
        let sum = inputs
            .iter()
            .zip(&self.weights)
            .map(|(input, weight)| input * weight)
            .sum::<f32>()
            + self.bias;
        // Leaky ReLU activation function
        if sum > 0.0 { sum } else { 0.01 * sum }
    }
}
