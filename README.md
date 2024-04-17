# Google Stock Price Prediction using Recurrent Neural Network

## Overview

This is a project for predicting the future stock prices of Google (GOOGL) using a Recurrent Neural Network (RNN) model. The project was developed as part of the DeepLearning course on Udemy.

## Dataset

The dataset used in this project is the historical stock prices of Google (GOOGL) from 2015 to 2023. The data includes the following features:

- Date

- Open

- High

- Low

- Close

- Adjusted Close

- Volume

## Model Architecture

The model architecture used in this project is a Recurrent Neural Network (RNN) with the following layers:

1. Input Layer: The input layer takes in the historical stock prices as a sequence of data.

2. LSTM Layer: An LSTM (Long Short-Term Memory) layer is used to capture the temporal dependencies in the stock price data.

3. Dense Layer: A dense (fully connected) layer is used to output the predicted stock price.

## Training and Evaluation

The model was trained using the following hyperparameters:

- Optimizer: Adam

- Loss Function: Mean Squared Error (MSE)

- Epochs: 50

- Batch Size: 32

The model was evaluated using the following metrics:

- Mean Squared Error (MSE)

- Root Mean Squared Error (RMSE)

## Results

The final model achieved an MSE of 0.0014 (first model) and an RMSE of 13.7 (first model) on the test set, indicating that the model is able to accurately predict the future stock prices of Google.

## Usage

To run the project, follow these steps:

1. Clone the repository: git clone https://github.com/tonimurfid/RNN_Google_Stock

2. Install the required dependencies: pip install pandas tensorflow matplotlib scikit-learn

3. Run the main script: python RNN.ipynb

## Dependencies

- Python 3.7+

- NumPy

- Pandas

- TensorFlow

- Keras

- Scikit-learn

## Future Improvements

- Explore other RNN architectures, such as GRU (Gated Recurrent Unit), to potentially improve model performance.

- Incorporate additional features, such as market indexes, macroeconomic indicators, or news sentiment, to enhance the model's predictive capabilities.

- Implement a more robust hyperparameter tuning and model selection process to further optimize the model.

## Acknowledgements

- Udemy DeepLearning course

- Google Finance for providing the stock price data
