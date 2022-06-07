# Deep Learning with R
## Keras

# 3.2.2 - Installing Keras
install.packages("keras")
library(keras)
install_keras()

# Installing the GPU
install_keras(tensorflow = "gpu")

################################################################################
# Classifying movie reviews: a binary classification example                ----
################################################################################
library(keras)

imdb <- dataset_imdb(num_words = 10000)
c(c(train_data, train_labels), c(test_data, test_labels)) %<-% imdb

# Preparing the data
## Encoding the integer sequences into a binary matrix
vectorize_sequences <- function(sequences, dimension = 10000){
  results <- matrix(0, nrow = length(sequences), ncol = dimension)
  for (i in 1:length(sequences))
    results[i, sequences[[i]]] <- 1
  results
}

# Vectorize the sequences of words
x_train <- vectorize_sequences(train_data)
x_test <- vectorize_sequences(test_data)

# Convert labels from integer to numeric
y_train <- as.numeric(train_labels)
y_test <- as.numeric(test_labels)

# Defining the model
model <- keras_model_sequential() %>% 
  layer_dense(units = 16, activation = "relu", input_shape = c(10000)) %>% 
  layer_dense(units = 16, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

# Using custom losses and metrics
model %>% compile(
  optimizer = optimizer_rmsprop(learning_rate = 0.001),
  loss = loss_binary_crossentropy,
  metrics = metric_binary_accuracy
)

# Validation your approach
val_indices <- 1:10000

x_val <- x_train[val_indices, ]
partial_x_train <- x_train[-val_indices, ]

y_val <- y_train[val_indices]
partial_y_train <- y_train[-val_indices]


history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)

results <- model %>% evaluate(x_test, y_test)


