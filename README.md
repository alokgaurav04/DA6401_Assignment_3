# DA6401_Assignment_3
Wandb link : https://wandb.ai/alokgaurav04-indian-institute-of-technology-madras/DA6401_Assignment_3/reports/Assignment-3-DA6401--VmlldzoxMjM5MDgyOQ

Github Link : https://github.com/alokgaurav04/DA6401_Assignment_3

This notebook is structured in such a way that all the cells can be run one after another. Run All Cells command can also be used, but be careful of WandB sweeps.

Run these two lines to install hindi font for ploting heatmap:-

## Installing font for Hindi for matplotlib ##

!apt-get install -y fonts-lohit-deva

!fc-list :lang=hi family

To run the model without WandB, use the following code:

model = test_on_dataset(language="hi",
                        embedding_dim=256,
                        encoder_layers=3,
                        decoder_layers=3,
                        layer_type="lstm",
                        units=256,
                        dropout=0.2,
                        attention=False)
                        
To run the model with WandB sweep, use the following code:

# Creating the WandB config

sweep_config = {

  "method": "grid",
  
  "metric":{
  
      'name':'val acc',
      
      'goal':'maximize'
      
  },
  
  "parameters": {
  
        "enc_dec_layers": {
        
           "values": [1, 2, 3]                 # Number of Encoder and Decoder layer
           
        },
        
        "units": {
        
            "values": [64, 128, 256]           # Dimensionality
            
        },
        
        "layer_type": {
        
            "values": ["rnn", "gru", "lstm"]   # Cell Type
            
        },
        
        "embedding_dim": {
        
            "values": [64, 128, 256]           #Embedding size
            
        },
        
        "dropout": {
        
            "values": [0.2, 0.3]               #Dropout
            
        }
        
    }
    
}

# Creating a sweep
sweep_id = wandb.sweep(sweep_config, project="DA6401_Assignment_3")

# Running the sweep
wandb.agent(sweep_id, function=lambda: train_with_wandb("hi"))

# Sample some words from the test data
test_words = get_test_words(5)

# Visualise connectivity for "test_words"
for word in test_words:

    visualise_connectivity(model, word, activation="scaler")

# For plotting heatmap 
model.plot_attention_heatmap(Word, ax)
