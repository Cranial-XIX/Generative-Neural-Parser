# Generative Neural Left Context Parser
This is the code base for the Generative Neural Left Context Parser. The model is capable of doing semi-supervised learning and the parser parses natural language sentences based on their left context features.
## Training
For supervised training:
'''
python main.py --mode spv_train --read-data yes
'''
For unsupervised training:
'''
python main.py --mode uspv_train --read-data yes
'''
## Parsing
To parse sentence from command line input:
'''
python main.py --mode parse --read-data yes
'''
## Testing
To test the parser on the 23rd folder of Wall Street Journal:
'''
python main.py --mode test --read-data yes
'''
