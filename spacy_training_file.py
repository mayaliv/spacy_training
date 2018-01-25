from __future__ import unicode_literals, print_function

import json
from pprint import pprint
import spacy
import random
from pathlib import Path
from datetime import datetime

start_time = datetime.now()

#choose a model
model='en_core_web_sm' # model package "en_core_web_sm"
#model='en_core_web_md' # model package "en_core_web_md"
#model='en_core_web_lg' # model package "en_core_web_lg"

#path for directory to save the model
output_dir='/Users/maya/PycharmProjects/spacy_training/en_example_model'

#input files
text_file="text.txt"
train_file="train_data.txt"

nlp = spacy.load(model)           # load model package

#load text file, read and extract named entities
f= open(text_file,'r')

texts=f.read().decode('utf-8').splitlines()
for text in texts:
    print('\n',text)
    doc = nlp(text)

    for ent in doc.ents:
        print(ent.text, ent.start_char, ent.end_char, ent.label_)


f.close()


# coding: utf8
"""training spaCy's named entity recognizer, starting off with an
existing model.
For more details, see the documentation:
* Training: https://spacy.io/usage/training
* NER: https://spacy.io/usage/linguistic-features#named-entities
    Compatible with: spaCy v2.0.0+
    """

#read training file from JSON format
f= open(train_file,'r')

TRAIN_DATA = json.load(open(train_file))
#pprint(TRAIN_DATA)

def main(model=None, output_dir=None, n_iter=100):
    """Load the model, set up the pipeline and train the entity recognizer."""

    nlp = spacy.load(model)  # load existing spaCy model
    print('\n',"Loaded model '%s'" % model)

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe('ner')

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update(
                    [text],  # batch of texts
                    [annotations],  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
            #print(losses)

    # test the trained model
    for text, _ in TRAIN_DATA:
        doc = nlp(text)
        print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
        print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])



    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print('\n',"Saved model to", output_dir)

        # test the saved model
        print("Test, Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        for text, _ in TRAIN_DATA:
            doc = nlp2(text)
            print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
            print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])


main(model,output_dir)
#load the saved model
nlp = spacy.load('/Users/maya/PycharmProjects/spacy_training/en_example_model')

print('\n',"Testing trained and untrained data")

#Testing both trained data and untrained data
for text in texts:
    print('\n',text)
    doc = nlp(text)

    for ent in doc.ents:
        print(ent.text, ent.start_char, ent.end_char, ent.label_)

time_elapsed = datetime.now() - start_time

print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))