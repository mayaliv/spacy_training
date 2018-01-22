from __future__ import unicode_literals, print_function

import plac
import spacy
import random
from pathlib import Path


TEXTS = [
    'Who is Shaka Khan?',
    'I like London and Berlin.',
]

nlp = spacy.load('en_core_web_sm')           # load model package "en_core_web_sm"

doc1 = nlp(u'Apple is looking at buying U.K. startup for $1 billion')

for ent in doc1.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)

doc2 = nlp(u'Who is Shaka Khan?')

for ent in doc2.ents:
     print(ent.text, ent.start_char, ent.end_char, ent.label_)

doc3 = nlp(u'I like London and Berlin')

for ent in doc3.ents:
     print(ent.text, ent.start_char, ent.end_char, ent.label_)


# !/usr/bin/env python
# coding: utf8
"""Example of training spaCy's named entity recognizer, starting off with an
existing model or a blank model.
For more details, see the documentation:
* Training: https://spacy.io/usage/training
* NER: https://spacy.io/usage/linguistic-features#named-entities
    Compatible with: spaCy v2.0.0+
    """

# a change

# training data
TRAIN_DATA = [
    ('Who is Shaka Khan?', {
        'entities': [(7, 17, 'PERSON')]
    }),
    ('I like London and Berlin.', {
        'entities': [(7, 13, 'LOC'), (18, 24, 'LOC')]
    })
]


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int))
def main2(model=None, output_dir=None, n_iter=100):
    """Load the model, set up the pipeline and train the entity recognizer."""
    if True: #model is not None:
        nlp = spacy.load('en_core_web_sm')  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")

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
            print(losses)

    # test the trained model
    for text, _ in TRAIN_DATA:
        doc = nlp(text)
        print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
        print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])

    output_dir='/Users/maya/PycharmProjects/spacy_training/en_example_model'

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        for text, _ in TRAIN_DATA:
            doc = nlp2(text)
            print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
            print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])


if __name__ == '__main__':
    plac.call(main2)







nlp = spacy.load('/Users/maya/PycharmProjects/spacy_training/en_example_model')

doc4 = nlp(u'Who is Shaka Khan?')

for ent in doc4.ents:
     print(ent.text, ent.start_char, ent.end_char, ent.label_)

doc5 = nlp(u'I like London and Berlin')

for ent in doc5.ents:
     print(ent.text, ent.start_char, ent.end_char, ent.label_)


doc6 = nlp(u'Apple is looking at buying U.K. startup for $1 billion')

for ent in doc6.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)
