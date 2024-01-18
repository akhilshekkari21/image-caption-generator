import tensorflow
print(&#39;tensorflow: %s&#39; % tensorflow. version )
# keras version
import keras
print(&#39;keras: %s&#39; % keras. version )
from os import listdir
from pickle import dump
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image
import img_to_array from keras.applications.vgg16
import preprocess_input from keras.models import Model
Preparing Text Data
# load doc into memory
def load_doc(filename):
# open the file as read only
file = open(filename, &#39;r&#39;)
# read all text
text = file.read()
# close the file

12

file.close() return text
filename = &#39;captions.txt&#39;
# load descriptions
doc = load_doc(filename)
Below defines a function load_descriptions() that, given the loaded document text, will
return a dictionary of photo identifiers to descriptions. Each photo identifier maps to a
list of one or more textual descriptions

# extract descriptions for images
def load_descriptions(doc): mapping = dict()
# process lines
for line in doc.split(&#39;\n&#39;):
# split line by white space
tokens = line.split()
if len(line) &lt; 2:
continue
# take the first token as the image id, the rest as the description
image_id, image_desc = tokens[0], tokens[1:]
# remove filename from image id
image_id = image_id.split(&#39;.&#39;)[0]
# convert description tokens back to string
image_desc = &#39; &#39;.join(image_desc)
# create the list if needed
if image_id not in mapping:
mapping[image_id] = list()
# store description
mapping[image_id].append(image_desc)
return mapping

13

parse descriptions

descriptions = load_descriptions(doc) print(&#39;Loaded: %d &#39; % len(descriptions)
Now we have to clean all the descriptions

--&gt; Convert all words to lowercase. --&gt; Remove all punctuation. --&gt; Remove all words
that are one character or less in length (e.g. ‘a’). --&gt; Remove all words with numbers
in them
import string

def clean_descriptions(descriptions):
# prepare translation table for removing punctuation
table = str.maketrans(&#39;&#39;, &#39;&#39;, string.punctuation)
for key, desc_list in descriptions.items():
for i in range(len(desc_list)): desc = desc_list[i]
# tokenize
desc = desc.split()
# convert to lower case
desc = [word.lower() for word in desc]
# remove punctuation from each token
desc = [w.translate(table) for w in desc]
# remove hanging &#39;s&#39; and &#39;a&#39;
desc = [word for word in desc if len(word)&gt;1]
# remove tokens with numbers in them
desc = [word for word in desc if word.isalpha()]
# store as string
desc_list[i] = &#39; &#39;.join(desc)

14

# clean descriptions
clean_descriptions(descriptions)
## Here you can see cleaned up version of
list(descriptions.items())[:2]

we have to clean descriptions !!
# convert the loaded descriptions into a vocabulary of words
def to_vocabulary(descriptions):
# build a list of all description strings
all_desc = set()
for key in descriptions.keys():
[all_desc.update(d.split()) for d in descriptions[key]]
return all_desc

# summarize vocabulary
vocabulary = to_vocabulary(descriptions) print(&#39;Vocabulary Size: %d&#39; % len(vocabulary))
print(vocabulary)

saving descriptions
# save descriptions to file, one per line
Def
save_descriptions
(descriptions filename):
lines = list()
for key, desc_list in descriptions.items():
for desc in desc_list:
lines.append(key + &#39; &#39; +desc)

15

data=&#39;\n&#39;.join(lines)
file=open(filena
me, &#39;w&#39;)
file.write(data)
file.close()
# save descriptions
save_descriptions(descriptions, &#39;descriptions.txt&#39;)

Developing Image Captioning Model

Loading data
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
fromkeras.utils import to_categorical
from keras.utils import plot_model
from keras.modelsimport Model
from keras.layersimport Input
from keras.layersimport Dense
from keras.layersimport LSTM
from keras.layersimport Embedding
from keras.layersimport Dropout
fromkeras.layers.merge import add
from keras.callbacks import ModelCheckpoint
from array import array
# load doc into memory
def load_doc(filename):
# open the file as read only
file = open(filename, &#39;r&#39;)
# read all text

16

text = file.read()
#close the file file.close() return text
# load a pre-defined list of photo identifiers
def load_set(filename): doc = load_doc(filename) dataset = list()
# process line by line
for line in doc.split(&#39;\n&#39;):
# skip empty lines if len(line) &lt; 1: continue
# get the image identifier
identifier = line.split(&#39;.&#39;)[0]
dataset.append(identifier)
return set(dataset)

# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
# load document
doc = load_doc(filename) descriptions = dict()
for line in doc.split(&#39;\n&#39;):
# split line by white space
tokens = line.split()
# split id from description
image_id, image_desc = tokens[0], tokens[1:]
# skip images not in the set
if image_id in
dataset:
# create list
if image_id not in descriptions:
descriptions[image_id] = list()
# wrap description in tokens
desc = &#39;startseq &#39; + &#39; &#39;.join(image_desc) + &#39; endseq&#39;

17

# store
descriptions[image_id].append(desc)
return descriptions

# load photo features
def load_photo_features(filename, dataset):
# load all features
all_features = load(open(filename, &#39;rb&#39;))
# filter features
features = {k: all_features[k] for k in dataset}
return features

# covert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
all_desc = list()
for key in descriptions.keys():
[all_desc.append(d) for d in descriptions[key]]
return all_desc
# fit a tokenizer given caption descriptions
Def
create_tokenizer(descrptions): lines = to_lines(descriptions)
tokenizer = Tokenizer() tokenizer.fit_on_texts(lines)
return tokenizer
# calculate the length of the description with the most words
def
max_length(descriptions):
lines =
to_lines(descriptions)
return max(len(d.split()) for d in lines)

18

# create sequences of images, input sequences and output words for an
image def create_sequences(tokenizer, max_length, desc_list, photo,
vocab_size): X1, X2, y = list(), list(), list()
# walk through each description for the image
for desc in desc_list:
# encode the sequence
seq = tokenizer.texts_to_sequences([desc])[0]
# split one sequence into multiple X,y pairs
for i in range(1, len(seq)):
# split into input and output pair
in_seq, out_seq = seq[:i], seq[i]
# pad input sequence
in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
# encode output sequence
out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
#store
X1.append(photo) X2.append(in_seq) y.append(out_seq)
return array(X1), array(X2), array(y)
# define the captioning model
Def define_model(vocab_size, max_length):
# feature extractor model inputs1 = Input(shape=(4096,))
fe1 = Dropout(0.5)(inputs1)
# load training dataset (6K)
filename = &#39;train.txt&#39; train = load_set(filename)
print(&#39;Dataset: %d&#39; % len(train))
# descriptions

19

train_descriptions = load_clean_descriptions(&#39;descriptions.txt&#39;, train)
print(&#39;Descriptions: train=%d&#39; % len(train_descriptions))
# photo features
train_features = load_photo_features(&#39;features.pkl&#39;, train)
print(&#39;Photos: train=%d&#39; % len(train_features))
# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1 print(&#39;Vocabulary Size: %d&#39; % vocab_size)
# determine the maximum
sequence length
max_length =
max_length(train_descriptions)
print(&#39;Description Length: %d&#39;
% max_length)

# define the model
model = define_model(vocab_size, max_length)
# train the model, run epochs manually and save after each epoch
epochs = 20
steps = len(train_descriptions)
for i in range(epochs):
# create the data generator
generator = data_generator(train_descriptions, train_features, tokenizer,
max_length, vocab_size)
# fit for one epoch
model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
# save model
model.save(&#39;model_&#39; + str(i) + &#39;.h5&#39;)
from numpy import argmax
from pickle import load

20
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu

# map an integer to a word
def word_for_id(integer, tokenizer):
for word, index in tokenizer.word_index.items():
if index == integer:
return word
return None

# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
# seed the generation process
in_text = &#39;startseq&#39;
# iterate over the whole length of the sequence
for i in range(max_length):
# integer encode input sequence
sequence = tokenizer.texts_to_sequences([in_text])[0]
# pad input
sequence = pad_sequences([sequence], maxlen=max_length)
# predict next word
yhat = model.predict([photo,sequence], verbose=0)
# convert probability to integer
yhat = argmax(yhat)
# map integer to word

21
word = word_for_id(yhat, tokenizer)
# stop if we cannot map the word
if word is None: break
# append as input for generating the next word
in_text += &#39; &#39; + word
# load doc into memory
def load_doc(filename):
# open the file as read only
file = open(filename, &#39;r&#39;)
# read all text
text = file.read()
# close the file
file.close() return text
# load a pre-defined list of photo identifiers
def load_set(filename): doc = load_doc(filename) dataset = list()
# process line by line
for line in doc.split(&#39;\n&#39;):
# skip empty lines if len(line) &lt; 1: continue
# get the imageidentifier identifier = line.split(&#39;.&#39;)[0] dataset.append(identifier)
return set(dataset)
# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
# load document
doc = load_doc(filename)
descriptions = dict()
for line in doc.split(&#39;\n&#39;):
# split line by white space

22

tokens = line.split()
# split id from description
image_id, image_desc = tokens[0], tokens[1:]
# skip images not in the set
if image_id in dataset:
# create list
if image_id not in
descriptions:
descriptions[image_id] =
list()
# wrap description in tokens
desc = &#39;startseq &#39; + &#39; &#39;.join(image_desc) + &#39; endseq&#39;
# store
descriptions[image_id].append(desc)
return descriptions

# load photo features
def load_photo_features(filename, dataset):
# load all features
all_features = load(open(filename, &#39;rb&#39;))
# filter features
features = {k: all_features[k] for k in dataset}
return features

# covert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
all_desc = list()

23

for key in descriptions.keys():
[all_desc.append(d) for d in
descriptions[key]]
return all_desc

# fit a tokenizer given caption descriptions
def
create_tokenizer(description
s): lines =
to_lines(descriptions)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
return tokenizer
# evaluate the skill of the model
def evaluate_model(model, descriptions, photos, tokenizer,
max_length): actual, predicted = list(), list()
# step over the whole set
for key, desc_list in descriptions.items():
# generate description
yhat = generate_desc(model, tokenizer, photos[key], max_length)
# store actual and predicted
references = [d.split() for d in desc_list]
actual.append(references)
predicted.append(yhat.split())
# calculate BLEU score
print(&#39;BLEU-1: %f&#39; % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
print(&#39;BLEU-2: %f&#39; % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
print(&#39;BLEU-3: %f&#39; % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
print(&#39;BLEU-4: %f&#39; % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25,
0.25)))
def load_doc(filename):
# open the file as read only

24

file = open(filename, &#39;r&#39;)
# read all text
text = file.read()
# close the
file
file.close()
return text

# load a pre-defined list of photo identifiers
def load_set(filename): doc = load_doc(filename) dataset = list()
# process line by line
for line in doc.split(&#39;\n&#39;):
# skip empty lines
if len(line) &lt; 1: continue
# get the image identifier
identifier = line.split(&#39;.&#39;)[0] dataset.append(identifier)
return set(dataset)

# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
# load document
doc = load_doc(filename)
descriptions = dict()
for line in doc.split(&#39;\n&#39;):
# split line by white space
tokens = line.split()
# split id from description
image_id, image_desc = tokens[0], tokens[1:]
# skip images not in the set

25

if image_id in dataset:
# create list
if image_id not in
descriptions:
descriptions[image_id] =
list()
# wrap description in tokens
desc = &#39;startseq &#39; + &#39; &#39;.join(image_desc) + &#39; endseq&#39;
# store
descriptions[image_id].append(desc)
return descriptions

# covert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
all_desc = list()
for key in descriptions.keys():
[all_desc.append(d) for d in
descriptions[key]]
return all_desc

# fit a tokenizer given caption descriptions
def
create_tokenizer(description
s): lines =
to_lines(descriptions)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
return tokenizer

# load training dataset (6K)
filename =
&#39;train.txt&#39; train =
load_set(filename)

26

print(&#39;Dataset: %d&#39; % len(train))
# descriptions
train_descriptions = load_clean_descriptions(&#39;descriptions.txt&#39;, train)
print(&#39;Descriptions: train=%d&#39; % len(train_descriptions))
# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
# save the tokenizer
dump(tokenizer, open(&#39;tokenizer.pkl&#39;, &#39;wb&#39;))
import warnings
warnings.filterwarnings(&quot;ign
ore&quot;) from pickle import
load
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import
img_to_array from keras.applications.vgg16
import preprocess_input from keras.models
import Model
from keras.models import load_model

# extract features from each photo in the directory
def extract_features(filename):
# load the model
model = VGG16()
# re-structure the model
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
# load the photo
image = load_img(filename, target_size=(224, 224))

27
# convert the image pixels to a numpy array
image = img_to_array(image)
# reshape data for the model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# prepare the image for the VGG model
image = preprocess_input(image)
# get features
feature = model.predict(image, verbose=0)
return feature
# map an integer to a word
def word_for_id(integer, tokenizer):
for word, index in tokenizer.word_index.items():
if index == integer:
return word
return None

# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
# seed the generation process
in_text = &#39;startseq&#39;
# iterate over the whole length of the sequence
for i in range(max_length):
# integer encode input sequence
sequence = tokenizer.texts_to_sequences([in_text])[0]
# pad input
sequence = pad_sequences([sequence], maxlen=max_length)
# predict next word

28
yhat = model.predict([photo,sequence], verbose=0)
# convert probability to integer
yhat = argmax(yhat)
# map integer to word
word = word_for_id(yhat, tokenizer)
# stop if we cannot map the word
if word is
None:
break
# append as input for generating the next word
in_text += &#39; &#39; + word
# stop if we predict the end of the sequence
if word == &#39;endseq&#39;:
break
return in_text

# load the tokenizer
tokenizer = load(open(&#39;tokenizer.pkl&#39;, &#39;rb&#39;))
# pre-define the max sequence length (from training)
max_length = 33
# load the model
model = load_model(&#39;model_1.h5&#39;)
# load and prepare the photograph
photo = extract_features(&#39;tig.jpg&#39;)
# generate description
description = generate_desc(model, tokenizer, photo, max_length)
print(description)
