from flask import Flask,request
import flask
import tensorflow as tf
from transformers import BertTokenizer,TFBertModel
import numpy as np
import pandas as pd

app=Flask(__name__)

# Defining Constants
MAX_LENGTH=85

# Preprocessing Data For Bert
tokenizer=BertTokenizer.from_pretrained("bert-base-cased",do_lower_case=False)
def tokenize(sentence,tokenizer,max_length):
	input_ids,attention_mask,token_type_ids=[],[],[]
	inputs=tokenizer.encode_plus(sentence,max_length=max_length,add_special_tokens=True,pad_to_max_length=True,truncation=True)
	input_ids.append(inputs["input_ids"])
	attention_mask.append(inputs["attention_mask"])
	token_type_ids.append(inputs["token_type_ids"])

	return np.array(input_ids),np.array(attention_mask),np.array(token_type_ids)  

def bert_binary():

	bert_encoder=TFBertModel.from_pretrained("bert-base-cased")

	for layer in bert_encoder.layers:
		layer.trainable=False

	input_ids=tf.keras.layers.Input(shape=(MAX_LENGTH,),dtype=tf.int32,name="input_word_ids")

	attention_mask_ids=tf.keras.layers.Input(shape=(MAX_LENGTH,),dtype=tf.int32,name="attention_mask_ids")

	token_type_ids=tf.keras.layers.Input(shape=(MAX_LENGTH,),dtype=tf.int32,name="token_type_ids")

	embeddings=bert_encoder([input_ids,attention_mask_ids,token_type_ids])[0]

	out=tf.keras.layers.Dense(1,activation='sigmoid')(embeddings[:,0,:])

	model=tf.keras.models.Model(inputs=[input_ids,attention_mask_ids,token_type_ids],outputs=out)

	return model


def bert_multiclass():

	bert_encoder=TFBertModel.from_pretrained("bert-base-cased")

	for layer in bert_encoder.layers:
		layer.trainable=False

	input_ids=tf.keras.layers.Input(shape=(MAX_LENGTH,),dtype=tf.int32,name="input_word_ids")

	attention_mask_ids=tf.keras.layers.Input(shape=(MAX_LENGTH,),dtype=tf.int32,name="attention_mask_ids")

	token_type_ids=tf.keras.layers.Input(shape=(MAX_LENGTH,),dtype=tf.int32,name="token_type_ids")

	embeddings=bert_encoder([input_ids,attention_mask_ids,token_type_ids])[0]

	out=tf.keras.layers.Dense(4,activation='softmax')(embeddings[:,0,:])

	model=tf.keras.models.Model(inputs=[input_ids,attention_mask_ids,token_type_ids],outputs=out)

	return model


# Loading Binary Model
model_binary=bert_binary()
model_binary.load_weights("model weights/model_binary.h5")

# Loading Multiclass Model
model_multiclass=bert_multiclass()
model_multiclass.load_weights("model weights/model_multiclass.h5")


@app.route("/welcome",methods=['GET','POST'])
def welcome():
	response={
	"welcome":"Welcome to the Tweep Fake Text Classifcation API"
	}
	return flask.jsonify(response)

@app.route("/predict/binary",methods=["GET","POST"])
def predict():
	response={}
	text=request.json
	text=text["input_text"]

	input_ids,attention_mask,token_type_ids=tokenize(text,tokenizer,MAX_LENGTH)

	predictions=model_binary.predict([input_ids,attention_mask,token_type_ids])
	predictions=np.where(predictions>0.5,1,0)[0]

	if predictions==1:

		response["response"]={
		"prediction":"Bot"
		}

		return flask.jsonify(response)

	else:

		response["response"]={
		"prediction":"Human"
		}

		return flask.jsonify(response)

@app.route("/predict/multiclass",methods=["GET","POST"])
def predict_multiclass():
	response={}
	text=request.json
	text=text["input_text"]

	input_ids,attention_mask,token_type_ids=tokenize(text,tokenizer,MAX_LENGTH)

	predictions=model_multiclass.predict([input_ids,attention_mask,token_type_ids])
	predictions=np.argmax(predictions,axis=1)

	if predictions==0:

		response["response"]={
		"prediction":"Human"
		}

		return flask.jsonify(response)

	elif predictions==1:

		response["response"]={
		"prediction":"gpt2"
		}

		return flask.jsonify(response)


	elif predictions==2:
		response["response"]={
		"prediction":"rnn"
		}

		return flask.jsonify(response)
		
	else:
		response["response"]={
		"prediction":"others"
		}

		return flask.jsonify(response)
		

if __name__=="__main__":
	app.run(debug=True)
