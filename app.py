from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)

cnn = load_model('model.h5')

# cnn.make_predict_function()

def predict_label(img_path):
	i = image.load_img(img_path, target_size=(64,64))
	i = image.img_to_array(i)
	i = np.expand_dims(i, axis = 0)
	px = cnn.predict(i)
	if px[0][0] == 1:
		return 'Dog'
	else:
		return 'Cat'


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "CNN"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)