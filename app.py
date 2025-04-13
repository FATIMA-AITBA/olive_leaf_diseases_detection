import os
import numpy as np
from flask import Flask, render_template, request, jsonify, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

app = Flask(__name__)


model = load_model('model/model2.keras')  


class_names = ['Healthy', 'aculus_olearius', 'olive_peacock_spot']


disease_info = {
    'Healthy': {
        'cause': 'Aucune maladie détectée.',
        'recommendation': 'Continuez à maintenir vos oliviers dans des conditions saines.',
        'products': []
    },
    'aculus_olearius': {
        'cause': 'Puceron de l\'olivier, un acarien qui attaque les feuilles.',
        'recommendation': 'Utiliser des acaricides à base de soufre ou d\'huile de neem.',
        'products': [
            {"name": "Agrimek", "img": "images/Aculus.png"},
            {"name": "Horticultural oil", "img": "images/Aculus2.png"}
        ]
    },
    'olive_peacock_spot': {
        'cause': 'Champignon *Spilocaea oleaginea* qui provoque des taches sur les feuilles.',
        'recommendation': 'Appliquer des fongicides à base de cuivre ou de triazole.',
        'products': [
            {"name": "Kocide 3000", "img": "images/spot.png"},
            {"name": "Folicur", "img": "images/spot2.png"}
        ]
    }
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier envoyé'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Fichier vide'}), 400

    try:
        
        img = Image.open(file.stream).convert('RGB')
        img = img.resize((256, 256))  
        img_array = img_to_array(img)

        img_array = np.expand_dims(img_array, axis=0)  

        
        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction[0])
        predicted_class = class_names[predicted_index]
        confidence = round(100 * np.max(prediction[0]), 2)

        
        cause = disease_info[predicted_class]['cause']
        recommendation = disease_info[predicted_class]['recommendation']
        products = disease_info[predicted_class]['products']

        
        for product in products:
            product['img_url'] = url_for('static', filename=product['img'])

        return jsonify({
            'class': predicted_class,
            'confidence': f"{confidence}%",
            'cause': cause,
            'recommendation': recommendation,
            'products': products  
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
