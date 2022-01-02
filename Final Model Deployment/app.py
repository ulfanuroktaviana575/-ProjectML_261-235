import os
import time
import numpy as np
from PIL import Image
from pathlib import Path
from random import shuffle
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from flask import Flask, request, render_template, redirect, url_for

app = Flask(__name__)

@app.after_request
def add_header(r): #setting cache
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

@app.route('/', methods=['GET', 'POST'])
def index():
    db_model_dict, random_test_image_dir, random_test_image_name = init_db_values()
    model_count = 0

    if request.method == 'POST':
        #  ambil data dari request
        choosen_model = request.form['choosen_model']
        compared_model = request.form.getlist('compare_model')

        try:
            # masuk kesini jika query dari upload
            file = request.files['query_img']
            img = Image.open(file.stream)
            query_img_name = file.filename
            
        except:
            # masuk kesini jika query dari sample
            filename = request.form.get('query_img')
            img = Image.open(filename)
            query_img_name = filename.split('/')[-1]
        
        true_label = query_img_name.split('-')[0].title()

        # simpan sementara query
        uploaded_img_path = "static/uploaded/temp" + os.path.splitext(query_img_name)[1]
        img.save(uploaded_img_path)

        model_count = len(compared_model)
        # choose model, jika jumlah model dipilih = satu
        if model_count == 1:
            # ambil hasil prediksi berupa label, keyakinan, dan waktu
            pred_label, pred_conf, pred_time = pred_choosen_model(db_model_dict, choosen_model, uploaded_img_path)
            
            uploaded_img_path = '../' + uploaded_img_path

            return render_template(
                'index.html',
                test_images=random_test_image_dir,
                test_images_names=random_test_image_name,
                model_name=choosen_model,
                model_count=model_count,
                query_path=uploaded_img_path,
                query_image_name=query_img_name,
                true_label=true_label,
                pred_label=pred_label,
                pred_conf=pred_conf,
                pred_time=pred_time,
                )
        
        # compare model, jika model > 1
        else:
            # prediksi menggunakan semua model yg dipilih
            compared_model_result_dict = dict()
            for model in compared_model:
                model_pred_result = pred_choosen_model(db_model_dict, model, uploaded_img_path)
                compared_model_result_dict[model] = model_pred_result

            pred_time_list = [value[2] for value in compared_model_result_dict.values()]

            uploaded_img_path = '../' + uploaded_img_path

            return render_template(
                'index.html',
                test_images=random_test_image_dir,
                test_images_names=random_test_image_name,
                model_name=choosen_model,
                model_count=model_count,
                query_path=uploaded_img_path,
                query_image_name=query_img_name,
                true_label=true_label,
                compared_result=compared_model_result_dict,
                pred_time_list=pred_time_list,
                )
    
    else:
        return render_template(
            'index.html',
            test_images=random_test_image_dir,
            test_images_names=random_test_image_name,
            model_count=model_count,
            )

# fungsi prediksi
def pred_choosen_model(db_model_dict, choosen_model, uploaded_img_path):
    model = load_model(db_model_dict[choosen_model])

    #img_target_size = tuple(model.layers[0].input_shape[1:3])

    img = load_img(uploaded_img_path, target_size=(150,150))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = np.vstack([img])

    start = time.time()
    pred_result = model.predict(img)
    pred_time = round(time.time() - start, 4)

    output_layer_activation = str(model.layers[-1].activation).split(' ')[1]
    if output_layer_activation == 'sigmoid':
        pred_conf = '-'
        pred_value = (pred_result[0][0] > 0.5).astype(int)
    
    elif output_layer_activation == 'softmax':
        pred_conf = round(np.max(pred_result) * 100, 2)
        pred_value = np.argmax(pred_result)
    
    else:
        pred_conf = 9000

    pred_label = 'covid' if pred_value == 0 else 'No_findings' if pred_value == 1 else 'normal' if pred_value == 2 else 'pneumonia_bacterial' if pred_value == 3 else 'pneumonia_viral' if pred_value == 4 else 'miss'

    return pred_label, pred_conf, pred_time

# fungsi init data
def init_db_values():
    db_model_dir = './static/db_model'
    db_test_image_dir = './static/db_test_image'

    db_model_dict = {
        'cnn_model':db_model_dir + '/cov-cnn.h5',
        'vgg19_model':db_model_dir + '/cov-vgg19.h5',
        'resnet-inception_model':db_model_dir + '/cov-resnet-inception.h5',
        # 'cxr_modul_5':db_model_dir + '/cxr_modul_5.h5',
    }

    db_test_image_paths = list()
    for img_path in sorted(Path(db_test_image_dir).glob("*.png")):
        db_test_image_paths.append(img_path)
    
    shuffle(db_test_image_paths)
    random_test_image_dir = db_test_image_paths[:16]
    random_test_image_name = [img_dir.name for img_dir in random_test_image_dir]

    return db_model_dict, random_test_image_dir, random_test_image_name

if __name__ == '__main__':
    app.run(debug=True)
