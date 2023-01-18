import io
import os                       # for working with files
import numpy as np              # for creating  neural networks
from keras.preprocessing import image
import tensorflow as tf
from PIL import Image           
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from  tensorflow import keras
import pymysql
from keras.models import load_model
from flask import Flask,request,render_template
import numpy as np
app = Flask(__name__)




#db資訊
db_settings = {
    "host": "localhost",
    "port": 3306,
    'user':"root",
    'password':"",
    "db": "db_lung"
}

#匯入影像預處理
def preprocess_image(image, target):
    # 將圖片轉為 RGB 模式方便 predict
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # 將資料進行前處理轉成 model 可以使用的 input
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    return image

    
@app.route("/image", methods=['POST'])
def predict_image():
    if 'lungImage' not in request.files:
        print("no image")
        return 'test'
    
    else:
        #匯入模組
        model=load_model("finalModel")

        #連結資料庫
        conn = pymysql.connect(**db_settings)
        #存進DB的路徑變數
        new_file_paths=""
        data = {'success': False}
        #影像獲取及影像預處理
        lungImage=request.files["lungImage"]
        lungImageI=Image.open(io.BytesIO(lungImage.read()))
        file_name=lungImage.filename
        file_path=r'.\image'
        file_paths=os.path.join(file_path,file_name)
        lungImageI.save(file_paths)
        
        #影像名稱
       
        #影像變數處理
        
        strreset=file_paths.rsplit('\\')
        for i in strreset:
            new_file_paths=new_file_paths+i+'\\\\'
        
        

        

        #回傳值處理
        image=tf.keras.preprocessing.image.load_img(file_paths,target_size=(224,224))
        image=tf.keras.preprocessing.image.img_to_array(image)
        imgae_predict=model.predict(np.array([image]))
        predict_result={}
        sen=0
        lun_label=['Lung adenocarcinoma','Lung benign tissue','Lung squamous cell carcinoma']
        for i in imgae_predict[0]:
            predict_result[lun_label[sen]]=i
            sen=sen+1
        sen=0

        #肺癌類型:0=Lung adenocarcinoma,1=Lung benign tissue,2=Lung squamous cell carcinoma
        if predict_result[0]>predict_result[1] and predict_result[0]>predict_result[2]:
            lungtp=0

        elif predict_result[1]>predict_result[0] and predict_result[1]>predict_result[2]:
            lungtp=1

        else:
            lungtp=2

        #轉成json格式
        result_js={'name':file_name,'content':[predict_result],'type':lungtp}
        result_js_str=str(result_js)

        #存DB
        cursor=conn.cursor()
        command = "INSERT INTO lung_image(Name, Path,type)VALUES( "+"'"+file_name+"'"+","+"'"+new_file_paths+"'"+","+"'"+lungtp+"'"+" )"
        cursor.execute(command)
        conn.commit()

        return result_js_str

    


if __name__=='__main__':
    app.run(host='0.0.0.0', debug = True, port=8001, threaded=True)