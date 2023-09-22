from flask import Flask, request, jsonify
import os
from flask_cors import CORS

import pandas as pd
import numpy as np

import json
from tensorflow import keras
from pickle import load
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
CORS(app)

# 과거데이터로 예측 --> 0~6시간 넣으면 7시부터 예측
new_model = keras.models.load_model('model_test(3)_cnn_300.h5')
new_model.summary()

load_scaler = load(open('3_cnn_300_scaler.pkl', 'rb'))
load_scaler_water = load(open('3_cnn_300_water_scaler.pkl', 'rb'))

# 실시간데이터로 예측 --> 0~6시간 넣으면 0시부터 예측   
# new_model = keras.models.load_model('model_test_re300.h5')
# new_model.summary()

# load_scaler = load(open('3_re300_scaler.pkl', 'rb'))

@app.route('/aws/update', methods=['POST'])
def receive_and_process():
    try:
        data1 = request.files['file1']
        data2 = request.files['file2']
        data3 = request.files['file3']
        
        # 파일 저장 
        # filepath = r"C:\Users\user\Desktop\main_project\flask\test\\"
        # file_path = os.path.join(filepath, data1.filename)
        # data1.save(file_path)
        # file_path = os.path.join(filepath, data2.filename)
        # data2.save(file_path)
        # file_path = os.path.join(filepath, data3.filename)
        # data3.save(file_path)
        
        # print("저장 완료:", data1)
        # print("저장 완료:", data2)
        # print("저장 완료:", data3)
        
        print('시작')
        # 받은 데이터를 모델에 적용하고 결과 값을 계산
        fcst = read_data(data1, data2, data3)
        result = apply_model(fcst)
        # result = "Success"
        
        print(result)
        # 결과 값을 클라이언트(서버)로 보냄
        return jsonify({'result': result}), 200
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

def read_data(file1, file2, file3):
    cnt = 3
    file_names = [file1, file2, file3]
    fcst_list = []
    for i in range(cnt):
        print('파일전처리')
        
        data = json.load(file_names[i])
        items = data['response']['body']['items']['item']
        df = pd.DataFrame(items)
        columns = ['fcstDate', 'fcstTime', 'category', 'fcstValue']
        df = df[columns]
        categories = ['T1H', 'RN1', 'REH', 'VEC', 'WSD']
        df = df[df['category'].isin(categories)]

        # 'category' 열의 값을 열 이름으로 변환하고 'fcstValue' 열의 값을 해당 열에 매핑
        df_pivot = df.pivot(index=['fcstDate', 'fcstTime'], columns='category', values='fcstValue').reset_index()
        order = ['fcstDate', 'fcstTime', 'T1H', 'VEC', 'WSD', 'RN1', 'REH']
        df_pivot = df_pivot[order]
        df_pivot = df_pivot[1:]
        
        file_name = f'fcst{i}.csv'
        df_pivot.to_csv(file_name, index=False)
        fcst_data = pd.read_csv(file_name)
        fcst_list.append(preprocess_data(fcst_data))
        
    return fcst_list
    
def preprocess_data(fcst_data):
    print('전처리시작')
    # 데이터 전처리
    data = fcst_data[['fcstDate','fcstTime','T1H', 'VEC', 'WSD', 'RN1', 'REH']]
    print(fcst_data)
    data['RN1'] = data['RN1'].str.replace('mm', '', regex=False)
    data['RN1'] = data['RN1'].replace('강수없음', 0.0)
    columns_to_convert = ['T1H', 'VEC', 'WSD', 'RN1', 'REH']
    data[columns_to_convert] = data[columns_to_convert].apply(pd.to_numeric, errors='coerce')
    data['RN1'] = pd.to_numeric(data['RN1'], errors='coerce')

    features = ['T1H', 'VEC', 'WSD', 'RN1', 'REH']
    data_value = data[features].values
    
    # 정규화
    fcst_z = load_scaler.fit_transform(data_value)
    print("정규화성공")
    
    
    time_steps = 3
    new_data = []
    for i in range(len(fcst_z) - time_steps + 1):
        batch = fcst_z[i:i+time_steps]
        new_data.append(batch)

    reshaped_fcst = np.array(new_data)
    print("차원변환성공")
    
    return reshaped_fcst

def apply_model(input_fcst):
    print('모델테스트')
    print(input_fcst)
    result = new_model.predict(input_fcst)
    print(result)
    predictions_scaler = load_scaler_water.inverse_transform(result)
    print(predictions_scaler)
    
    predictions_list = predictions_scaler.tolist()
    
    return [sublist[0] for sublist in predictions_list]

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
