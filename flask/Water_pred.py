from flask import Flask, request, jsonify
import os
from flask_cors import CORS

import pandas as pd
import numpy as np

import json
from tensorflow import keras

app = Flask(__name__)
CORS(app)


model = keras.models.load_model('model-water2.h5')
model.summary()

# 수위데이터 까지 입력하는 모델
@app.route('/aws/update', methods=['POST'])
def receive_and_process():
    try:
        data1 = request.files['file1']
        data2 = request.files['file2']
        data3 = request.files['file3']
        data4 = request.files['file4']
        
        
        cnt = 3
        file_names = [data1, data2, data3]
        file_names = [data1, data2]
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
            order = ['fcstDate', 'fcstTime', 'T1H', 'VEC', 'WSD', 'RN1', 'REH']  # 원하는 순서로 열 이름을 나열
            df_pivot = df_pivot[order]
            df_pivot['RN1'] = df_pivot['RN1'].str.replace('mm', '')
            df_pivot['RN1'] = df_pivot['RN1'].replace('강수없음', 0)
            # 여러 개의 컬럼을 숫자로 변환
            columns_to_convert = ['T1H', 'VEC', 'WSD', 'RN1', 'REH']
            df_pivot[columns_to_convert] = df_pivot[columns_to_convert].apply(pd.to_numeric, errors='coerce')
            df_pivot.to_csv(f'fcst{i}.csv', index=False)


        data = json.load(data4)
        levels = data['list']
        df = pd.DataFrame(levels)
        df['wl'] = pd.to_numeric(df['wl'], errors='coerce')
        df = df['wl'][-7:]
        df.to_csv('water.csv', index=False)


        seq_len = 3

        fcst_files = ['fcst0.csv', 'fcst1.csv', 'fcst2.csv']
        fcst_files = ['fcst0.csv', 'fcst1.csv']
        inputs = []
        for i in range(cnt):
            df = pd.read_csv(fcst_files[i])[:-3]
            drop_cols = ['fcstDate', 'fcstTime']
            df = df.drop(drop_cols, axis=1)
            inputs.append(np.expand_dims(df.to_numpy(), axis=0))
        water = pd.read_csv('water.csv').to_numpy()
        past = []
        for i in range(seq_len):
            wl = water[i: i + 5]
            wl = [sublist[0] for sublist in wl]
            past.append(wl)
        inputs.append(np.expand_dims(past, axis=0))
        
        y_result = model.predict(inputs)
        print(inputs)
        print(y_result)
        predictions_list = y_result.tolist()
        
        
        return jsonify({'result': predictions_list[0]}), 200
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
