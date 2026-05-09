import os
import pandas as pd
import json
from sklearn.linear_model import LinearRegression
from flask import Flask, render_template

app = Flask(__name__)

# --- 1. NẠP DỮ LIỆU & HUẤN LUYỆN MÔ HÌNH ---
DATA_LINK = "phong_tro_cleaned_v1.csv"
JSON_LINK = "toa_do_phuong.json"

CAMPUS_UEH = {
    "Campus A": {"lat": 10.783243427186257, "lng": 106.6947473067545},
    "Campus B": {"lat": 10.761164563919763, "lng": 106.66837039484633},
    "Campus C": {"lat": 10.7732108512662, "lng": 106.67758649855381},
    "Campus E": {"lat": 10.790673699999417, "lng": 106.69921000674033},
    "Campus N": {"lat": 10.706212757415516, "lng": 106.64011424232824},
    "Campus H": {"lat": 10.796161656622639, "lng": 106.67215974232825},
    "Campus I": {"lat": 10.78326050900015, "lng": 106.69506351534353},
    "Campus V": {"lat": 10.782990766779873, "lng": 106.68559978465645},
    "Campus ISB": {"lat": 10.783265584750298, "lng": 106.695181},
}

# Đọc file dữ liệu
df = pd.read_csv(DATA_LINK)
with open(JSON_LINK, 'r', encoding='utf-8') as f:
    LOCATION_MAP = json.load(f)

# Ghép tọa độ vào dataset
df['lat'] = df['phuong'].map(lambda x: LOCATION_MAP.get(x, {}).get('lat', 10.7626))
df['lng'] = df['phuong'].map(lambda x: LOCATION_MAP.get(x, {}).get('lng', 106.6601))

# Train model hồi quy tuyến tính
feature_cols = [c for c in df.columns if c not in ['phuong', 'gia_thue', 'lat', 'lng']]
X = df[feature_cols]
y = df['gia_thue']
reg_model = LinearRegression().fit(X, y)

accuracy = round(reg_model.score(X, y) * 100, 1)
coefficients = dict(zip(feature_cols, reg_model.coef_.tolist()))
intercept = float(reg_model.intercept_)

# Chuẩn bị dữ liệu gửi sang Web
records = df[['lat', 'lng', 'gia_thue', 'dien_tich', 'khoang_cach',
              'so_nguoi_toidau', 'tu_do', 'cho_de_xe', 'has_aircon',
              'has_private_wc', 'has_kitchen', 'has_bus', 'phuong',
              'phuong_encoded']].to_dict(orient='records')

campus_options = "".join([f'<option value="{k}">{k}</option>' for k in CAMPUS_UEH.keys()])
phuong_options = "".join([
    f'<option value="{row["phuong_encoded"]}">{row["phuong"]}</option>'
    for _, row in df[['phuong', 'phuong_encoded']].drop_duplicates().sort_values('phuong').iterrows()
])

# --- 2. CUNG CẤP GIAO DIỆN ---
@app.route('/')
def index():
    return render_template('index.html',
                           records=records,
                           campus_data=CAMPUS_UEH,
                           coef=coefficients,
                           intercept=intercept,
                           accuracy=accuracy,
                           features=feature_cols,
                           campus_options=campus_options,
                           phuong_options=phuong_options)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)