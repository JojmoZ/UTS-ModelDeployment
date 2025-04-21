
import streamlit as st
import pickle
import pandas as pd

with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

meal_plan_mapping = {
    "Not Selected": 0,
    "Meal Plan 1": 1,
    "Meal Plan 2": 2,
    "Meal Plan 3": 3
}
room_type_mapping = {
    "Room_Type 1": 0,
    "Room_Type 2": 1,
    "Room_Type 3": 2,
    "Room_Type 4": 3,
    "Room_Type 5": 4,
    "Room_Type 6": 5,
    "Room_Type 7": 6
}
market_segment_mapping = {
    "Online": 0,
    "Offline": 1,
    "Corporate": 2,
    "Complementary": 3,
    "Aviation": 4
}
parking_space_mapping = {
    "tidak":0,
    "iya": 1
}
st.title("2702259632 MODEL DEPLOYMENT FOR HOTEL")

st.markdown("""
Masukkan detail pemesanan hotel di bawah ini untuk memprediksi apakah pemesanan akan **dibatalkan (Canceled)** atau **tidak (Not Canceled)**.
""")

tab1, tab2, tab3 = st.tabs(["Manual Input", "Test Case 1", "Test Case 2"])

with tab1:
    st.subheader("Manual Input")
    with st.form("manual_form"):
        no_of_adults = st.number_input("Jumlah Dewasa", 0, 10, 2)
        no_of_children = st.number_input("Jumlah Anak", 0, 10, 0)
        no_of_weekend_nights = st.number_input("Malam Akhir Pekan", 0, 10, 1)
        no_of_week_nights = st.number_input("Malam Hari Kerja", 0, 15, 2)
        selected_meal_plan = st.selectbox("Meal Plan", list(meal_plan_mapping.keys()))
        selected_parking = st.selectbox("Perlu Parkir?", list(parking_space_mapping.keys()))
        required_car_parking_space = parking_space_mapping[selected_parking]
        selected_room_type = st.selectbox("Tipe Kamar", list(room_type_mapping.keys()))
        lead_time = st.slider("Lead Time (hari)", 0, 500, 100)
        arrival_year = st.selectbox("Tahun Kedatangan", [2017, 2018], index=1)
        arrival_month = st.selectbox("Bulan Kedatangan", list(range(1, 13)), index=6)
        arrival_date = st.selectbox("Tanggal Kedatangan", list(range(1, 32)), index=9)
        selected_market_segment = st.selectbox("Segment Pasar", list(market_segment_mapping.keys()))
        repeated_guest = st.selectbox("Pelanggan Pernah Booking?", [0, 1])
        no_of_previous_cancellations = st.number_input("Jumlah Pembatalan Sebelumnya", 0, 20, 0)
        no_of_previous_bookings_not_canceled = st.number_input("Jumlah Booking Sukses Sebelumnya", 0, 60, 0)
        avg_price_per_room = st.number_input("Harga Rata-Rata per Kamar", 0.0, 600.0, 100.0)
        no_of_special_requests = st.number_input("Jumlah Permintaan Khusus", 0, 5, 1)
        submit = st.form_submit_button("Prediksi")

    if submit:
        input_df = pd.DataFrame([{
            "no_of_adults": no_of_adults,
            "no_of_children": no_of_children,
            "no_of_weekend_nights": no_of_weekend_nights,
            "no_of_week_nights": no_of_week_nights,
            "type_of_meal_plan": meal_plan_mapping[selected_meal_plan],
            "required_car_parking_space": required_car_parking_space,
            "room_type_reserved": room_type_mapping[selected_room_type],
            "lead_time": lead_time,
            "arrival_year": arrival_year,
            "arrival_month": arrival_month,
            "arrival_date": arrival_date,
            "market_segment_type": market_segment_mapping[selected_market_segment],
            "repeated_guest": repeated_guest,
            "no_of_previous_cancellations": no_of_previous_cancellations,
            "no_of_previous_bookings_not_canceled": no_of_previous_bookings_not_canceled,
            "avg_price_per_room": avg_price_per_room,
            "no_of_special_requests": no_of_special_requests
        }])
        prediction = model.predict(input_df)[0]
        result = "Booking Canceled" if prediction == 1 else "Booking Not Canceled"
        st.subheader("Hasil Prediksi:")
        st.success(result)

with tab2:
    st.subheader("Test Case 1 (Diperkirakan Not Canceled)")
    test_case_1_input = {
        "no_of_adults": 2,
        "no_of_children": 0,
        "no_of_weekend_nights": 1,
        "no_of_week_nights": 2,
        "type_of_meal_plan": 1,
        "required_car_parking_space": 0,
        "room_type_reserved": 1,
        "lead_time": 10,
        "arrival_year": 2018,
        "arrival_month": 6,
        "arrival_date": 15,
        "market_segment_type": 2,
        "repeated_guest": 0,
        "no_of_previous_cancellations": 0,
        "no_of_previous_bookings_not_canceled": 0,
        "avg_price_per_room": 120.0,
        "no_of_special_requests": 2
    }
    st.json(test_case_1_input)
    if st.button("Prediksi Test Case 1"):
        data = pd.DataFrame([test_case_1_input])
        prediction = model.predict(data)[0]
        st.success("Booking Not Canceled" if prediction == 0 else "❌ Booking Canceled")

with tab3:
    st.subheader("Test Case 2 (Diperkirakan Canceled)")
    test_case_2_input = {
        "no_of_adults": 1,
        "no_of_children": 0,
        "no_of_weekend_nights": 2,
        "no_of_week_nights": 1,
        "type_of_meal_plan": 1,
        "required_car_parking_space": 0,
        "room_type_reserved": 0,
        "lead_time": 1,
        "arrival_year": 2018,
        "arrival_month": 2,
        "arrival_date": 28,
        "market_segment_type": 0,
        "repeated_guest": 0,
        "no_of_previous_cancellations": 0,
        "no_of_previous_bookings_not_canceled": 0,
        "avg_price_per_room": 60.0,
        "no_of_special_requests": 0
    }
    st.json(test_case_2_input)
    if st.button("Prediksi Test Case 2"):
        data = pd.DataFrame([test_case_2_input])
        prediction = model.predict(data)[0]
        st.success("Booking Not Canceled" if prediction == 0 else "Booking Canceled")
