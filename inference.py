import pickle
import pandas as pd

with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

data = {
    'no_of_adults': [2],
    'no_of_children': [0],
    'no_of_weekend_nights': [1],
    'no_of_week_nights': [2],
    'type_of_meal_plan': [1],  
    'required_car_parking_space': [0],
    'room_type_reserved': [0], 
    'lead_time': [100],
    'arrival_year': [2018],
    'arrival_month': [7],
    'arrival_date': [10],
    'market_segment_type': [1],  
    'repeated_guest': [0],
    'no_of_previous_cancellations': [0],
    'no_of_previous_bookings_not_canceled': [0],
    'avg_price_per_room': [100.0],
    'no_of_special_requests': [0]
}

df = pd.DataFrame(data)
pred = model.predict(df)
print("Canceled" if pred[0] == 1 else "Not Canceled")
