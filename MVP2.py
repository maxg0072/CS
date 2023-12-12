import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import folium
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim

# Funktion, um ähnliche Immobilien zu finden
def find_similar_properties(input_zip, input_rooms, input_size, data, threshold=10):
    similar_properties = data[
        (data['area_code'] == input_zip) &
        (data['Rooms'].between(input_rooms - 1, input_rooms + 1)) &
        (data['Size_m2'].between(input_size - threshold, input_size + threshold))
    ]
    return similar_properties

# Backend Code: Data Preprocessing and Model Training
def preprocess_and_train():
    # Load the dataset (replace with your actual file path)
    real_estate_data = pd.read_excel('real-estate-scraped-data.xlsx')

    # Data Preprocessing
    # Define the function to split 'Col3'
    def split_column(row):
        parts = row.split(' • ')
        property_type = parts[0]
        rooms = parts[1] if len(parts) > 1 else None
        size_m2 = parts[2] if len(parts) > 2 else None
        return {'Property_Type': property_type, 'Rooms': rooms, 'Size_m2': size_m2}

    # Apply the function to each row in 'Col3' and create a new DataFrame
    split_data = real_estate_data['Col3'].apply(split_column).apply(pd.Series)

    # Cleaning and renaming columns
    real_estate_data['area_code'] = real_estate_data['Col4'].str.extract(r'\b(\d{4})\b')

    # Extracting numeric values from 'Col5' and 'Col6'
    real_estate_data['price_per_month'] = real_estate_data['Col5'].str.extract(r'(\d+[\’\']?\d*)')[0].str.replace("’", "").str.replace("'", "").str.strip()
    real_estate_data['price_per_m2_per_year'] = real_estate_data['Col6'].str.extract(r'(\d+[\’\']?\d*)')[0].str.replace("’", "").str.replace("'", "").str.strip()

    # Remove 'Zi.' from 'Rooms' and 'm²' from 'Size_m2', with checks for non-string data
    split_data['Rooms'] = split_data['Rooms'].str.replace(' Zi.', '').str.strip() if split_data['Rooms'].dtype == "object" else split_data['Rooms']
    split_data['Size_m2'] = split_data['Size_m2'].str.replace(' m²', '').str.strip() if split_data['Size_m2'].dtype == "object" else split_data['Size_m2']

    # Concatenate the new DataFrame with the original one, now including cleaned columns
    real_estate_data = pd.concat([split_data, real_estate_data.drop(columns=['Col3', 'Col4', 'Col5', 'Col6'])], axis=1)

    # Rearrange columns
    new_columns = ['Property_Type', 'Rooms', 'Size_m2', 'area_code', 'price_per_month', 'price_per_m2_per_year']
    real_estate_data = real_estate_data[new_columns]

    real_estate_data.dropna(inplace=True)

    # Convert columns to numeric as necessary
    real_estate_data['Rooms'] = pd.to_numeric(real_estate_data['Rooms'], errors='coerce')
    real_estate_data['Size_m2'] = pd.to_numeric(real_estate_data['Size_m2'], errors='coerce')
    real_estate_data['area_code'] = pd.to_numeric(real_estate_data['area_code'], errors='coerce')
    real_estate_data['price_per_month'] = pd.to_numeric(real_estate_data['price_per_month'], errors='coerce')

    # Drop any rows with NaN values
    real_estate_data.dropna(inplace=True)

    # Selecting features and target for the model
    X = real_estate_data[['Rooms', 'Size_m2', 'area_code']]  # Example features
    y = real_estate_data['price_per_month']  # Target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, real_estate_data

# Extraktion der Postleitzahl
def extract_zip_from_address(address):
    valid_st_gallen_zip_codes = ['9000', '9001', '9004', '9006', '9007', '9008', '9010', '9011', '9012', '9013', '9014', '9015', '9016', '9020', '9021', '9023', '9024', '9026', '9027', '9028', '9029']
    geolocator = Nominatim(user_agent="http")
    location = geolocator.geocode(address + ", St. Gallen", country_codes='CH')
    if location:
        address_components = location.raw.get('display_name', '').split(',')
        for component in address_components:
            if component.strip() in valid_st_gallen_zip_codes:
                return component.strip()
    return None


# Funktion zur Preisvorhersage
def predict_price(size_m2, extracted_zip_code, rooms, model):
    try:
        area_code = int(extracted_zip_code)
    except ValueError:
        st.error("Bitte geben Sie eine gültige Postleitzahl ein.")
        return None

    input_features = pd.DataFrame({
        'Rooms': [rooms],
        'Size_m2': [size_m2],
        'area_code': [area_code]
    })
    predicted_price = model.predict(input_features)
    return predicted_price[0]

# Function to get latitude and longitude from an address
def get_lat_lon_from_zip(address):
    geolocator = Nominatim(user_agent="http")
    location = geolocator.geocode(address)
    if location:
        return location.latitude, location.longitude
    else:
        return None, None


# Streamlit UI
st.title("Rental Price Prediction")

# Modell und Daten laden
model, real_estate_data = preprocess_and_train()

# Input für Adresse oder Postleitzahl
address_input = st.text_input("Enter an address or zip code in St. Gallen:")

# Extrahieren der Postleitzahl aus der Eingabe
extracted_zip_code = extract_zip_from_address(address_input)

# Display the map based on the address or zip code
if address_input:
    lat, lon = get_lat_lon_from_zip(address_input)
    if lat and lon:
        map = folium.Map(location=[lat, lon], zoom_start=16)
        folium.Marker([lat, lon]).add_to(map)
        folium_static(map)
    else:
        st.write("Invalid zip code or location not found.")


# Input für die Anzahl der Zimmer und Größe in Quadratmetern
rooms = st.number_input("Enter the number of rooms", min_value=1, max_value=10)
size_m2 = st.number_input("Enter the size in square meters", min_value=0)

# Predict Rental Price button and functionality
if st.button('Predict Rental Price'):
    if extracted_zip_code:
        predicted_price = predict_price(size_m2, extracted_zip_code, rooms, model)
        if predicted_price is not None:
            st.write(f"The predicted price for the apartment is CHF {predicted_price:.2f}")

            # Ähnliche Immobilien finden und anzeigen
            similar_properties = find_similar_properties(extracted_zip_code, rooms, size_m2, real_estate_data)
            if not similar_properties.empty:
                st.write("Ähnliche Immobilien:")
                st.dataframe(similar_properties[['Property_Type', 'Rooms', 'Size_m2', 'area_code']])
            else:
                st.write("Keine ähnlichen Immobilien gefunden.")
        else:
            st.write("Unable to predict price. Please check your inputs.")
    else:
        st.write("Please enter a valid address or zip code in St. Gallen.")
