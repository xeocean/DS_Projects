import streamlit as st
import pandas as pd
import folium
from folium.plugins import TimestampedGeoJson

# streamlit run .\bonus.py
st.set_page_config(page_title="City Life", layout="wide")
DATA_PATH = "data/taxi_locations.csv"

@st.cache_data
def load(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().replace("  ", " ") for c in df.columns]
    for c in ["Trip Start Timestamp", "Trip End Timestamp"]:
        df[c] = pd.to_datetime(df[c], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    df["Fare"] = pd.to_numeric(df["Fare"], errors="coerce")
    return df.dropna(subset=["Trip Start Timestamp", "Trip End Timestamp"])


df = load(DATA_PATH)


def show(m):
    st.iframe(m.get_root().render(), height=640)


def line(pu, do, t0, t1, color, popup=None):
    p = {"times": [t0.isoformat(), t1.isoformat()], "style": {"color": color, "weight": 2, "opacity": .7}}
    if popup: p["popup"] = popup
    return {"type": "Feature", "geometry": {"type": "LineString", "coordinates": [pu, do]}, "properties": p}


def point(coords, t, color, radius, popup=None):
    p = {"times": [t.isoformat()], "icon": "circle",
         "iconstyle": {"fillColor": color, "color": color, "fillOpacity": 1, "radius": radius}}
    if popup: p["popup"] = popup
    return {"type": "Feature", "geometry": {"type": "Point", "coordinates": coords}, "properties": p}


def tgj(feats, duration, m):
    TimestampedGeoJson({"type": "FeatureCollection", "features": feats},
                       period="PT5M", duration=duration, transition_time=100,
                       add_last_point=True, loop=False, auto_play=False).add_to(m)
    return m


def pu_do(r):
    return ([r["Pickup Centroid Longitude"], r["Pickup Centroid Latitude"]],
            [r["Dropoff Centroid Longitude"], r["Dropoff Centroid Latitude"]])


st.title("City Life — визуализации")
view = st.sidebar.radio("Визуализация", ["День города", "День водителя"])

if view == "День города":
    date = st.date_input("Дата", pd.to_datetime("2019-05-16"))
    max_trips = st.slider("Макс. поездок (производительность)", 500, 8000, 4000, 500)
    day = df[df["Trip Start Timestamp"].dt.date == date]
    st.caption(f"Поездок за день: {len(day)}")
    if len(day) > max_trips:
        day = day.sample(max_trips, random_state=42)
    feats = [line(*pu_do(r), r["Trip Start Timestamp"], r["Trip End Timestamp"], "#e6194b")
             for _, r in day.iterrows()]
    m = folium.Map([41.88, -87.63], zoom_start=11, tiles="cartodbpositron")
    show(tgj(feats, "PT10M", m))  # duration задан -> поездки гаснут (нейроны)

elif view == "День водителя":
    drivers = df["Taxi ID"].value_counts().head(50).index.tolist()
    drv = st.selectbox("Водитель (топ-50 по числу поездок)", drivers)
    date = st.date_input("Дата", pd.to_datetime("2019-05-31"))
    day = df[(df["Taxi ID"] == drv) & (df["Trip Start Timestamp"].dt.date == date)] \
        .sort_values("Trip End Timestamp")
    if day.empty:
        st.warning("У этого водителя нет поездок в выбранный день.")
    else:
        day = day.reset_index(drop=True);
        day["cum"] = day["Fare"].cumsum()
        feats = []
        for i, (_, r) in enumerate(day.iterrows()):
            pu, do = pu_do(r)
            feats.append(line(pu, do, r["Trip Start Timestamp"], r["Trip End Timestamp"], "#1f77b4",
                              f"Поездка #{i + 1}: ${r['Fare']:.2f} · всего ${r['cum']:.2f}"))
            feats.append(point(pu, r["Trip Start Timestamp"], "green", 5))
            feats.append(point(do, r["Trip End Timestamp"], "red", 6, f"Всего ${r['cum']:.2f}"))
        st.metric("Заработано за день", f"${day['Fare'].sum():.2f}")
        m = folium.Map([day["Pickup Centroid Latitude"].mean(), day["Pickup Centroid Longitude"].mean()],
                       zoom_start=11, tiles="cartodbpositron")
        show(tgj(feats, None, m))  # duration=None -> поездки НЕ исчезают
